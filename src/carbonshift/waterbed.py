"""The waterbed: what a carbon-aware kWh cuts when the grid sits under an emissions cap (F31).

Every carbon-aware scheduling result in this repository (and in the field) scores a
shift by the change in power-plant emissions it causes. But most cloud regions in
Europe, California, Quebec, the US Northeast, Washington and Korea sit inside a
*cap-and-trade* system. Under a binding cap the cumulative number of allowances is
fixed, so a tonne a data centre avoids is a tonne another installation may emit
later: emissions are *moved* through the allowance bank, not cut (the "waterbed
effect", environmental economics). The shift lowers the allowance price instead.

The EU ETS punctured its waterbed in 2018 with the Market Stability Reserve (MSR;
Perino 2018): when the surplus of allowances in circulation (TNAC) is large, part
of it is moved into the reserve, and reserve holdings above a threshold are
invalidated (destroyed). An avoided tonne raises the TNAC by one, so whether it is
*really* avoided is decided by the MSR rules and the TNAC path:

    TNAC > 1096 Mt   : 24% of TNAC enters the MSR      (slope 0.24 per avoided tonne)
    833 < TNAC <= 1096: TNAC - 833 enters the MSR      (slope 1: the tonne is absorbed)
    400 <= TNAC <= 833: nothing enters                 (slope 0: pure waterbed)
    TNAC < 400       : 100 Mt is released from the MSR  (slope 0 unless the MSR runs dry)

Intake set from TNAC in year y runs from September y+1 to August y+2, so 4/12 of it
lands in calendar year y+1 and 8/12 in y+2. On 1 January each year reserve holdings
above the invalidation threshold are destroyed (400 Mt under Directive 2023/959;
700 Mt under the October 2026 provisional agreement; no invalidation in the
Commission's April 2026 proposal).

This module simulates the TNAC/MSR bank mechanically (emissions exogenous, as in
the "mechanical" MSR literature) and reports the *fate* of an avoided tonne: the
part destroyed by invalidation, the part still locked in the MSR when the cap ends
(never auctioned), and the part that leaks back to the market (the waterbed).
Two independent derivations are provided: a finite-difference re-simulation
(:func:`fate_bruteforce`) and a forward-mode linearisation along the baseline
path (:func:`fate_linear`); they agree away from kinks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --- Published EU ETS figures (Mt), Commission TNAC communications ---------------
TNAC_PUBLISHED = {2022: 1134.540, 2023: 1110.840, 2024: 1148.050, 2025: 1023.494}
MSR_HOLDINGS_JAN_2026 = 400.0  # after the 1 Jan 2026 invalidation
START_YEAR = 2026  # first simulated calendar year


@dataclass(frozen=True)
class MSRRules:
    upper: float = 1096.0
    lower: float = 833.0
    rate: float = 0.24
    release_trigger: float = 400.0
    release: float = 100.0
    # invalidation threshold by year (None = no invalidation from that year on)
    inval: tuple = ((2023, 400.0),)
    split: bool = True  # Sep-Aug intake year (True) or calendar-year intake (False)
    enabled: bool = True

    def threshold(self, year: int):
        thr = None
        for y, t in self.inval:
            if year >= y:
                thr = t
        return thr

    def net_intake(self, tnac: float) -> float:
        """Allowances moved into (+) or out of (-) the MSR over the following 12 months."""
        if not self.enabled:
            return 0.0
        if tnac > self.upper:
            return self.rate * tnac
        if tnac > self.lower:
            return tnac - self.lower
        if tnac < self.release_trigger:
            return -self.release
        return 0.0

    def slope(self, tnac: float) -> float:
        if not self.enabled:
            return 0.0
        if tnac > self.upper:
            return self.rate
        if tnac > self.lower:
            return 1.0
        return 0.0


RULES = {
    # Directive (EU) 2023/959 as in force before the October 2026 deal
    "law_2023 (400 Mt)": MSRRules(inval=((2023, 400.0),)),
    # provisional Council-Parliament agreement of 20 Oct 2026 (threshold 700 Mt)
    "deal_2026 (700 Mt)": MSRRules(inval=((2023, 400.0), (2027, 700.0))),
    # Commission proposal of 1 Apr 2026: stop invalidation, keep a buffer
    "proposal_2026 (none)": MSRRules(inval=((2023, 400.0), (2027, None))),
    # a cap with no supply adjustment (UK ETS, WCI above the floor, K-ETS)
    "pure cap": MSRRules(enabled=False, inval=()),
}


@dataclass
class Path:
    tnac: dict = field(default_factory=dict)
    msr: dict = field(default_factory=dict)  # holdings at year end
    intake: dict = field(default_factory=dict)  # net flow into the MSR in the calendar year
    invalidated: dict = field(default_factory=dict)  # destroyed on 1 Jan of the year
    end_year: int = 0

    @property
    def total_invalidated(self) -> float:
        return float(sum(self.invalidated.values()))

    @property
    def msr_end(self) -> float:
        return float(self.msr[self.end_year])


def net_supply_path(d2026: float, decline: float, end_year: int = 2039) -> dict:
    """Stylised net supply before the MSR, d_y = supply_y - emissions_y (Mt).

    Calibrated on the published TNACs: d_2024 = 307.6 and d_2025 = 145.1 Mt (see
    :func:`implied_net_supply`); then falls linearly by ``decline`` Mt/yr as the cap's
    4.3-4.4%/yr linear reduction factor outruns the fall in emissions.
    """
    return {y: d2026 - decline * (y - START_YEAR) for y in range(START_YEAR, end_year + 1)}


def implied_net_supply(rules: MSRRules = RULES["law_2023 (400 Mt)"]) -> dict:
    """Back out d_y = TNAC_y - TNAC_{y-1} + MSR flow_y from the published TNACs."""
    out = {}
    for y in (2024, 2025):
        flow = (8 / 12) * rules.net_intake(TNAC_PUBLISHED[y - 2]) + (4 / 12) * rules.net_intake(TNAC_PUBLISHED[y - 1])
        out[y] = TNAC_PUBLISHED[y] - TNAC_PUBLISHED[y - 1] + flow
    return out


def simulate(d: dict, rules: MSRRules, abate: dict | None = None,
             tnac_hist: dict | None = None, msr0: float = MSR_HOLDINGS_JAN_2026) -> Path:
    """Simulate the allowance bank from START_YEAR to max(d).

    ``abate`` maps year -> Mt of extra emissions avoided (raises the TNAC). Negative
    values model extra emissions (e.g. new demand).
    """
    abate = abate or {}
    hist = dict(TNAC_PUBLISHED if tnac_hist is None else tnac_hist)
    for ya, x in abate.items():  # avoided before the simulation starts: already in the TNAC
        for yh in hist:
            if ya < START_YEAR and yh >= ya:
                hist[yh] += x
    years = sorted(d)
    p = Path(end_year=years[-1])
    tnac = dict(hist)
    msr = msr0
    for y in years:
        if y > START_YEAR:  # the 1 Jan 2026 invalidation is already in msr0
            thr = rules.threshold(y) if rules.enabled else None
            inv = max(0.0, msr - thr) if thr is not None else 0.0
            msr -= inv
            p.invalidated[y] = inv
        if rules.split:
            flow = (8 / 12) * rules.net_intake(tnac[y - 2]) + (4 / 12) * rules.net_intake(tnac[y - 1])
        else:
            flow = rules.net_intake(tnac[y - 1])
        flow = max(flow, -msr)  # cannot release more than the reserve holds
        msr += flow
        tnac[y] = tnac[y - 1] + d[y] + abate.get(y, 0.0) - flow
        p.tnac[y], p.msr[y], p.intake[y] = tnac[y], msr, flow
    return p


def fate_from_paths(base: Path, pert: Path, size: float) -> dict:
    """Split an avoided tonne into destroyed / locked / leaked (shares of ``size``)."""
    destroyed = (pert.total_invalidated - base.total_invalidated) / size
    locked = (pert.msr_end - base.msr_end) / size
    return {"destroyed": destroyed, "locked": locked, "real_upper": destroyed + locked,
            "real_lower": destroyed, "leaked": 1.0 - destroyed - locked}


def fate_bruteforce(d: dict, rules: MSRRules, year: int, size: float = 1.0,
                    tnac_hist: dict | None = None) -> dict:
    """Fate of ``size`` Mt avoided in ``year``, by re-simulating the bank (derivation 1)."""
    base = simulate(d, rules, tnac_hist=tnac_hist)
    pert = simulate(d, rules, abate={year: size}, tnac_hist=tnac_hist)
    return fate_from_paths(base, pert, size)


def fate_linear(d: dict, rules: MSRRules, year: int, tnac_hist: dict | None = None) -> dict:
    """Same quantity by forward-mode differentiation along the baseline path (derivation 2).

    Propagates the perturbation (eT, eM) of TNAC and MSR holdings one tonne at a time
    through the piecewise-linear rules, using the slope of each rule at the baseline.
    """
    hist = dict(TNAC_PUBLISHED if tnac_hist is None else tnac_hist)
    base = simulate(d, rules, tnac_hist=hist)
    years = sorted(d)
    eT = {y: (1.0 if y >= year else 0.0) for y in hist}
    eM, destroyed = 0.0, 0.0
    msr_prev = MSR_HOLDINGS_JAN_2026
    for y in years:
        if y > START_YEAR and rules.enabled:
            thr = rules.threshold(y)
            if thr is not None and msr_prev > thr:  # one more tonne held -> one more destroyed
                destroyed += eM
                eM = 0.0
        tn = {k: base.tnac.get(k, hist.get(k)) for k in (y - 2, y - 1)}
        if rules.split:
            dflow = (8 / 12) * rules.slope(tn[y - 2]) * eT[y - 2] + (4 / 12) * rules.slope(tn[y - 1]) * eT[y - 1]
        else:
            dflow = rules.slope(tn[y - 1]) * eT[y - 1]
        # release capped by holdings: if the cap binds the flow follows the holdings
        pre = msr_prev - (base.invalidated.get(y, 0.0))
        if rules.split:
            raw = (8 / 12) * rules.net_intake(tn[y - 2]) + (4 / 12) * rules.net_intake(tn[y - 1])
        else:
            raw = rules.net_intake(tn[y - 1])
        if raw < -pre:
            dflow = -eM
        eM += dflow
        eT[y] = eT[y - 1] + (1.0 if y == year else 0.0) - dflow
        msr_prev = base.msr[y]
    return {"destroyed": destroyed, "locked": eM, "real_upper": destroyed + eM,
            "real_lower": destroyed, "leaked": 1.0 - destroyed - eM}


# --- Which grids sit under which cap --------------------------------------------
EU_ETS = {"AT", "BE", "BG", "CY", "CZ", "DE", "DK-BHM", "DK-DK1", "DK-DK2", "EE", "ES", "FI", "FR",
          "GR", "HR", "HU", "IE", "IT-CNO", "IT-CSO", "IT-NO", "IT-SAR", "IT-SIC", "IT-SO", "LT", "LU",
          "LV", "NL", "PL", "PT", "RO", "SE-SE1", "SE-SE2", "SE-SE3", "SE-SE4", "SI", "SK",
          # EEA members and linked or covered systems
          "NO-NO1", "NO-NO2", "NO-NO3", "NO-NO4", "NO-NO5", "CH", "GB-NIR"}
WCI = {"US-CAL-BANC", "US-CAL-CISO", "US-CAL-LDWP", "US-CAL-TIDC", "CA-QC"}
RGGI = {"US-NE-ISNE", "US-NY-NYIS"}
WA_CCA = {"US-NW-SCL", "US-NW-TPWR", "US-NW-PSEI", "US-NW-CHPD", "US-NW-DOPD", "US-NW-GCPD"}
OTHER_CAP = {"GB": "UK ETS", "KR": "K-ETS", "NZ": "NZ ETS"}


def cap_regime(zone: str) -> str:
    if zone in EU_ETS:
        return "EU ETS (MSR)"
    if zone in WCI:
        return "WCI (cap + floor)"
    if zone in RGGI:
        return "RGGI (cap + ECR)"
    if zone in WA_CCA:
        return "Washington CCA"
    if zone in OTHER_CAP:
        return OTHER_CAP[zone]
    return "no cap"


# Real share of an avoided tonne outside the EU, as a (low, high) bracket. A cap with
# a binding auction price floor behaves like a tax (unsold allowances are withheld):
# high = 1. Above the floor, or with fixed-quantity reserves, it is a waterbed: low = 0.
REGIME_BRACKET = {
    "no cap": (1.0, 1.0),
    "UK ETS": (0.0, 1.0),  # auction reserve price
    "WCI (cap + floor)": (0.0, 1.0),
    "RGGI (cap + ECR)": (0.0, 0.1),
    "Washington CCA": (0.0, 1.0),
    "K-ETS": (0.0, 0.0),
    "NZ ETS": (0.0, 1.0),
}
