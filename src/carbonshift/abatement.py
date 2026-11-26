"""F34: the price of a shifted tonne. Carbon-aware shifting as an abatement technology.

Every earlier finding measured carbon-aware shifting in grams: per kWh, per kW of
headroom (F7), per kg of embodied carbon (F8, F16), per kWh of battery (F23). An
abatement option is judged in money per tonne, against the social cost of carbon
(SCC; US EPA 2023: $190/t at a 2% discount rate, $120-340 over 2.5-1.5%) or a
carbon market price. This module prices the capacity that shifting needs.

A unit (kW) of headroom built for shifting saves t = 8.76e-3 * v tonnes per year,
where v = S(h)/h is its carbon value in g per (kW of headroom) per hour (S from the
capped LP, F21/F22). Its annualised cost is kappa = capex * CRF(r, n). So

    MAC(h) = kappa / (8.76e-3 * v(h))      ($/t, v in g/kWh, kappa in $/kW-yr)

Floor (horizon-free, any slack, any phi): with d_t = served - arrivals, sum d_t = 0
and d_t <= h, so S = (1/L) sum_t (c_max - c_t) d_t <= h * (c_max - cbar). Hence
v <= c_max - cbar and

    MAC >= kappa / (8.76e-3 * (c_max - cbar))                    (``mac_floor``)

i.e. headroom passes an SCC test only where the grid's year peak sits at least
kappa / (8.76e-3 * SCC) g/kWh above its mean (``swing_needed``), and only if every
unit of headroom is used, every hour, to move work out of the year's dirtiest hour.

The same capacity is cheaper per tonne when it already exists for another reason
(growth, redundancy: F9) because then its capex is sunk, and MAC -> 0 (minus the
idle-power penalty of F33 if it stays powered).

Electricity bills. Shifting into clean hours can also save money where clean hours
are cheap. With M = MWh moved per kW of headroom per year, the headroom pays for
itself (net of its SCC-valued carbon) if each moved MWh earns at least
(kappa - SCC * t) / M dollars (``price_gain_needed``). No price data are in this
repository, so this is reported as a requirement, not as an estimate.

Cost inputs (low, central, high) and sources (accessed 2026-11-26; secondary
coverage of the reports, not the reports themselves):

facility_power   data-centre capacity, $ per W of IT load = $k per kW:
                 6.7 (KPMG 2026 benchmark, Spain, lowest market), 10.7 (CBRE via
                 Build, North America average 2025), 15.2 (Turner & Townsend
                 2025-26 index, Tokyo, the most expensive market). Life 20 years.
gpu_servers      NVIDIA DGX H100, 10.2 kW system maximum (NVIDIA SuperPOD design
                 guide); reseller prices $300-400k (no list price published).
                 Life 5 years. Capacity for GPU work also needs facility power,
                 so ``gpu_capacity`` = gpu_servers + facility_power.
battery_4h       $ per kWh: 110 (BNEF Energy Storage Systems Cost Survey 2025,
                 global 4 h turnkey), 220 (same survey, US turnkey all durations
                 ~219), 400 (assumed installed cost incl. EPC and grid connection).
                 Life 15 years.
"""

from __future__ import annotations

import numpy as np

from .exchange import spread_saving_per_kwh

T_PER_GKWH_YEAR = 8760 / 1e6       # tonnes per year from 1 kW at 1 g/kWh, all year
SCC = {"low": 120.0, "central": 190.0, "high": 340.0}   # US EPA 2023, $/t (2.5%, 2%, 1.5%)

COSTS: dict[str, dict] = {
    "facility_power": dict(capex=(6.7e3, 10.7e3, 15.2e3), life=20, unit="kW"),
    "gpu_servers": dict(capex=(300e3 / 10.2, 350e3 / 10.2, 400e3 / 10.2), life=5, unit="kW"),
    "battery_4h": dict(capex=(110.0, 220.0, 400.0), life=15, unit="kWh"),
}
LEVEL = {"low": 0, "central": 1, "high": 2}


def crf(rate: float, years: float) -> float:
    """Capital recovery factor: annual payment per $ of capex."""
    if rate == 0:
        return 1.0 / years
    return rate / (1.0 - (1.0 + rate) ** (-years))


def annual_cost(item: str, level: str = "central", rate: float = 0.07) -> float:
    """Annualised capex, $ per kW-year (or per kWh-year for a battery)."""
    if item == "gpu_capacity":
        return annual_cost("gpu_servers", level, rate) + annual_cost("facility_power", level, rate)
    c = COSTS[item]
    return c["capex"][LEVEL[level]] * crf(rate, c["life"])


def tonnes_per_year(v_g_per_kwh: float) -> float:
    """Tonnes per year saved by 1 kW (or 1 kWh of battery) whose carbon value is v g per hour."""
    return float(v_g_per_kwh) * T_PER_GKWH_YEAR


def mac(kappa: float, tonnes: float) -> float:
    """Marginal (or average) abatement cost, $/t; infinite if nothing is saved."""
    return float(kappa / tonnes) if tonnes > 0 else float("inf")


def breakeven_capex(tonnes: float, scc: float, life: float, rate: float = 0.07) -> float:
    """Largest capex ($ per kW) at which a kW saving ``tonnes`` per year passes an SCC test."""
    return scc * tonnes / crf(rate, life)


def value_bound(ci: np.ndarray) -> float:
    """Horizon-free bound on v = S(h)/h: c_max - cbar (g per kW of headroom per hour)."""
    ci = np.asarray(ci, dtype=float)
    return max(0.0, float(ci.max() - ci.mean()))


def mac_floor(kappa: float, ci: np.ndarray) -> float:
    """Lowest abatement cost any shifting schedule can reach with capacity costing kappa $/kW-yr."""
    return mac(kappa, tonnes_per_year(value_bound(ci)))


def swing_needed(kappa: float, scc: float) -> float:
    """g/kWh by which a grid's peak must exceed its mean for headroom to pass the SCC test."""
    return kappa / (T_PER_GKWH_YEAR * scc)


def battery_tonnes_per_kwh_year(ci: np.ndarray, duration_h: float = 4.0, eta: float = 0.90) -> float:
    """Tonnes per year saved per kWh of daily-cycling battery (closed-form spread law, F23)."""
    return tonnes_per_year(spread_saving_per_kwh(ci, duration_h, eta))


def moved_mwh_per_kw_year(served: np.ndarray, arrivals: np.ndarray, h: float) -> float:
    """MWh of flexible work moved per kW of headroom per year (served above arrival)."""
    served = np.asarray(served, dtype=float)
    arrivals = np.broadcast_to(np.asarray(arrivals, dtype=float), served.shape)
    moved = np.clip(served - arrivals, 0.0, None).sum() * 8760 / len(served)  # kWh per kW mean load
    return float(moved / h / 1000.0)


def price_gain_needed(kappa: float, tonnes: float, moved_mwh: float, scc: float) -> float:
    """$ per moved MWh the shift must earn on the electricity bill for the headroom to pay (net of SCC carbon)."""
    return max(0.0, kappa - scc * tonnes) / moved_mwh if moved_mwh > 0 else float("inf")
