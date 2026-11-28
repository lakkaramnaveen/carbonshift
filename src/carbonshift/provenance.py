"""Signal provenance: the carbon signal is a methodology, not a measurement (F36).

Two defensible *average* carbon-intensity signals exist for the same US grid-hour:

* production-based: the balancing authority's own generation mix times lifecycle
  factors (``eia930.signals``, built from EIA-930);
* consumption-based (flow-traced): Electricity Maps, which follows imports
  through the interconnection (``data.load_carbon``, the EuroSys'24 data).

A scheduler optimises on one of them and reports its saving on the same one.
This module asks what that schedule saves under the other methodology.

Water-filling schedule (24 h slack, daily blocks, cap 1 + h, flexible share phi;
see ``waterfill``): each day's 24 phi units of flexible work fill the
k = 24 phi / (phi + h) cleanest hours at rate phi + h. The schedule depends on the
signal only through the *ranking of hours within each day*, so (F15 lemma) any
affine re-scaling of a signal leaves it unchanged. What matters is the
within-day shape.

Cross-evaluation. Write the saving of schedule x under signal c as
S_c(x) = -sum_t c_t (x_t - phi) / T = -T^-1 <c - cbar_day, x - phi>, the
within-day covariance of the signal and the flexible draw (F25). Regress the
within-day deviations of B on those of A: dB = beta dA + e, beta = rho sB / sA.
Then S_B(x_A) = beta S_A(x_A) - <e, x_A - phi>/T. If e is unrelated to the
ranking of A, the second term vanishes in expectation and

    S_B(x_A) / S_A(x_A) ~= beta = rho sB/sA      (the claim B confirms)
    S_B(x_A) / S_B(x_B) ~= rho                   (the share of B's optimum kept,
                                                  using S proportional to s, F15)

where rho is the correlation of the two signals' within-day deviations. The first
ratio is the decision-analysis "optimizer's curse" (Smith & Winkler 2006) in
regression form: the reported saving is shrunk by the reliability of the shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _days(ci: np.ndarray) -> np.ndarray:
    ci = np.asarray(ci, dtype=float)
    return ci[: len(ci) // 24 * 24].reshape(-1, 24)


def waterfill_schedule(ci: np.ndarray, phi: float, cap) -> np.ndarray:
    """Flexible work served each hour (daily blocks, 24 h slack, cap 1 + h); sums to 24 phi per day.

    Ties are broken by hour order (stable sort), so a constant day is served at the
    start of the day; with ``cap = inf`` all of a day's work goes into its cleanest hour.
    """
    days = _days(ci)
    n, _ = days.shape
    order = np.argsort(days, axis=1, kind="stable")
    served = np.zeros_like(days)
    total = 24.0 * phi
    rate = total if np.isinf(cap) else phi + (cap - 1.0)
    if rate <= 0:
        raise ValueError("cap must exceed 1 - phi")
    k = total / rate
    kf = int(np.floor(k + 1e-12))
    rows = np.arange(n)[:, None]
    if kf > 0:
        served[rows, order[:, :kf]] = rate
    if kf < 24 and k - kf > 1e-12:
        served[np.arange(n), order[:, kf]] = (k - kf) * rate
    return served.ravel()


def saving_of(ci: np.ndarray, served: np.ndarray, phi: float) -> float:
    """Saving per hour of mean load (g/kWh) of a flexible schedule, versus running on arrival."""
    c = _days(ci).ravel()
    x = np.asarray(served, dtype=float)[: len(c)]
    return float((c * phi).mean() - (c * x).mean())


def within_day(ci: np.ndarray) -> np.ndarray:
    """Hourly deviations from each day's mean."""
    d = _days(ci)
    return (d - d.mean(axis=1, keepdims=True)).ravel()


@dataclass
class CrossResult:
    own_a: float  # saving A reports for the schedule optimised on A
    own_b: float  # saving B reports for the schedule optimised on B
    a_on_b: float  # schedule optimised on A, evaluated under B
    b_on_a: float  # schedule optimised on B, evaluated under A
    rho: float  # correlation of within-day deviations
    sd_a: float
    sd_b: float

    @property
    def confirmed_a(self) -> float:
        """Share of A's reported saving that B confirms."""
        return self.a_on_b / self.own_a if self.own_a > 0 else np.nan

    @property
    def kept_b(self) -> float:
        """Share of B's optimum obtained by scheduling on A."""
        return self.a_on_b / self.own_b if self.own_b > 0 else np.nan

    @property
    def beta(self) -> float:
        return self.rho * self.sd_b / self.sd_a if self.sd_a > 0 else np.nan


def cross_evaluate(a: np.ndarray, b: np.ndarray, phi: float, cap) -> CrossResult:
    xa = waterfill_schedule(a, phi, cap)
    xb = waterfill_schedule(b, phi, cap)
    da, db = within_day(a), within_day(b)
    sa, sb = float(da.std()), float(db.std())
    rho = float(np.corrcoef(da, db)[0, 1]) if sa > 0 and sb > 0 else np.nan
    return CrossResult(
        own_a=saving_of(a, xa, phi),
        own_b=saving_of(b, xb, phi),
        a_on_b=saving_of(b, xa, phi),
        b_on_a=saving_of(a, xb, phi),
        rho=rho,
        sd_a=sa,
        sd_b=sb,
    )


def paired_signals(year: int = 2022) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hourly UTC (Electricity Maps, EIA-930 production-based) average signals for the BAs in both."""
    from .data import load_carbon
    from .eia930 import signals

    em = load_carbon(year)
    s = signals()
    s = s.assign(utc=pd.to_datetime(s["utc"]))
    eia = s.pivot(index="utc", columns="ba", values="average")
    zone = {c.split("-")[-1]: c for c in em.columns if c.startswith("US-")}
    common = sorted(set(zone) & set(eia.columns))
    em_ = em[[zone[b] for b in common]].set_axis(common, axis=1)
    idx = em_.index.intersection(eia.index)
    return em_.loc[idx], eia.loc[idx, common]
