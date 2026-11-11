"""Curtailment-aware carbon accounting for a growing data-centre fleet (F24).

Carbon-aware studies treat the grid signal as exogenous: one more kWh in a clean hour
costs that hour's (average or marginal) intensity, however many kWh arrive. The one
part of the grid that is truly free for flexible load is *curtailed* renewable energy,
and it is finite. A new fleet of mean load P (MW) drawing L_t in hour t emits

    e_t(L_t) = m_t * max(0, L_t - C_t),

where C_t is the renewable energy curtailed in that hour and m_t the marginal emission
rate of the rest of the supply. e_t is convex in L_t, so a coordinated fleet solves a
linear programme, while a "herd" of price-taking schedulers optimises against an
exogenous signal and is scored on e_t afterwards.

Data: CAISO Production and Curtailments workbook (5-minute MW, public), aggregated
here to hourly MWh. Raw files live in data/caiso/raw (git-ignored).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from carbonshift.facility import _cumulative_bounds, capped_shift

RAW = Path(__file__).resolve().parents[2] / "data" / "caiso" / "raw"
DERIVED = Path(__file__).resolve().parents[2] / "data" / "caiso" / "caiso_hourly_{year}.csv.gz"


@lru_cache(maxsize=4)
def caiso_hourly(year: int = 2022) -> pd.DataFrame:
    """Hourly CAISO series (MWh): load, solar, wind, thermal, imports, curtailment. Local prevailing time.

    Each hour is the mean of its 5-minute MW values; curtailment intervals missing from
    the curtailment table are zero.
    """
    out = Path(str(DERIVED).format(year=year))
    if out.exists():
        return pd.read_csv(out, parse_dates=["date"])
    src = RAW / f"ProductionAndCurtailmentsData_{year}.xlsx"
    prod = pd.read_excel(src, sheet_name="Production")
    cur = pd.read_excel(src, sheet_name="Curtailments")
    prod["date"] = pd.to_datetime(prod["Date"]).dt.normalize()
    cur["date"] = pd.to_datetime(cur["Date"]).dt.normalize()
    cur["curtail_mw"] = cur[["Wind Curtailment", "Solar Curtailment"]].fillna(0).sum(axis=1)
    keys = ["date", "Hour", "Interval"]
    p = prod[keys + ["Load", "Solar", "Wind", "Thermal", "Imports"]].merge(
        cur.groupby(keys, as_index=False).curtail_mw.sum(), on=keys, how="left").fillna({"curtail_mw": 0.0})
    h = p.groupby(["date", "Hour"], as_index=False).agg(
        load=("Load", "mean"), solar=("Solar", "mean"), wind=("Wind", "mean"), thermal=("Thermal", "mean"),
        imports=("Imports", "mean"), curtail=("curtail_mw", lambda v: v.sum() / 12.0))
    h = h.rename(columns={"Hour": "hour"}).sort_values(["date", "hour"], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    h.to_csv(out, index=False, compression="gzip")
    return h


def fleet_emissions(load_mw: np.ndarray, curtail_mw: np.ndarray, marginal: np.ndarray) -> float:
    """Tonnes CO2 of a fleet drawing ``load_mw`` when the first ``curtail_mw`` each hour is free."""
    return float((np.asarray(marginal) * np.maximum(0.0, np.asarray(load_mw) - np.asarray(curtail_mw))).sum() / 1000.0)


def herd_profile(signal: np.ndarray, phi: float, cap, slack: int) -> np.ndarray:
    """Per-unit fleet draw when every facility optimises alone against an exogenous signal."""
    r = capped_shift(np.asarray(signal, dtype=float), phi, cap, slack)
    return (1.0 - phi) + r.served


def coordinated_profile(curtail_pu: np.ndarray, marginal: np.ndarray, phi: float, cap, slack: int) -> np.ndarray:
    """Per-unit fleet draw minimising sum m_t * max(0, L_t - C_t), C in units of the fleet's mean load.

    Variables: cumulative flexible work Y (L) and overflow u (L) >= 0 with u >= base + y - C.
    """
    m = np.asarray(marginal, dtype=float)
    C = np.asarray(curtail_pu, dtype=float)
    L = len(m)
    base = (1.0 - phi) * np.ones(L)
    lo, hi = _cumulative_bounds(phi * np.ones(L), slack)
    room = np.full(L, phi * (slack + 1)) if np.isinf(cap) else np.full(L, cap - (1.0 - phi))
    D = sparse.diags([np.ones(L), -np.ones(L - 1)], [0, -1], format="csr")
    Z = sparse.csr_matrix((L, L))
    I = sparse.identity(L, format="csr")
    A_ub = sparse.vstack([sparse.hstack([D, Z]), sparse.hstack([-D, Z]), sparse.hstack([D, -I])], format="csr")
    b_ub = np.concatenate([room, np.zeros(L), C - base])
    c = np.concatenate([np.zeros(L), m])
    bounds = np.vstack([np.column_stack([lo, hi]), np.column_stack([np.zeros(L), np.full(L, np.inf)])])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    return base + np.diff(res.x[:L], prepend=0.0)


def absorbable_daily(curtail_pu: np.ndarray, phi: float, cap) -> float:
    """Closed-form upper bound on the share of fleet energy served by curtailment (daily water-filling).

    Each day the fleet can take at most min(C_t, cap) in every hour, and beyond its
    always-on base it has only 24*phi units of flexible energy to place:
    absorbed_day = min(sum_t min(C_t, cap), sum_t min(C_t, 1 - phi) + 24*phi).
    """
    C = np.asarray(curtail_pu, dtype=float)
    days = C[: len(C) // 24 * 24].reshape(-1, 24)
    top = np.inf if np.isinf(cap) else cap
    a = np.minimum(np.minimum(days, top).sum(axis=1), np.minimum(days, 1.0 - phi).sum(axis=1) + 24 * phi)
    return float(a.sum() / days.size)
