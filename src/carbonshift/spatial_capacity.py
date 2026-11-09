"""Spatial shadow price of capacity: what is one more kW worth in each region of a cloud?

Every region hosts a facility with mean load 1. Each hour a share ``phi`` of each
origin's load is flexible and may run in any region reachable within the latency
limit (including its own); the rest stays. A destination's total load may not
exceed its capacity ``cap``. Each hour is an independent min-carbon
transportation LP:

    min  sum_{o,d} ci_d x_od
    s.t. sum_d x_od = phi                       (each origin's flexible load is placed)
         (1 - phi) + sum_o x_od <= cap_d        (destination capacity)
         x_od >= 0, only for latency(o, d) <= limit

The dual of each destination's capacity constraint is the grams of CO2 saved per
extra kW in that region in that hour; summed over the year it is the region's
spatial shadow price (g per kW-year). This ignores time shifting; see
``facility.capped_shift`` for the temporal counterpart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


@dataclass(frozen=True)
class SpatialResult:
    regions: list[str]
    baseline_g: float          # mean g/kWh if nothing moves
    optimised_g: float         # mean g/kWh with migration
    shadow: np.ndarray         # (hours, regions) g saved per extra kW-hour of capacity

    @property
    def annual_shadow(self) -> np.ndarray:
        """g CO2 per year saved by one more kW in each region (per kW of mean regional load)."""
        return self.shadow.sum(axis=0)

    @property
    def reduction_pct(self) -> float:
        return (self.baseline_g - self.optimised_g) / self.baseline_g * 100


def allowed_pairs(latency: np.ndarray, limit_ms: float) -> list[tuple[int, int]]:
    R = latency.shape[0]
    return [(o, d) for o in range(R) for d in range(R) if o == d or latency[o, d] <= limit_ms]


def spatial_shift(ci: np.ndarray, regions: list[str], latency: np.ndarray, phi: float, cap, limit_ms: float) -> SpatialResult:
    """``ci`` is (hours, regions); ``latency[o, d]`` in ms; ``cap`` scalar or per-region array (x mean load)."""
    ci = np.asarray(ci, dtype=float)
    T, R = ci.shape
    caps = np.broadcast_to(np.asarray(cap, dtype=float), (R,))
    if np.any(caps < 1 - 1e-12):
        raise ValueError("capacity must cover each region's own load")
    pairs = allowed_pairs(latency, limit_ms)
    P = len(pairs)
    o_idx = np.array([p[0] for p in pairs])
    d_idx = np.array([p[1] for p in pairs])
    A_eq = sparse.csr_matrix((np.ones(P), (o_idx, np.arange(P))), shape=(R, P))
    A_ub = sparse.csr_matrix((np.ones(P), (d_idx, np.arange(P))), shape=(R, P))
    b_eq = np.full(R, phi)
    b_ub = caps - (1 - phi)
    shadow = np.zeros((T, R))
    total = 0.0
    for t in range(T):
        res = linprog(ci[t, d_idx], A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
        if res.status != 0:
            raise RuntimeError(f"hour {t}: {res.message}")
        total += res.fun + (1 - phi) * ci[t].sum()
        shadow[t] = -res.ineqlin.marginals
    baseline = float(ci.sum() / (T * R))
    return SpatialResult(list(regions), baseline, float(total / (T * R)), shadow)


def joint_shift(ci: np.ndarray, regions: list[str], latency: np.ndarray, phi: float, cap, limit_ms: float, slack: int) -> float:
    """Joint spatial + temporal shifting under per-region capacity; returns mean g/kWh over regions and hours.

    Flexible work arriving at origin o in hour t (phi per hour) may run in any
    reachable region d in any hour s in [t, t+slack]. Because every arrival has
    the same slack, FIFO per origin is optimal, so feasibility reduces to
    cumulative bounds on each origin's served work (as in ``facility.capped_shift``).

    Variables: y[p, s] (work of pair p = (o, d) served in hour s) and the
    cumulative Y[o, s] = Y[o, s-1] + sum_d y[(o, d), s].
    """
    ci = np.asarray(ci, dtype=float)
    T, R = ci.shape
    caps = np.broadcast_to(np.asarray(cap, dtype=float), (R,))
    pairs = allowed_pairs(latency, limit_ms)
    P = len(pairs)
    o_of = np.array([p[0] for p in pairs])
    d_of = np.array([p[1] for p in pairs])
    ny = P * T
    n = ny + R * T
    yi = lambda p, s: p * T + s
    Yi = lambda o, s: ny + o * T + s

    c = np.zeros(n)
    c[:ny] = ci[:, d_of].T.ravel()  # y[p, s] costs ci[s, d_p]

    # Equalities: Y[o,s] - Y[o,s-1] - sum_{p: o_p=o} y[p,s] = 0
    rows, cols, vals = [], [], []
    s_idx = np.arange(T)
    for o in range(R):
        r0 = o * T
        rows += list(r0 + s_idx); cols += list(Yi(o, s_idx)); vals += [1.0] * T
        rows += list(r0 + s_idx[1:]); cols += list(Yi(o, s_idx[1:] - 1)); vals += [-1.0] * (T - 1)
        for p in np.where(o_of == o)[0]:
            rows += list(r0 + s_idx); cols += list(yi(p, s_idx)); vals += [-1.0] * T
    A_eq = sparse.csr_matrix((vals, (rows, cols)), shape=(R * T, n))
    b_eq = np.zeros(R * T)

    # Destination capacity: sum_{p: d_p=d} y[p,s] <= cap_d - (1 - phi)
    rows, cols = [], []
    for d in range(R):
        for p in np.where(d_of == d)[0]:
            rows += list(d * T + s_idx); cols += list(yi(p, s_idx))
    A_ub = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(R * T, n))
    b_ub = np.repeat(caps - (1 - phi), T)

    arrived = phi * np.arange(1, T + 1)
    due = np.zeros(T)
    due[slack:] = arrived[: T - slack]
    due[-1] = arrived[-1]
    bounds = [(0, None)] * ny + [(due[s], arrived[s]) for _ in range(R) for s in range(T)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    return float((res.fun + (1 - phi) * ci.sum()) / (T * R))
