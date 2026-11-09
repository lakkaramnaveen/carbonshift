"""Carbon-aware batching for LLM inference, from measured batch curves.

For each batch size B we have measured chip energy per token e(B) and
throughput tps(B) (results/llm_power_summary.csv). Each sequence waits one
decode step per token, so its time per output token is lat(B) = B / tps(B).
Bigger batches save energy and cost latency.

Each hour a fixed number of tokens is served (flat demand). A schedule picks a
mix of batch sizes per hour (x[t, b] >= 0, summing to 1) under a
token-weighted mean latency budget:

    min  sum_t ci_t * sum_b e_b x[t, b]
    s.t. sum_t sum_b lat_b x[t, b] <= budget * T

*Static* batching uses the same mix every hour (best on the convex hull of
(lat, e)); *carbon-aware* batching may use small, fast batches in clean hours
and large, efficient ones in dirty hours.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

SUMMARY = Path(__file__).resolve().parents[2] / "results" / "llm_power_summary.csv"


@dataclass(frozen=True)
class BatchCurve:
    batch: np.ndarray
    energy_j: np.ndarray      # chip J per token
    latency_s: np.ndarray     # seconds per output token per sequence (decode step time)


def measured_curve(model_prefix: str, path: Path = SUMMARY) -> BatchCurve:
    s = pd.read_csv(path)
    d = s[s.model.str.startswith(model_prefix)].sort_values("batch")
    if d.empty:
        raise ValueError(f"no measurements for {model_prefix}")
    e = d.all_j_per_tok if "all_j_per_tok" in d else d.all_power_w / d.tok_per_s
    return BatchCurve(d.batch.to_numpy(), e.to_numpy(float), (d.batch / d.tok_per_s).to_numpy(float))


def static_best(ci: np.ndarray, curve: BatchCurve, budget_s: float) -> tuple[float, np.ndarray]:
    """Mean g CO2 per token-hour weight with the best fixed mix; returns (objective, mix)."""
    res = linprog(curve.energy_j, A_ub=[curve.latency_s], b_ub=[budget_s], A_eq=[np.ones(len(curve.batch))], b_eq=[1.0],
                  bounds=(0, None), method="highs")
    if res.status != 0:
        raise ValueError("latency budget below the fastest batch size")
    return float(np.mean(ci) * res.fun), res.x


def _pick(ci: np.ndarray, curve: BatchCurve, lam: float) -> np.ndarray:
    """Index of the batch minimising ci_t e_b + lam * lat_b in each hour (ties -> lower latency)."""
    cost = ci[:, None] * curve.energy_j[None, :] + lam * curve.latency_s[None, :]
    return cost.argmin(axis=1)


def carbon_aware(ci: np.ndarray, curve: BatchCurve, budget_s: float, iters: int = 200) -> tuple[float, np.ndarray]:
    """Exact solution of the hourly-mix LP by Lagrangian bisection on the single latency constraint.

    For a multiplier lam, every hour independently picks its best batch. Mean
    latency falls as lam grows, so bisection finds lam* where it crosses the
    budget. A convex combination of the two pure schedules either side of lam*
    meets the budget exactly and attains the Lagrangian dual bound, so it is
    LP-optimal (tested against the LP in ``_carbon_aware_lp``).
    Returns (mean objective, x[T, B]).
    """
    ci = np.asarray(ci, dtype=float)
    T, B = len(ci), len(curve.batch)

    def evaluate(lam):
        idx = _pick(ci, curve, lam)
        x = np.zeros((T, B))
        x[np.arange(T), idx] = 1.0
        return x, float(curve.latency_s[idx].mean()), float((ci * curve.energy_j[idx]).mean())

    if curve.latency_s.min() > budget_s + 1e-15:
        raise ValueError("latency budget below the fastest batch size")
    x0, l0, f0 = evaluate(0.0)
    if l0 <= budget_s:  # energy-optimal schedule already meets the budget
        return f0, x0
    lo, hi = 0.0, 1.0
    while evaluate(hi)[1] > budget_s:
        hi *= 2.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if evaluate(mid)[1] > budget_s:
            lo = mid
        else:
            hi = mid
    xa, la, fa = evaluate(lo)   # latency above budget, lower carbon
    xb, lb, fb = evaluate(hi)   # latency within budget
    theta = 1.0 if la - lb < 1e-15 else (budget_s - lb) / (la - lb)
    theta = float(np.clip(theta, 0.0, 1.0))
    return theta * fa + (1 - theta) * fb, theta * xa + (1 - theta) * xb


def _carbon_aware_lp(ci: np.ndarray, curve: BatchCurve, budget_s: float) -> float:
    """Reference LP (slow; used in tests)."""
    ci = np.asarray(ci, dtype=float)
    T, B = len(ci), len(curve.batch)
    c = (ci[:, None] * curve.energy_j[None, :]).ravel()
    A_eq = sparse.kron(sparse.identity(T), np.ones((1, B)), format="csr")
    A_ub = sparse.csr_matrix(np.tile(curve.latency_s, T)[None, :])
    res = linprog(c, A_ub=A_ub, b_ub=[budget_s * T], A_eq=A_eq, b_eq=np.ones(T), bounds=(0, None), method="highs")
    if res.status != 0:
        raise ValueError(res.message)
    return float(res.fun / T)
