"""Temporal workload shifting upper bounds (Sukprasert et al., EuroSys'24, Sec. 3.2.1, 5.2).

A job of length ``J`` hours arrives at hour ``i`` with slack ``S`` hours, and the
future carbon intensity is known exactly. Three emission values per arrival:

* ``baseline``  run immediately: sum(ci[i:i+J])
* ``defer``     best contiguous J-hour block starting in [i, i+S]
                (k-element subarray with minimum sum)
* ``interrupt`` the J cheapest hours anywhere in [i, i+S+J) (pause/resume for free)

Edge handling follows the original artifact exactly, so numbers are comparable:

* finite slack: arrivals i = 0 .. L-J-1, and windows are truncated at the end of
  the year;
* one-year slack (S >= L): the window is "from arrival to the end of the year",
  with arrivals i = 0 .. L-J.

Every function is vectorised with NumPy; one zone-year takes milliseconds.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HOURS_PER_YEAR = 8760
SLACKS = (24, 24 * 7, 24 * 20, 24 * 30, 24 * 365)
JOB_LENGTHS = (1, 6, 12, 24, 48, 96, 168)

# Job-length mixes from the original artifact (Sec. 5.2.5; Azure V2 and Google Borg v3).
JOB_WEIGHTS = {
    "equal": tuple([1 / len(JOB_LENGTHS)] * len(JOB_LENGTHS)),
    "gcp": (0.02, 0.04, 0.04, 0.10, 0.05, 0.05, 0.70),
    "azure": (0.004, 0.009, 0.009, 0.014, 0.013, 0.02, 0.931),
}


@dataclass(frozen=True)
class ShiftResult:
    """Per-arrival emissions in g·CO2eq per kWh of hourly load (summed over the job)."""

    baseline: np.ndarray
    defer: np.ndarray
    interrupt: np.ndarray

    def per_unit(self, job: int) -> dict[str, float]:
        """Mean reduction per job-hour: what the paper's Figures 7-10 plot."""
        return {
            "defer": float((self.baseline - self.defer).mean() / job),
            "interrupt_extra": float((self.defer - self.interrupt).mean() / job),
            "combined": float((self.baseline - self.interrupt).mean() / job),
        }


def rolling_sum(ci: np.ndarray, job: int) -> np.ndarray:
    """R[k] = sum(ci[k:k+job]) for k = 0 .. len(ci)-job."""
    c = np.concatenate(([0.0], np.cumsum(ci)))
    return c[job:] - c[:-job]


def _suffix_k_smallest_sum(ci: np.ndarray, k: int, n_starts: int) -> np.ndarray:
    """out[i] = sum of the k smallest values of ci[i:], for i < n_starts.

    Sweeps from the end with a size-k max-heap: O(L log k).
    """
    L = len(ci)
    out = np.empty(n_starts)
    heap: list[float] = []  # negated values -> max-heap
    total = 0.0
    for i in range(L - 1, -1, -1):
        v = ci[i]
        if len(heap) < k:
            heapq.heappush(heap, -v)
            total += v
        elif v < -heap[0]:
            total += v + heapq.heapreplace(heap, -v)
        if i < n_starts:
            out[i] = total
    return out


def shift_one_zone(ci: np.ndarray, job: int, slack: int) -> ShiftResult:
    """Baseline, deferred and interruptible emissions for every arrival hour."""
    ci = np.asarray(ci, dtype=float)
    L = len(ci)
    R = rolling_sum(ci, job)

    if slack >= L:  # one-year slack: window runs to the end of the year
        n = L - job + 1
        defer = np.minimum.accumulate(R[::-1])[::-1][:n]
        interrupt = _suffix_k_smallest_sum(ci, job, n)
        return ShiftResult(R[:n].copy(), defer, interrupt)

    n = L - job
    # Deferral: min of R over start offsets [i, i+slack], clipped at the year's end.
    R_pad = np.concatenate((R, np.full(slack, np.inf)))
    defer = sliding_window_view(R_pad, slack + 1)[:n].min(axis=1)
    # Interruption: sum of the `job` smallest hours in ci[i : i+slack+job].
    ci_pad = np.concatenate((ci, np.full(slack, np.inf)))
    windows = sliding_window_view(ci_pad, slack + job)[:n]
    interrupt = np.partition(windows, job - 1, axis=1)[:, :job].sum(axis=1)
    return ShiftResult(R[:n].copy(), defer, interrupt)


def per_unit_table(ci_matrix: np.ndarray, zones: list[str], job: int, slack: int):
    """DataFrame (zone x {defer, interrupt_extra, combined}) of per-job-hour reductions."""
    import pandas as pd

    rows = {z: shift_one_zone(ci_matrix[:, j], job, slack).per_unit(job) for j, z in enumerate(zones)}
    return pd.DataFrame.from_dict(rows, orient="index")


def relative_reduction(ci_matrix: np.ndarray, job: int, slack: int) -> dict[str, float]:
    """Figure 9: reduction as % of baseline, pooled over zones then averaged over arrivals."""
    res = [shift_one_zone(ci_matrix[:, j], job, slack) for j in range(ci_matrix.shape[1])]
    base = np.sum([r.baseline for r in res], axis=0)
    defer = np.sum([r.defer for r in res], axis=0)
    inter = np.sum([r.interrupt for r in res], axis=0)
    return {
        "defer_pct": float(np.mean((base - defer) / base) * 100),
        "interrupt_pct": float(np.mean((defer - inter) / base) * 100),
    }
