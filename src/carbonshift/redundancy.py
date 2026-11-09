"""The redundancy dividend: redundant power paths as free carbon-shifting headroom.

Tier-rated facilities install more power-delivery capacity than the critical
load needs: N+1 adds one spare module, and 2N duplicates the whole path. This
spare capacity sits idle except during maintenance or failures. Microsoft's
Flex (ISCA'21) showed the reserve can host *sheddable* work and be reclaimed
within seconds on a failure. Here that work is carbon-aware deferrable batch work.
When a path is down, the facility falls back to its design capacity and the
flexible work is postponed, never lost, because it already has slack.

All power is in units of the facility's mean IT load. ``u`` is the design
utilisation: mean load / design (critical) capacity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

TOPOLOGIES = {"design_only": None, "N+1 (N=4)": 4, "N+1 (N=2)": 2, "2N": "2N"}


def installed_cap(u: float, topology: str) -> float:
    """Normal-operation capacity in units of mean load."""
    design = 1.0 / u
    t = TOPOLOGIES[topology]
    if t is None:
        return design
    if t == "2N":
        return 2.0 * design
    return design * (t + 1) / t


@dataclass(frozen=True)
class OutagePlan:
    maintenance_windows: int = 2      # planned windows per year on the redundant path
    maintenance_hours: int = 24       # length of each window
    failure_rate_per_year: float = 0.5  # unplanned events (Poisson)
    failure_hours: int = 8            # mean repair time for each event


def random_windows(rng: np.random.Generator, L: int, n: int, length: int) -> list[int]:
    """n non-overlapping window starts, uniformly at random."""
    starts: list[int] = []
    tries = 0
    while len(starts) < n and tries < 10_000:
        s = int(rng.integers(0, L - length))
        if all(abs(s - o) >= length for o in starts):
            starts.append(s)
        tries += 1
    return starts


def carbon_aware_windows(shadow: np.ndarray, n: int, length: int, min_gap: int = 24 * 14) -> list[int]:
    """Pick n maintenance windows where capacity is worth least (min summed shadow price).

    Windows are kept at least ``min_gap`` hours apart so maintenance stays spread over the year.
    """
    cost = sliding_window_view(shadow, length).sum(axis=1)
    starts: list[int] = []
    cost = cost.astype(float).copy()
    for _ in range(n):
        s = int(np.argmin(cost))
        starts.append(s)
        lo, hi = max(0, s - min_gap), min(len(cost), s + min_gap)
        cost[lo:hi] = np.inf
    return starts


def cap_profile(L: int, normal: float, degraded: float, windows: list[tuple[int, int]]) -> np.ndarray:
    cap = np.full(L, normal)
    for s, n in windows:
        cap[s : s + n] = degraded
    return cap


def failure_windows(rng: np.random.Generator, L: int, plan: OutagePlan) -> list[tuple[int, int]]:
    k = rng.poisson(plan.failure_rate_per_year * L / 8760)
    out = []
    for _ in range(k):
        dur = max(1, int(round(rng.exponential(plan.failure_hours))))
        out.append((int(rng.integers(0, L - 1)), dur))
    return out
