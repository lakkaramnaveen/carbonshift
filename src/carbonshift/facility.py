"""Facility-side limits on carbon-aware temporal shifting (this project's extension).

The EuroSys'24 temporal analysis optimises each job on its own, so every job
can pile into the same low-carbon hours. A real facility cannot do that: grid
draw is bounded by the utility service, transformers, switchgear and the UPS
rating (with redundancy, e.g. N+1 or 2N, the usable rating is below nameplate).

Model (all power normalised to the facility's mean IT load = 1):

* inflexible load ``1 - phi`` every hour (interactive services, storage, ...);
* flexible, pre-emptible work ``phi`` arrives every hour and must be served
  within ``slack`` hours of arrival (same slack for every arrival);
* the total draw in any hour is at most ``cap`` (cap = 1 means no headroom).

Because every arrival has the same slack, earliest-deadline-first equals
FIFO, so any served profile y is feasible exactly when its cumulative sum Y
lies between the cumulative arrivals delayed by ``slack`` and the cumulative
arrivals. That makes the problem a linear programme in Y with L variables
(L = 8760), banded constraints and no per-arrival variables. HiGHS solves it in
tens of milliseconds.

With ``cap = inf`` the optimum equals the paper's 1-hour-job result with the
same slack (unit-tested), so the extension nests the reproduced baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


@dataclass(frozen=True)
class CappedResult:
    baseline_g: float      # g CO2eq/kWh-of-mean-load, run-on-arrival
    optimised_g: float     # with carbon-aware shifting under the cap
    served: np.ndarray     # flexible kWh served each hour
    # Shadow carbon price of capacity: grams CO2eq saved per extra unit of power
    # (in units of mean load) made available for one hour, per hour. None if uncapped.
    shadow: np.ndarray | None = None

    @property
    def annual_marginal_saving(self) -> float:
        """g CO2eq saved per year per extra kW of capacity, per kW of mean load.

        Equivalently: grams per year saved by one more kW of headroom at a
        facility whose mean load is 1 kW (sum of hourly shadow prices).
        """
        return float(self.shadow.sum()) if self.shadow is not None else 0.0

    @property
    def saving_per_flex_kwh(self) -> float:
        """Average reduction per kWh of *flexible* work (comparable to the paper's metric)."""
        flex = self.served.sum()
        return (self.baseline_g - self.optimised_g) * len(self.served) / flex

    @property
    def facility_reduction_pct(self) -> float:
        return (self.baseline_g - self.optimised_g) / self.baseline_g * 100


def _cumulative_bounds(arrivals: np.ndarray, slack: int) -> tuple[np.ndarray, np.ndarray]:
    """Bounds on cumulative flexible work served by the end of each hour."""
    L = len(arrivals)
    arrived = np.cumsum(arrivals)                        # arrivals at start of hours 0..s
    due = np.zeros(L)
    due[slack:] = arrived[: L - slack]                   # arrival t must be done by end of t+slack
    due[-1] = arrived[-1]                                # windows are clipped at the end of the year
    return due, arrived


def capped_shift(ci: np.ndarray, phi: float, cap, slack: int, load: np.ndarray | None = None) -> CappedResult:
    """Minimum-carbon schedule of flexible work under a facility power cap.

    ``cap`` is a scalar or an hourly array (e.g. lower during maintenance or
    after an equipment failure), in units of mean load; ``np.inf`` = uncapped.
    ``load`` is an optional hourly load shape with mean 1 (default flat). Each
    hour, ``(1 - phi) * load`` is inflexible and ``phi * load`` arrives as
    flexible work.
    """
    ci = np.asarray(ci, dtype=float)
    L = len(ci)
    shape = np.ones(L) if load is None else np.asarray(load, dtype=float)
    if shape.shape != (L,):
        raise ValueError("load must have one value per hour")
    base = (1.0 - phi) * shape
    arrivals = phi * shape
    cap_arr = np.broadcast_to(np.asarray(cap, dtype=float), (L,))
    if np.any(cap_arr < shape - 1e-12):
        raise ValueError("cap must be >= the load in every hour or the load cannot be served")
    baseline = float((ci * shape).sum() / L)  # run on arrival

    lo, hi = _cumulative_bounds(arrivals, slack)
    shadow = None

    if np.all(np.isinf(cap_arr)):
        # Uncapped: each arrival is served at the cheapest hour in its window.
        served = _uncapped_profile(ci, arrivals, slack)
    else:
        never_binding = arrivals.max() * (slack + 1)
        room = np.where(np.isinf(cap_arr), never_binding, cap_arr - base)
        # Variables Y_0..Y_{L-1}; y_s = Y_s - Y_{s-1} in [0, room_s].
        # objective sum_s ci_s y_s = sum_s Y_s (ci_s - ci_{s+1}) with ci_L = 0
        c = ci - np.append(ci[1:], 0.0)
        D = sparse.diags([np.ones(L), -np.ones(L - 1)], [0, -1], format="csr")  # (DY)_s = y_s
        A_ub = sparse.vstack([D, -D], format="csr")
        b_ub = np.concatenate([room, np.zeros(L)])
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=np.column_stack([lo, hi]), method="highs")
        if res.status != 0:
            raise RuntimeError(f"LP failed: {res.message}")
        served = np.diff(res.x, prepend=0.0)
        # d(objective)/d(room_s) <= 0; its negation is grams saved per unit of extra capacity-hour.
        shadow = -np.asarray(res.ineqlin.marginals[:L])

    optimised = float((ci * (base + served)).sum() / L)
    return CappedResult(baseline, optimised, served, shadow)


def _uncapped_profile(ci: np.ndarray, arrivals: np.ndarray, slack: int) -> np.ndarray:
    """Each hourly arrival goes to the cheapest hour in [t, min(t+slack, L-1)]."""
    from numpy.lib.stride_tricks import sliding_window_view

    L = len(ci)
    padded = np.concatenate([ci, np.full(slack, np.inf)])
    target = np.arange(L) + sliding_window_view(padded, slack + 1)[:L].argmin(axis=1)
    return np.bincount(target, weights=arrivals, minlength=L)


# --------------------------------------------------------------------------- UPS battery
@dataclass(frozen=True)
class BatteryResult:
    baseline_g: float
    optimised_g: float
    throughput_kwh: float  # energy discharged over the year, per kW of mean load

    @property
    def reduction_pct(self) -> float:
        return (self.baseline_g - self.optimised_g) / self.baseline_g * 100


def battery_shift(
    ci: np.ndarray,
    energy_h: float,
    power: float = 1.0,
    round_trip_eff: float = 0.90,
    reserve_frac: float = 0.0,
    cap: float = np.inf,
) -> BatteryResult:
    """Carbon-optimal dispatch of a battery behind a flat 1.0 load.

    ``energy_h`` is usable capacity in hours of mean load (a UPS sized for
    10 minutes of ride-through at full load is about 1/6 h). ``reserve_frac``
    of it must stay charged for ride-through and cannot be cycled. Grid draw
    is limited to ``cap``. The state of charge is cyclic over the year.
    """
    ci = np.asarray(ci, dtype=float)
    L = len(ci)
    eta = np.sqrt(round_trip_eff)
    usable = energy_h * (1 - reserve_frac)
    baseline = float(ci.mean())
    if usable <= 0:
        return BatteryResult(baseline, baseline, 0.0)

    # x = [charge c (L), discharge d (L), state e (L)]
    n = 3 * L
    obj = np.concatenate([ci, -ci, np.zeros(L)])  # grid = 1 + c - d
    I = sparse.identity(L, format="csr")
    shift = sparse.diags([np.ones(L - 1)], [-1], shape=(L, L), format="lil")
    shift[0, L - 1] = 1.0  # cyclic: e_{-1} = e_{L-1}
    # e_s - e_{s-1} - eta*c_s + d_s/eta = 0
    A_eq = sparse.hstack([-eta * I, I / eta, I - shift.tocsr()], format="csr")
    b_eq = np.zeros(L)
    bounds = [(0, power)] * L + [(0, min(power, 1.0))] * L + [(0, usable)] * L
    A_ub = b_ub = None
    if np.isfinite(cap):
        A_ub = sparse.hstack([I, -I, sparse.csr_matrix((L, L))], format="csr")
        b_ub = np.full(L, cap - 1.0)
    res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    c, d = res.x[:L], res.x[L : 2 * L]
    optimised = float((ci * (1 + c - d)).sum() / L)
    return BatteryResult(baseline, optimised, float(d.sum()))
