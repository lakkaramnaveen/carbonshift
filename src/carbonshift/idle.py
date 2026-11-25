"""F33: the idle-power wall. Spare capacity for carbon-aware shifting leaks idle power.

Carbon-aware shifting needs spare capacity h (F3, F4, F21). Every earlier result
priced that capacity as if servers were energy proportional, or counted only its
embodied carbon (F8). Real servers draw a fraction p0 of their full-load power
when idle (SPECpower_ssj2008 x86 mean: 25.7% in 2024, minimum 15.7% in 2017).

Model (power normalised to the facility's mean IT work = 1, full-load power per
unit of capacity = 1, capacity K = 1 + h):

    always on:  P_t = p0 * K + (1 - p0) * W_t     (spare servers stay powered)
    sleep:      P_t = W_t                         (idealised fleet-level power-down)

where W_t is the work served (inflexible 1 - phi plus the shifted flexible work,
<= K). Moving work only moves the dynamic part, so with always-on servers

    saving of shifting      = (1 - p0) * S(h)      (S: the energy-proportional saving)
    standby cost of the h   = p0 * h * cbar        (if the h is built for shifting)

In the battery picture of F22 the spare capacity is a battery with a standby
(parasitic) load, like a battery's own cooling and electronics, paid whether or
not it is used.

Break-even law. dS/dh at h = 0+ is the value of the first unit of headroom. For
daily water-filling (F21) it equals D = mean over days of (daily max - daily mean),
and for a sinusoidal day of amplitude A it equals A exactly (sinc'(pi) = -1/pi).
Building any always-on capacity for shifting lowers emissions only if

    (1 - p0) * D > p0 * cbar    <=>    p0 < p0* = D / (D + cbar).

Since D <= cbar for a non-negative signal that is close to sinusoidal, p0* is
below about one half everywhere, and it is far smaller on most grids.

With very little headroom the virtual battery plans beyond one day (F22: horizon
s*phi/h), so the exact LP value of the first units is somewhat above D. Whatever
the horizon, the first unit can at best move work out of the year's dirtiest hour,
so p0* <= 1 - cbar / max(c) (``breakeven_bound``). The exact break-even for a
finite build is max over h of v(h) / (v(h) + cbar) with v(h) = S(h)/h, the
average carbon value of a unit of headroom from the LP (``breakeven_idle_lp``).
"""

from __future__ import annotations

import numpy as np

from .facility import capped_shift
from .waterfill import saving_general

HOURS_PER_YEAR = 8760


def _days(ci: np.ndarray) -> np.ndarray:
    ci = np.asarray(ci, dtype=float)
    return ci[: len(ci) // 24 * 24].reshape(-1, 24)


def first_unit_value(ci: np.ndarray) -> float:
    """Closed-form value of the first unit of headroom, g per (unit of h) per hour: mean daily (max - mean).

    With a tiny extra rate r = phi + h, each day's work fills all but a sliver of the
    dirtiest hour; the derivative of the water-filling saving is the area between the
    water level (the day's maximum) and the price curve, divided by 24.
    """
    d = _days(ci)
    return float((d.max(axis=1) - d.mean(axis=1)).mean())


def breakeven_idle(ci: np.ndarray) -> float:
    """Largest idle fraction p0 at which always-on capacity built for shifting still cuts carbon."""
    D = first_unit_value(ci)
    return D / (D + float(np.mean(_days(ci))))


def breakeven_bound(ci: np.ndarray) -> float:
    """Horizon-free upper bound on the break-even idle fraction: 1 - mean / max."""
    ci = np.asarray(ci, dtype=float)
    return float(1.0 - ci.mean() / ci.max())


def capped_saving(ci: np.ndarray, phi: float, h: float, slack: int = 24, exact: bool = True) -> float:
    """Energy-proportional saving S(h), g per kWh of mean work: LP, or the daily water-filling law."""
    if h <= 0:
        return 0.0
    if exact:
        r = capped_shift(ci, phi, 1.0 + h, slack)
        return r.baseline_g - r.optimised_g
    return saving_general(ci, phi, 1.0 + h, slack)


def breakeven_idle_lp(ci: np.ndarray, phi: float, hs=(0.01, 0.02, 0.05, 0.1, 0.2, 0.42, 1.0), slack: int = 24) -> tuple[float, float]:
    """(p0*, h at which it is attained): the largest idle fraction at which some always-on build h pays."""
    cbar = float(np.mean(ci))
    best, hb = 0.0, 0.0
    for h in hs:
        v = capped_saving(ci, phi, h, slack) / h
        p = v / (v + cbar)
        if p > best:
            best, hb = p, h
    return best, hb


def net_build_saving(ci: np.ndarray, phi: float, h: float, p0: float, slack: int = 24, exact: bool = False) -> float:
    """Net saving (g per kWh of mean work) of building h of always-on capacity and shifting into it.

    exact=True uses the LP (facility.capped_shift); otherwise the daily water-filling law.
    """
    ci = np.asarray(ci, dtype=float)
    cbar = float(ci.mean())
    if h <= 0:
        return 0.0
    S = capped_saving(ci, phi, h, slack, exact)
    return (1.0 - p0) * S - p0 * h * cbar


def optimal_headroom(ci: np.ndarray, phi: float, p0: float, grid=None, slack: int = 24,
                     exact: bool = False) -> tuple[float, float]:
    """(h*, net saving at h*) maximising the always-on net saving over a grid of h (0 if nothing helps)."""
    grid = np.concatenate([[0.0], np.geomspace(0.005, 4.0, 60)]) if grid is None else np.asarray(grid, dtype=float)
    vals = np.array([net_build_saving(ci, phi, h, p0, slack, exact) for h in grid])
    i = int(np.argmax(vals))
    return float(grid[i]), float(vals[i])


def existing_spare(ci: np.ndarray, phi: float, h: float, p0: float, slack: int = 24) -> dict:
    """Uses of spare capacity that already exists (sunk), relative to the always-on, unshifted baseline.

    Returns savings in g per kWh of mean work and as a share of baseline emissions
    cbar * (1 + p0 * h):
      shift_only   (1 - p0) * S(h)          servers stay on, work moves
      sleep_only   p0 * h * cbar            spare servers powered down, no shifting
      sleep_shift  p0 * h * cbar + S(h)     power down whatever is not running, and shift
    """
    ci = np.asarray(ci, dtype=float)
    cbar = float(ci.mean())
    r = capped_shift(ci, phi, 1.0 + h, slack)
    S = r.baseline_g - r.optimised_g
    base = cbar * (1.0 + p0 * h)
    out = dict(S=S, shift_only=(1 - p0) * S, sleep_only=p0 * h * cbar, sleep_shift=p0 * h * cbar + S, baseline=base)
    for k in ("shift_only", "sleep_only", "sleep_shift"):
        out[k + "_pct"] = 100 * out[k] / base
    return out
