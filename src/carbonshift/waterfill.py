"""Closed-form laws for carbon-aware shifting under a facility power cap.

Setting (as facility.capped_shift): mean load 1, flexible share phi, 24 h slack,
hourly grid draw capped at ``cap`` = 1 + h. The flexible work can be served at up
to r = phi + h per hour, so each day's 24*phi units fill the k = 24*phi/r
cleanest hours ("water-filling"). Saving per hour of mean load:

    S = phi*mean(ci) - (r/24) * sum(lowest k hourly values)          (order statistics)

For a sinusoidal day ci = M - A*cos(2*pi*t/24) this has the closed form

    S = phi * A * sinc(pi*q),   q = phi / (phi + h),   sinc(x) = sin(x)/x

so the share of the uncapped saving that survives a cap is sinc(pi*q), a single
universal curve, and the carbon value of one extra unit of headroom is

    dS/dh = pi * A * q**2 * |sinc'(pi*q)|,

linear in the daily amplitude A and independent of the mean (the swing law).
Peak draw of the uncapped schedule is 1 + slack*phi: an hour can collect the flexible
arrivals of all slack + 1 hours whose window contains it (13x mean for phi = 0.5, 24 h).
"""

from __future__ import annotations

import numpy as np

HOURS_PER_YEAR = 8760


def sinc(x):
    x = np.asarray(x, dtype=float)
    return np.where(np.abs(x) < 1e-12, 1.0, np.sin(x) / np.where(x == 0, 1, x))


def dsinc(x):
    x = np.asarray(x, dtype=float)
    return np.where(np.abs(x) < 1e-6, -x / 3, (x * np.cos(x) - np.sin(x)) / np.where(x == 0, 1, x) ** 2)


def q_of(phi: float, cap) -> np.ndarray:
    """Fraction of the day the flexible work occupies at full rate (1 = no headroom, 0 = uncapped)."""
    h = np.asarray(cap, dtype=float) - 1.0
    return np.where(np.isinf(h), 0.0, phi / (phi + np.maximum(h, 0.0)))


def retained_sinc(phi: float, cap) -> np.ndarray:
    """Share of the uncapped saving kept under the cap, for a sinusoidal day."""
    return sinc(np.pi * q_of(phi, cap))


def saving_sinc(amplitude, phi: float, cap) -> np.ndarray:
    """Saving (g per kWh of mean load) for a sinusoidal day of amplitude A."""
    return phi * np.asarray(amplitude, dtype=float) * retained_sinc(phi, cap)


def value_per_kw_year_sinc(amplitude, phi: float, cap) -> np.ndarray:
    """Marginal value of headroom, g CO2 per kW of extra cap per year, for a sinusoidal day."""
    q = q_of(phi, cap)
    return np.pi * np.asarray(amplitude, dtype=float) * q ** 2 * np.abs(dsinc(np.pi * q)) * HOURS_PER_YEAR


def saving_waterfill(ci: np.ndarray, phi: float, cap) -> float:
    """Order-statistic saving per calendar day (exact when every day's work stays within its day)."""
    ci = np.asarray(ci, dtype=float)
    days = ci[: len(ci) // 24 * 24].reshape(-1, 24)
    srt = np.sort(days, axis=1)
    if np.isinf(cap):
        return float(phi * (days.mean() - srt[:, 0].mean()))
    r = phi + (cap - 1.0)
    if r <= phi + 1e-12:
        return 0.0
    k = 24 * phi / r
    kf = int(np.floor(k))
    filled = srt[:, :kf].sum(axis=1) + (k - kf) * (srt[:, kf] if kf < 24 else 0.0)
    return float((phi * days.sum(axis=1) - r * filled).mean() / 24)


def daily_amplitude(ci: np.ndarray, how: str = "std") -> float:
    """Effective sinusoidal amplitude of the daily cycle: sqrt(2) x mean daily std, or mean (mean - min)."""
    days = np.asarray(ci, dtype=float)[: len(ci) // 24 * 24].reshape(-1, 24)
    if how == "std":
        return float(np.sqrt(2) * days.std(axis=1, ddof=0).mean())
    return float((days.mean(axis=1) - days.min(axis=1)).mean())


def value_waterfill(ci: np.ndarray, phi: float, cap) -> float:
    """Marginal value of headroom (g per kW of extra cap per year) from daily water-filling.

    Raising the flexible rate r lets the same work fit in fewer, cleaner hours. Per day,
    dS/dr = sum over filled hours of (water level - price): the area between the
    water level (price of the marginal filled hour) and the price curve below it.
    """
    days = np.asarray(ci, dtype=float)[: len(ci) // 24 * 24].reshape(-1, 24)
    srt = np.sort(days, axis=1)
    r = phi + (cap - 1.0)
    if np.isinf(cap) or r <= 0:
        return 0.0
    k = 24 * phi / r
    kf = int(np.floor(k))
    level = srt[:, min(kf, 23)]
    full = (level[:, None] - srt[:, :kf]).sum(axis=1)  # the fractional hour sits at the level: zero gap
    return float(full.mean() / 24 * HOURS_PER_YEAR)


def lower_partial_mean(ci: np.ndarray, q: float, window: int = 24) -> float:
    """Mean over windows of the average of each window's cleanest fraction q of hours (q -> 0: the window minimum)."""
    w = np.asarray(ci, dtype=float)[: len(ci) // window * window].reshape(-1, window)
    srt = np.sort(w, axis=1)
    if q <= 0:
        return float(srt[:, 0].mean())
    k = q * window
    kf = int(np.floor(k))
    tot = srt[:, :kf].sum(axis=1) + (k - kf) * (srt[:, kf] if kf < window else 0.0)
    return float((tot / k).mean())


def saving_general(ci: np.ndarray, phi: float, cap, slack: int = 24) -> float:
    """General water-filling law for any slack: S = phi * (mean - L_slack(q)), q = phi / (phi + h).

    Equals saving_waterfill for slack = 24; the sinc law is its sinusoidal special case.
    """
    q = float(q_of(phi, cap))
    return float(phi * (np.mean(np.asarray(ci, dtype=float)[: len(ci) // slack * slack]) - lower_partial_mean(ci, q, slack)))


def effective_horizon(phi: float, cap, slack: int) -> int:
    """Planning horizon of the virtual battery (F22): the slack, or its refill time s*phi/h if longer."""
    h = float(cap) - 1.0
    if not np.isfinite(h) or h <= 0:
        return int(slack)
    return int(round(slack * max(1.0, phi / h)))


def saving_law(ci: np.ndarray, phi: float, cap, slack: int = 24) -> float:
    """Best closed-form estimate of the capped saving: water-filling over the effective horizon (no optimiser)."""
    return saving_general(ci, phi, cap, effective_horizon(phi, cap, slack))
