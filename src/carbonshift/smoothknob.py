"""Smooth knobs: carbon-aware speed scaling (DVFS, GPU power limits) vs deferral (F29).

Setting. A server (or fleet) must do a fixed amount of work in each window (a day,
i.e. 24 h slack). It may run at speed s_t (work per hour, mean 1) and draws dynamic
power P(s) = s**alpha, alpha > 1 (frequency x voltage**2; alpha ~ 2-3 for DVFS).
Carbon-aware speed scaling solves

    min  sum_t c_t * s_t**alpha    s.t.  mean_t s_t = 1,  s_min <= s_t <= s_max.

KKT: s_t = clip((lam / c_t)**(1/(alpha-1)), s_min, s_max), with lam set so the mean
is 1 (found here by vectorised bisection).

Uncapped closed form (the power-mean law). With no speed bounds the optimal cost per
hour is M_p(c), the power mean of the signal with exponent p = -1/(alpha-1), so the
saving relative to constant speed is

    saving = 1 - M_p(c) / M_1(c).

alpha -> 1 gives p -> -inf, M_p -> min(c): this is deferral (all work in the cleanest
hour). alpha -> inf gives the geometric mean.

Harberger law (second order). For any smooth convex power curve P and small
relative variation, the gain over the best constant speed is

    saving ~= (kappa / 2) * CV**2,   kappa = P'(1)**2 / (P(1) * P''(1)),

(alpha/(alpha-1) for P = s**alpha) where CV is the within-window coefficient of
variation. Like a Harberger deadweight triangle, the gain is quadratic in the
'distortion' (carbon variation) because the cost of deviating from the efficient
constant speed is itself quadratic. Deferral on an energy-proportional server is the
linear case (P'' = 0, kappa = inf): its gain is first order in the swing (F15, F21)
and is limited by the power cap instead.
"""

from __future__ import annotations

import numpy as np

EPS_CI = 1.0  # g/kWh floor so power means with p < 0 stay finite on zero-carbon hours


def _windows(ci: np.ndarray, window: int) -> np.ndarray:
    ci = np.maximum(np.asarray(ci, dtype=float), EPS_CI)
    n = len(ci) // window * window
    return ci[:n].reshape(-1, window)


def optimal_speeds(c: np.ndarray, alpha: float, s_min: float = 0.0, s_max: float = np.inf,
                   iters: int = 200) -> np.ndarray:
    """Exact optimal speeds for each row of c (rows = windows), mean speed 1 per row."""
    c = np.atleast_2d(np.maximum(np.asarray(c, dtype=float), EPS_CI))
    if alpha <= 1:
        raise ValueError("alpha must exceed 1 (alpha = 1 is deferral; use waterfill)")
    if not (s_min <= 1 <= s_max):
        raise ValueError("need s_min <= 1 <= s_max")
    e = 1.0 / (alpha - 1.0)
    # log s_t = e * (log lam - log c_t). At lam = min(c) every s_t <= 1 (mean <= 1); at
    # lam = max(c) every s_t >= 1 (mean >= 1); clipping to [s_min, s_max] keeps both.
    logc = np.log(c)
    lo, hi = logc.min(axis=1), logc.max(axis=1)

    def speeds(loglam):
        return np.clip(np.exp(np.clip(e * (loglam[:, None] - logc), -700, 700)), s_min, s_max)

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        too_much = speeds(mid).mean(axis=1) > 1.0
        hi = np.where(too_much, mid, hi)
        lo = np.where(too_much, lo, mid)
    return speeds(0.5 * (lo + hi))


def saving_speed_scaling(ci: np.ndarray, alpha: float, s_min: float = 0.0, s_max: float = np.inf,
                         window: int = 24, p0: float = 0.0) -> float:
    """Fractional carbon saving of carbon-aware speed scaling vs constant speed 1.

    p0 is static power (fraction of dynamic power at speed 1) drawn in every hour
    regardless of speed; it adds to the baseline but not to the gain.
    """
    c = _windows(ci, window)
    s = optimal_speeds(c, alpha, s_min, s_max)
    base = (c * (1.0 + p0)).sum()
    opt = (c * (p0 + s ** alpha)).sum()
    return float(1.0 - opt / base)


def power_mean(c: np.ndarray, p: float, axis=-1) -> np.ndarray:
    c = np.maximum(np.asarray(c, dtype=float), EPS_CI)
    if p == 0:
        return np.exp(np.log(c).mean(axis=axis))
    return (np.mean(c ** p, axis=axis)) ** (1.0 / p)


def saving_power_mean(ci: np.ndarray, alpha: float, window: int = 24) -> float:
    """Closed-form uncapped saving: 1 - sum_d T*M_p(c_d) / sum_d T*M_1(c_d)."""
    c = _windows(ci, window)
    p = -1.0 / (alpha - 1.0)
    return float(1.0 - power_mean(c, p).sum() / c.mean(axis=1).sum())


def kappa(alpha: float, p0: float = 0.0) -> float:
    """P'^2 / (P P'') at s = 1 for P(s) = p0 + s**alpha."""
    return alpha ** 2 / ((1.0 + p0) * alpha * (alpha - 1.0))


def saving_harberger(ci: np.ndarray, alpha: float, window: int = 24, p0: float = 0.0) -> float:
    """Second-order law: sum_d (kappa/2) * T * var_d / mean_d  /  sum_d T * mean_d."""
    c = _windows(ci, window)
    mu = c.mean(axis=1)
    var = c.var(axis=1)
    return float(0.5 * kappa(alpha, p0) * (var / mu).sum() / mu.sum())


def within_window_cv(ci: np.ndarray, window: int = 24) -> float:
    """Energy-weighted RMS within-window CV, the statistic in the Harberger law."""
    c = _windows(ci, window)
    mu = c.mean(axis=1)
    return float(np.sqrt((c.var(axis=1) / mu).sum() / mu.sum()))


def loglog_slope(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = (x > 0) & (y > 0)
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])
