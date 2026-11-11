"""What a carbon-aware data centre's power meter reveals (F25).

Identity. With hourly grid draw d_t normalised to mean 1 and carbon intensity c_t,
the carbon saved against a flat draw is, per hour,

    mean(c) - mean(c * d) = -cov(c, d),

so *any* carbon saving leaves a negative covariance between the meter and the public
carbon signal. An observer who sees the meter (utility, co-located tenant, anyone with
a clamp meter on the feed) and the public signal can therefore test for
carbon-awareness and, through the virtual-battery theorem (F22), read the facility's
backlog of deferred work off the meter: B_t = cumsum(1 - d_t) up to a constant, so the
battery's energy s*phi, the inflexible floor 1 - phi and the cap 1 + h all show up.

Detection uses day-to-day variation only: both series are residualised against their
own hour-of-day means inside the observation window, so an ordinary diurnal load
(busy by day) that happens to align with a solar grid is not flagged.
"""

from __future__ import annotations

import numpy as np


def saving_from_cov(ci: np.ndarray, draw: np.ndarray) -> float:
    """Saving per hour (g per kWh of mean load) against a flat draw, as -cov(c, d) with d scaled to mean 1."""
    c = np.asarray(ci, dtype=float)
    d = np.asarray(draw, dtype=float)
    d = d / d.mean()
    return float(-np.mean((c - c.mean()) * (d - 1.0)))


def _resid_hod(x: np.ndarray, hours: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    means = np.bincount(hours, weights=x, minlength=24) / np.maximum(np.bincount(hours, minlength=24), 1)
    return x - means[hours]


def detection_stat(ci: np.ndarray, draw: np.ndarray, hours: np.ndarray) -> float:
    """Correlation of hour-of-day residuals of draw and carbon (strongly negative = carbon-aware)."""
    rc, rd = _resid_hod(ci, hours), _resid_hod(draw, hours)
    den = np.sqrt((rc ** 2).sum() * (rd ** 2).sum())
    return float((rc * rd).sum() / den) if den > 0 else 0.0


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability that a carbon-aware facility scores lower (more negative) than a non-aware one."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    lt = (pos[:, None] < neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(lt + 0.5 * eq)


def infer_battery(draw: np.ndarray, floor_q: float = 0.01, top_q: float = 0.99) -> dict:
    """Estimate the virtual battery from the meter alone (draw normalised to mean 1).

    phi_hat = 1 - low quantile of draw (the inflexible floor when all flexible work waits),
    h_hat   = high quantile - 1 (the cap the catch-up runs into),
    E_hat   = median weekly range of the reconstructed backlog cumsum(1 - d)
              (the deferred work the facility actually carries), slack_hat = E_hat / phi_hat.
    """
    d = np.asarray(draw, dtype=float)
    d = d / d.mean()
    phi = float(1.0 - np.quantile(d, floor_q))
    h = float(np.quantile(d, top_q) - 1.0)
    B = np.cumsum(1.0 - d)
    weeks = B[: len(B) // 168 * 168].reshape(-1, 168)
    E = float(np.median(weeks.max(axis=1) - weeks.min(axis=1))) if len(weeks) else float(B.max() - B.min())
    return {"phi_hat": phi, "h_hat": h, "energy_hat": E, "slack_hat": E / phi if phi > 1e-6 else float("nan")}


def time_of_day_signal(ci: np.ndarray, hours: np.ndarray, months: np.ndarray) -> np.ndarray:
    """A 'privacy-preserving' schedule signal: each month's average carbon at that hour of day."""
    ci = np.asarray(ci, dtype=float)
    key = months * 24 + hours
    sums = np.bincount(key, weights=ci, minlength=13 * 24)
    cnt = np.maximum(np.bincount(key, minlength=13 * 24), 1)
    return (sums / cnt)[key]
