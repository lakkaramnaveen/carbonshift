"""F35: who produced the saving? Euler attribution between flexibility and headroom.

Carbon-aware shifting needs two inputs: flexible work (phi, supplied by users who
accept delay) and spare power capacity (h, supplied by the operator). The field
credits the saving to the *workload's* flexibility ("this job saved X g by waiting").

Under a power cap the capped-shifting saving S(phi, h) is homogeneous of degree one
in (phi, h) (F21 homogeneity theorem). Euler's theorem (Wicksteed-Flux product
exhaustion, 1894) then splits it exactly into marginal products:

    S = phi * dS/dphi + h * dS/dh,

and for a homogeneous function this split is also the Aumann-Shapley value of the
two inputs. The capacity share is  sigma_h = h * dS/dh / S  and the flexibility share
1 - sigma_h. Consequences:

* a marginal flexible kW is worth (1 - sigma_h) of the average flexible kW, so the
  sum of every job's counterfactual saving ("what if this job were not flexible")
  is only (1 - sigma_h) of the facility's saving;
* per-job carbon claims at the average rate overstate the counterfactual by
  1 / (1 - sigma_h).

Sinusoidal day (F21: S = phi A sinc(pi q), q = phi / (phi + h)):

    sigma_h(q) = (1 - q) * (1 - pi q cot(pi q))

which -> 1 as h -> 0 (headroom is the scarce input) and ~ (pi q)^2 / 3 -> 0 as
h -> infinity (the uncapped world of the EuroSys'24 bounds, where flexibility is
everything). It depends only on h / phi.

Real hourly signals: with 24 h calendar-day water-filling (waterfill.saving_waterfill),
r = phi + h, k = 24 phi / r cleanest hours filled at rate r, and water level
lambda = the k-th cleanest price,

    dS/dphi|_h = mean(c) - E[lambda] + dS/dr,     dS/dh = dS/dr = E[sum_filled (lambda - c)] / 24,

both in g per kWh of mean load per unit of input (``euler_waterfill``). The LP route
(``euler_lp``) gets dS/dh from the cap duals and dS/dphi by finite differences, an
independent check.
"""

from __future__ import annotations

import numpy as np

from .facility import capped_shift
from .waterfill import q_of, saving_waterfill


def capacity_share_sinc(phi: float, h) -> np.ndarray:
    """Euler capacity share sigma_h for a sinusoidal day (closed form)."""
    q = np.asarray(q_of(phi, 1.0 + np.asarray(h, dtype=float)), dtype=float)
    x = np.pi * q
    with np.errstate(divide="ignore", invalid="ignore"):
        xcot = np.where(np.abs(x) < 1e-8, 1.0 - x ** 2 / 3, x * np.cos(x) / np.sin(x))
    out = (1.0 - q) * (1.0 - xcot)
    return np.where(q >= 1.0 - 1e-12, 1.0, out)


def _days(ci):
    ci = np.asarray(ci, dtype=float)
    return np.sort(ci[: len(ci) // 24 * 24].reshape(-1, 24), axis=1)


def euler_waterfill(ci: np.ndarray, phi: float, h: float) -> dict:
    """Analytic marginal products of phi and h under daily water-filling.

    Returns S, S_phi, S_h (g per kWh of mean load, per unit of input), the Euler
    residual S - phi S_phi - h S_h, and the capacity share sigma_h.
    """
    if h <= 0:
        raise ValueError("h must be > 0 (with h = 0 there is no shifting and no saving)")
    srt = _days(ci)
    r = phi + h
    k = 24.0 * phi / r
    kf = int(np.floor(k))
    level = srt[:, min(kf, 23)]
    S = saving_waterfill(ci, phi, 1.0 + h)
    S_r = float((level[:, None] - srt[:, :kf]).sum(axis=1).mean() / 24)
    S_phi_r = float(srt.mean() - level.mean())  # d/dphi at fixed rate r
    S_phi = S_phi_r + S_r                         # at fixed h, r = phi + h moves too
    S_h = S_r
    sigma = h * S_h / S if S > 0 else np.nan
    return dict(S=S, S_phi=S_phi, S_h=S_h, residual=S - phi * S_phi - h * S_h, sigma_h=sigma)


def euler_fd(ci: np.ndarray, phi: float, h: float, eps: float = 1e-4) -> dict:
    """Same quantities by central finite differences of saving_waterfill (numerical check)."""
    f = lambda p, hh: saving_waterfill(ci, p, 1.0 + hh)  # noqa: E731
    S = f(phi, h)
    S_phi = (f(phi + eps, h) - f(phi - eps, h)) / (2 * eps)
    S_h = (f(phi, h + eps) - f(phi, h - eps)) / (2 * eps)
    return dict(S=S, S_phi=S_phi, S_h=S_h, residual=S - phi * S_phi - h * S_h,
                sigma_h=h * S_h / S if S > 0 else np.nan)


def euler_lp(ci: np.ndarray, phi: float, h: float, slack: int = 24, eps: float = 1e-3) -> dict:
    """Independent route: sliding-window LP (facility.capped_shift).

    S_h from the LP duals of the cap (shadow price summed over hours, per hour of
    mean load); S_phi by a central finite difference in phi at fixed h.
    """
    def sav(p):
        r = capped_shift(ci, p, 1.0 + h, slack)
        return r.baseline_g - r.optimised_g, r

    S, res = sav(phi)
    S_h = float(np.sum(res.shadow) / len(ci))
    S_phi = (sav(phi + eps)[0] - sav(phi - eps)[0]) / (2 * eps)
    return dict(S=S, S_phi=S_phi, S_h=S_h, residual=S - phi * S_phi - h * S_h,
                sigma_h=h * S_h / S if S > 0 else np.nan)


def overclaim_factor(sigma_h) -> np.ndarray:
    """Average-rate per-job claim divided by the sum of per-job counterfactuals: 1 / (1 - sigma_h)."""
    s = np.asarray(sigma_h, dtype=float)
    return 1.0 / np.maximum(1.0 - s, 1e-12)
