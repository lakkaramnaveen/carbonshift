"""Forecast error for carbon-aware shifting: schedule on a forecast, pay the true intensity.

The forecast is ci * (1 + e), where e is a zero-mean Gaussian AR(1) process
(autocorrelation ``rho`` per hour) scaled so that E|e| equals the target MAPE.
For a normal distribution E|e| = sigma * sqrt(2/pi). Autocorrelated errors are
realistic: forecasts tend to be wrong for several hours in a row, not one hour
at a time. The schedule optimised for the forecast is always feasible for the
true problem (the constraints do not depend on carbon), so its true emissions
are never below the perfect-information optimum.
"""

from __future__ import annotations

import numpy as np

from carbonshift.facility import capped_shift


def noisy_forecast(ci: np.ndarray, mape: float, rng: np.random.Generator, rho: float = 0.9) -> np.ndarray:
    ci = np.asarray(ci, dtype=float)
    if mape <= 0:
        return ci.copy()
    sigma = mape / np.sqrt(2 / np.pi)
    innov = rng.normal(0.0, sigma * np.sqrt(1 - rho**2), len(ci))
    e = np.empty(len(ci))
    e[0] = rng.normal(0.0, sigma)
    for t in range(1, len(ci)):
        e[t] = rho * e[t - 1] + innov[t]
    return np.maximum(ci * (1 + e), 0.0)


def true_emissions_with_forecast(ci: np.ndarray, forecast: np.ndarray, phi: float, cap, slack: int) -> float:
    """Mean g/kWh actually emitted when the schedule is optimised on ``forecast``."""
    r = capped_shift(forecast, phi, cap, slack)
    return float((np.asarray(ci) * ((1 - phi) + r.served)).sum() / len(ci))


def regret(ci: np.ndarray, mape: float, phi: float, cap, slack: int, rng: np.random.Generator) -> dict:
    """Share of the perfect-information saving lost when scheduling on a forecast with this MAPE."""
    ci = np.asarray(ci, dtype=float)
    perfect = capped_shift(ci, phi, cap, slack)
    base = perfect.baseline_g
    best_saving = base - perfect.optimised_g
    got = base - true_emissions_with_forecast(ci, noisy_forecast(ci, mape, rng), phi, cap, slack)
    return {"baseline_g": base, "perfect_saving_g": best_saving, "forecast_saving_g": got,
            "lost_share": (best_saving - got) / best_saving if best_saving > 1e-12 else np.nan}
