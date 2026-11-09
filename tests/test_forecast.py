import numpy as np
import pytest

from carbonshift.forecast import noisy_forecast, regret, true_emissions_with_forecast
from carbonshift.facility import capped_shift


def _trace(n=24 * 40, seed=0):
    t = np.arange(n)
    return 300 + 120 * np.sin(2 * np.pi * t / 24) + np.random.default_rng(seed).normal(0, 15, n)


def test_forecast_mape_calibrated_and_autocorrelated():
    ci = np.full(200_000, 400.0)
    f = noisy_forecast(ci, 0.10, np.random.default_rng(1), rho=0.9)
    e = f / ci - 1
    assert np.mean(np.abs(e)) == pytest.approx(0.10, rel=0.05)
    assert np.corrcoef(e[:-1], e[1:])[0, 1] == pytest.approx(0.9, abs=0.02)


def test_zero_error_is_perfect():
    ci = _trace()
    r = regret(ci, 0.0, 0.5, 1.3, 24, np.random.default_rng(0))
    assert r["lost_share"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("cap", [1.2, np.inf])
def test_forecast_never_beats_perfect(cap):
    ci = _trace(seed=4)
    perfect = capped_shift(ci, 0.5, cap, 24).optimised_g
    for seed in range(3):
        f = noisy_forecast(ci, 0.2, np.random.default_rng(seed))
        assert true_emissions_with_forecast(ci, f, 0.5, cap, 24) >= perfect - 1e-9
