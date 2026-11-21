import numpy as np
import pytest
from scipy.optimize import minimize

from carbonshift.smoothknob import (kappa, loglog_slope, optimal_speeds, saving_harberger,
                                    saving_power_mean, saving_speed_scaling)
from carbonshift.waterfill import saving_waterfill


def _day(amp, days=30, seed=0, mean=400.0, noise=0.02):
    rng = np.random.default_rng(seed)
    t = np.arange(24 * days)
    return mean * (1 - amp * np.cos(2 * np.pi * t / 24)) * (1 + noise * rng.standard_normal(len(t)))


@pytest.mark.parametrize("alpha,smax", [(3.0, np.inf), (2.0, 1.3), (1.5, 1.1)])
def test_bisection_matches_generic_solver(alpha, smax):
    rng = np.random.default_rng(1)
    c = rng.uniform(50, 800, 24)
    s = optimal_speeds(c[None], alpha, 0.0, smax)[0]
    assert abs(s.mean() - 1) < 1e-9 and s.max() <= smax + 1e-12
    f = lambda x: (c * x ** alpha).sum()
    r = minimize(f, np.ones(24), method="SLSQP", bounds=[(0, None if np.isinf(smax) else smax)] * 24,
                 constraints=[{"type": "eq", "fun": lambda x: x.mean() - 1}], options={"ftol": 1e-14, "maxiter": 500})
    assert f(s) <= r.fun * (1 + 1e-7)
    assert f(s) == pytest.approx(r.fun, rel=1e-6)


def test_uncapped_equals_power_mean_closed_form():
    c = _day(0.3)
    for a in (1.5, 2.0, 3.0, 5.0):
        assert saving_speed_scaling(c, a) == pytest.approx(saving_power_mean(c, a), abs=1e-10)


def test_harberger_law_small_variation():
    c = _day(0.05)
    for a in (1.5, 2.0, 3.0):
        assert saving_harberger(c, a) == pytest.approx(saving_speed_scaling(c, a), rel=0.03)
    assert kappa(3.0) == pytest.approx(1.5)
    # static power only enlarges the baseline
    assert saving_speed_scaling(c, 3.0, p0=0.5) == pytest.approx(saving_speed_scaling(c, 3.0) / 1.5, rel=1e-9)
    assert saving_harberger(c, 3.0, p0=0.5) == pytest.approx(saving_harberger(c, 3.0) / 1.5, rel=1e-9)


def test_constant_and_scale_invariance():
    assert saving_speed_scaling(np.full(48, 300.0), 3.0) == pytest.approx(0.0, abs=1e-12)
    c = _day(0.2)
    assert saving_speed_scaling(3 * c, 2.0, s_max=1.2) == pytest.approx(saving_speed_scaling(c, 2.0, s_max=1.2), rel=1e-9)


def test_alpha_to_one_recovers_deferral():
    """Near-linear power: speed scaling under a speed cap becomes water-filling (deferral)."""
    c = _day(0.3)
    for cap in (1.2, 1.5):
        defer = saving_waterfill(c, 1.0, cap) / c.mean()
        assert saving_speed_scaling(c, 1.001, s_max=cap) == pytest.approx(defer, rel=0.02)


def test_second_order_vs_first_order_scaling():
    amps = np.array([0.02, 0.04, 0.08, 0.16])
    days = [_day(a, noise=0.0) for a in amps]
    dv = [saving_speed_scaling(c, 3.0) for c in days]
    de = [saving_waterfill(c, 1.0, 1.2) / c.mean() for c in days]
    assert loglog_slope(amps, dv) == pytest.approx(2.0, abs=0.1)
    assert loglog_slope(amps, de) == pytest.approx(1.0, abs=0.1)
