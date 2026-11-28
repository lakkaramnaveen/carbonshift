import numpy as np
import pytest

from carbonshift import provenance as pv
from carbonshift.facility import capped_shift
from carbonshift.waterfill import saving_waterfill

rng = np.random.default_rng(0)
DAYS = 200
T = np.arange(24 * DAYS)


def _signal(noise=20.0, seed=1):
    r = np.random.default_rng(seed)
    return 400 - 80 * np.cos(2 * np.pi * (T % 24) / 24) + r.normal(0, noise, T.size)


@pytest.mark.parametrize("cap", [1.2, 1.42, np.inf])
def test_schedule_matches_order_statistic_saving(cap):
    c = _signal()
    x = pv.waterfill_schedule(c, 0.5, cap)
    assert np.allclose(x.reshape(-1, 24).sum(axis=1), 12.0)
    assert x.max() <= (12.0 if np.isinf(cap) else 0.5 + cap - 1) + 1e-12
    assert pv.saving_of(c, x, 0.5) == pytest.approx(saving_waterfill(c, 0.5, cap), rel=1e-12)


def test_affine_invariance_keeps_all():
    c = _signal()
    r = pv.cross_evaluate(c, 50 + 0.3 * c, 0.5, 1.2)
    assert r.rho == pytest.approx(1.0)
    assert r.kept_b == pytest.approx(1.0)
    assert r.confirmed_a == pytest.approx(0.3)  # the claim scales with the slope, the decision does not


def test_kept_equals_rho_for_gaussian_shapes():
    """Jointly Gaussian within-day shapes: E[S_B(x_A)] = rho * S_B(x_B)."""
    r_ = np.random.default_rng(3)
    for rho in (0.9, 0.5, 0.0):
        z1 = r_.normal(size=T.size)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * r_.normal(size=T.size)
        a, b = 300 + 40 * z1, 500 + 25 * z2
        res = pv.cross_evaluate(a, b, 0.5, 1.2)
        assert res.kept_b == pytest.approx(rho, abs=0.04)
        assert res.confirmed_a == pytest.approx(res.beta, abs=0.04)


def test_negative_correlation_raises_emissions():
    a = _signal(noise=1.0)
    res = pv.cross_evaluate(a, 1000 - a, 0.5, 1.2)
    assert res.a_on_b < 0 and res.kept_b == pytest.approx(-1.0, abs=0.01)


def test_constant_signal_has_no_saving():
    c = np.full(T.size, 24.0)
    x = pv.waterfill_schedule(_signal(), 0.5, 1.2)
    assert pv.saving_of(c, x, 0.5) == pytest.approx(0.0, abs=1e-12)


def test_lp_route_agrees_with_daily_waterfill_on_cross_evaluation():
    """Independent route: sliding-window LP schedule evaluated on a second signal."""
    a = _signal(seed=4)[: 24 * 40]
    b = (0.7 * a + _signal(noise=30, seed=5)[: 24 * 40] * 0.3)
    wf = pv.cross_evaluate(a, b, 0.5, 1.2)
    ra, rb = capped_shift(a, 0.5, 1.2, 24), capped_shift(b, 0.5, 1.2, 24)
    kept_lp = ((b * 0.5).mean() - (b * ra.served).mean()) / ((b * 0.5).mean() - (b * rb.served).mean())
    assert kept_lp == pytest.approx(wf.kept_b, abs=0.05)
