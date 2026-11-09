import numpy as np
import pytest

from carbonshift.llm_batching import BatchCurve, _carbon_aware_lp, carbon_aware, static_best

CURVE = BatchCurve(batch=np.array([1, 4, 16, 64]), energy_j=np.array([0.10, 0.05, 0.035, 0.03]),
                   latency_s=np.array([0.005, 0.008, 0.02, 0.05]))


def test_constant_carbon_static_equals_dynamic():
    ci = np.full(200, 300.0)
    s, _ = static_best(ci, CURVE, 0.012)
    d, _ = carbon_aware(ci, CURVE, 0.012)
    assert d == pytest.approx(s, rel=1e-9)


def test_dynamic_never_worse_and_budget_respected():
    ci = 300 + 150 * np.sin(np.arange(24 * 30) * 2 * np.pi / 24)
    budget = 0.012
    s, _ = static_best(ci, CURVE, budget)
    d, x = carbon_aware(ci, CURVE, budget)
    assert d <= s + 1e-9
    assert (x @ CURVE.latency_s).mean() <= budget + 1e-9
    big_in_dirty = (x @ CURVE.batch)[np.argmax(ci)] >= (x @ CURVE.batch)[np.argmin(ci)]
    assert big_in_dirty  # larger batches when the grid is dirtier


def test_budget_below_fastest_raises():
    with pytest.raises(ValueError):
        static_best(np.ones(10), CURVE, 0.001)


@pytest.mark.parametrize("budget", [0.006, 0.01, 0.02, 0.04])
def test_lagrangian_equals_lp(budget):
    ci = np.random.default_rng(int(budget * 1e4)).uniform(50, 700, 300)
    fast, x = carbon_aware(ci, CURVE, budget)
    assert fast == pytest.approx(_carbon_aware_lp(ci, CURVE, budget), rel=1e-6)
    assert (x @ CURVE.latency_s).mean() <= budget + 1e-9
