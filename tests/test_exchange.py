import numpy as np
import pytest

from carbonshift.exchange import exchange_rate, spread_saving_per_kwh
from carbonshift.facility import battery_shift


def _trace(days=30, seed=0):
    t = np.arange(24 * days)
    return 400 - 150 * np.cos(2 * np.pi * t / 24) + np.random.default_rng(seed).normal(0, 5, len(t))


@pytest.mark.parametrize("duration", [2.0, 4.0])
def test_spread_law_matches_battery_lp_on_daily_cycle(duration):
    ci = _trace()
    b = battery_shift(ci, energy_h=1.0, power=1.0 / duration, round_trip_eff=0.9)
    assert spread_saving_per_kwh(ci, duration, 0.9) == pytest.approx(b.baseline_g - b.optimised_g, rel=0.03)


def test_battery_saving_is_linear_in_capacity():
    ci = _trace(seed=2)
    one = battery_shift(ci, energy_h=1.0, power=0.25)
    three = battery_shift(ci, energy_h=3.0, power=0.75)
    assert three.baseline_g - three.optimised_g == pytest.approx(3 * (one.baseline_g - one.optimised_g), rel=1e-6)


def test_flat_grid_gives_battery_nothing():
    assert spread_saving_per_kwh(np.full(240, 300.0), 4.0, 0.9) == 0.0
    assert exchange_rate(5.0, 0.0) == float("inf")
    assert exchange_rate(0.0, 0.0) == 0.0
    assert exchange_rate(6.0, 2.0) == pytest.approx(3.0)
