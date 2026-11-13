import numpy as np
import pytest

from carbonshift.cooling import PLANTS, pue
from carbonshift.thermal import mech_per_heat, tes_shift

L = 24 * 30
H = np.arange(L) % 24


def test_mech_is_the_weather_part_of_pue():
    t = np.linspace(-10, 40, 101)
    for plant in PLANTS.values():
        assert np.allclose(1 + plant.fixed + plant.fans + mech_per_heat(t, plant), pue(t, plant))


def test_no_tank_or_flat_price_saves_nothing():
    ci = 300 + 100 * np.cos(2 * np.pi * H / 24)
    m = np.full(L, 0.2)
    assert tes_shift(ci, m, 0).saving_g == 0
    flat = tes_shift(np.full(L, 400.0), m, 6, loss_per_day=0.0)
    assert flat.saving_g == pytest.approx(0, abs=1e-6)


def test_two_price_day_moves_all_cold_to_cheap_hours():
    # cheap 12 h, dear 12 h; a lossless 12 h tank with enough chiller capacity
    ci = np.where(H < 12, 100.0, 500.0)
    m = np.full(L, 0.25)
    r = tes_shift(ci, m, 12, q_max=2.0, loss_per_day=0.0)
    assert r.optimised_g == pytest.approx(100 * 0.25, rel=1e-6)
    assert r.produced[H >= 12].max() < 1e-6


def test_saving_grows_with_capacity_and_energy_balances():
    rng = np.random.default_rng(0)
    ci = 400 - 120 * np.cos(2 * np.pi * (H - 12) / 24) + rng.normal(0, 10, L)
    temp = 22 + 8 * np.cos(2 * np.pi * (H - 14) / 24)
    m = mech_per_heat(temp, PLANTS["legacy"])
    s = [tes_shift(ci, m, c).saving_g for c in (0, 1, 2, 4, 8)]
    assert all(b >= a - 1e-9 for a, b in zip(s, s[1:]))
    r = tes_shift(ci, m, 4, loss_per_day=0.02)
    lost = (0.02 / 24) * np.roll(r.state, 1).sum()
    assert r.produced.sum() == pytest.approx(L + lost, rel=1e-6)   # heat met plus standing loss


def test_chiller_must_cover_peak():
    with pytest.raises(ValueError):
        tes_shift(np.ones(L), np.ones(L), 2, q_max=0.5)
