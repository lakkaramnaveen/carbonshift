import numpy as np
import pytest

from carbonshift.cooling import PLANTS, decompose, penalty_law, pue
from carbonshift.facility import capped_shift

L = 24 * 90
H = np.arange(L) % 24


def _solar_grid():
    """Clean at midday (hour 12), hot at midday: the solar case."""
    rng = np.random.default_rng(1)
    ci = 400 - 100 * np.cos(2 * np.pi * (H - 12) / 24) + rng.normal(0, 5, L)
    temp = 22 + 8 * np.cos(2 * np.pi * (H - 14) / 24) + rng.normal(0, 0.5, L)
    return ci, temp


@pytest.mark.parametrize("name", list(PLANTS))
def test_pue_rises_with_temperature_and_floors(name):
    t = np.linspace(-20, 45, 200)
    p = pue(t, PLANTS[name])
    assert np.all(np.diff(p) >= -1e-12)
    plant = PLANTS[name]
    assert p.min() >= 1 + plant.fixed + plant.fans - 1e-12
    assert 1.05 < p[t.searchsorted(20)] < 1.8


def test_economiser_gives_free_cooling():
    plant = PLANTS["hyperscale"]
    p = pue(np.array([0.0, plant.econ_full]), plant)
    assert np.allclose(p, 1 + plant.fixed + plant.fans)


def test_constant_pue_has_no_penalty_and_same_schedule():
    rng = np.random.default_rng(0)
    ci = 300 + 80 * np.cos(2 * np.pi * H / 24) + rng.normal(0, 10, L)
    p = np.full(L, 1.3)
    a = capped_shift(ci, 0.5, 1.2, 24)
    b = capped_shift(ci * p, 0.5, 1.2, 24)
    d = 0.5 + a.served
    out = decompose(ci, p, d)
    assert out["penalty"] == pytest.approx(0, abs=1e-9)
    assert out["true"] == pytest.approx(1.3 * (a.baseline_g - a.optimised_g), rel=1e-9)
    assert np.allclose(a.served, b.served, atol=1e-6)
    assert penalty_law(ci, p) == 0.0


def test_true_saving_is_minus_cov_of_cooled_carbon():
    ci, temp = _solar_grid()
    p = pue(temp, PLANTS["legacy"])
    d = 0.5 + capped_shift(ci, 0.5, 1.3, 24).served
    out = decompose(ci, p, d)
    assert out["true"] == pytest.approx(-np.cov(ci * p, d, bias=True)[0, 1], rel=1e-9)
    assert out["claimed"] == pytest.approx(-p.mean() * np.cov(ci, d, bias=True)[0, 1], rel=1e-9)


def test_solar_grid_overstates_and_cooling_aware_recovers():
    ci, temp = _solar_grid()
    p = pue(temp, PLANTS["legacy"])
    d_c = 0.5 + capped_shift(ci, 0.5, 1.3, 24).served
    d_w = 0.5 + capped_shift(ci * p, 0.5, 1.3, 24).served
    a, b = decompose(ci, p, d_c), decompose(ci, p, d_w)
    assert a["penalty"] > 0                 # clean hours are hot: the claim is too high
    assert b["true"] >= a["true"] - 1e-9    # optimising the right signal never does worse
    law = penalty_law(ci, p)
    assert law > 0
    assert law == pytest.approx(a["penalty"] / a["claimed"], rel=0.35)


def test_flat_grid_still_saves_by_weather():
    _, temp = _solar_grid()
    ci = np.full(L, 500.0)
    p = pue(temp, PLANTS["legacy"])
    d_w = 0.5 + capped_shift(ci * p, 0.5, 1.3, 24).served
    assert decompose(ci, p, d_w)["true"] > 1.0
    assert np.isnan(penalty_law(ci, p))


def test_decompose_rejects_unnormalised_draw():
    with pytest.raises(ValueError):
        decompose(np.ones(24), np.ones(24), np.full(24, 2.0))


@pytest.mark.parametrize("year", [2020, 2021, 2022])
def test_weather_aligns_with_carbon(year):
    from carbonshift.cooling import HUBS, load_weather
    from carbonshift.data import load_carbon

    w, ci = load_weather(year), load_carbon(year)
    assert len(w) == len(ci)
    assert (w.index == ci.index).all()                 # both hourly UTC, same stamps
    assert w.notna().all().all()
    for z in HUBS:
        assert (w[f"{z}|wb"] <= w[f"{z}|db"] + 0.05).all()   # wet bulb never above dry bulb
