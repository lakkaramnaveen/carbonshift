import numpy as np
import pytest

from carbonshift.facility import capped_shift
from carbonshift.waterfill import (daily_amplitude, q_of, retained_sinc, saving_sinc, saving_waterfill,
                                   value_per_kw_year_sinc)


def _cosine_days(days=20, A=100.0, M=400.0, res=1):
    t = np.arange(24 * days * res) / res
    return M - A * np.cos(2 * np.pi * t / 24)


def test_retained_limits():
    assert retained_sinc(0.5, 1.0) == pytest.approx(0.0, abs=1e-12)    # no headroom: nothing moves
    assert retained_sinc(0.5, np.inf) == pytest.approx(1.0)              # uncapped: full saving
    assert q_of(0.5, 1.5) == pytest.approx(0.5)


def test_waterfill_matches_lp_on_periodic_signal():
    ci = _cosine_days(days=60) + np.tile(np.random.default_rng(0).normal(0, 3, 24), 60)  # same shape every day; long horizon
    for cap in (1.1, 1.3, 1.6, 2.5):
        lp = capped_shift(ci, 0.5, cap, 24)
        assert saving_waterfill(ci, 0.5, cap) == pytest.approx(lp.baseline_g - lp.optimised_g, rel=0.02)  # gap ~ 1/days (horizon edge)


def test_sinc_law_matches_waterfill_for_cosine():
    ci = _cosine_days(days=2, A=80.0, res=60)        # fine resolution removes hourly discretisation
    for cap in (1.05, 1.2, 1.5, 2.0, 4.0):
        # water-fill on a 1-minute grid: scale phi-units per sample consistently by treating samples as hours of a 1440 h "day"
        days = ci.reshape(-1, 1440)
        srt = np.sort(days, axis=1)
        r = 0.5 + cap - 1
        k = 1440 * 0.5 / r
        kf = int(k)
        filled = srt[:, :kf].sum(axis=1) + (k - kf) * srt[:, kf]
        s = (0.5 * days.sum(axis=1) - r * filled).mean() / 1440
        assert s == pytest.approx(float(saving_sinc(80.0, 0.5, cap)), rel=1e-3)


def test_value_is_derivative_of_saving_and_linear_in_amplitude():
    for cap in (1.1, 1.3, 2.0):
        dh = 1e-5
        num = (saving_sinc(50.0, 0.5, cap + dh) - saving_sinc(50.0, 0.5, cap - dh)) / (2 * dh) * 8760
        assert float(value_per_kw_year_sinc(50.0, 0.5, cap)) == pytest.approx(float(num), rel=1e-5)
    assert float(value_per_kw_year_sinc(100.0, 0.5, 1.2)) == pytest.approx(2 * float(value_per_kw_year_sinc(50.0, 0.5, 1.2)))


def test_daily_amplitude_of_cosine():
    ci = _cosine_days(A=60.0)
    assert daily_amplitude(ci, "std") == pytest.approx(60.0, rel=1e-6)
    assert daily_amplitude(ci, "min") == pytest.approx(60.0, rel=1e-6)


def test_value_waterfill_is_derivative_of_saving_waterfill():
    rng = np.random.default_rng(5)
    ci = 300 + 90 * np.sin(np.arange(24 * 30) * 2 * np.pi / 24) + rng.normal(0, 20, 24 * 30)
    from carbonshift.waterfill import value_waterfill
    for cap in (1.13, 1.37, 1.9):   # away from breakpoints where k is an integer
        dh = 1e-6
        num = (saving_waterfill(ci, 0.5, cap + dh) - saving_waterfill(ci, 0.5, cap - dh)) / (2 * dh) * 8760
        assert value_waterfill(ci, 0.5, cap) == pytest.approx(num, rel=1e-4)


def test_general_law_equals_daily_waterfill_and_lp_for_week():
    from carbonshift.waterfill import saving_general
    rng = np.random.default_rng(2)
    ci = 300 + 80 * np.sin(np.arange(24 * 40) * 2 * np.pi / 24) + rng.normal(0, 15, 24 * 40)
    for cap in (1.15, 1.6, np.inf):
        assert saving_general(ci, 0.5, cap, 24) == pytest.approx(saving_waterfill(ci, 0.5, cap), rel=1e-9)
    week = np.tile(300 + 80 * np.sin(np.arange(168) * 2 * np.pi / 24) + 60 * np.sin(np.arange(168) * 2 * np.pi / 168), 30)
    for cap in (1.2, 2.0):  # periodic weekly signal: block law equals the LP up to the horizon edge (~1/weeks)
        lp = capped_shift(week, 0.5, cap, 168)
        assert saving_general(week, 0.5, cap, 168) == pytest.approx(lp.baseline_g - lp.optimised_g, rel=0.05)


@pytest.mark.parametrize("phi,cap", [(0.5, 1.1), (0.5, 1.4), (0.3, 1.2)])
def test_capped_shift_is_a_lossless_battery(phi, cap):
    """Flexible load (slack s, share phi, headroom h <= phi) == battery(E = s*phi, P_out = phi, P_in = h, eta = 1)."""
    from carbonshift.facility import battery_shift
    rng = np.random.default_rng(11)
    n = 24 * 120
    ci = 350 + 90 * np.sin(np.arange(n) * 2 * np.pi / 24) + 40 * np.sin(np.arange(n) * 2 * np.pi / 127) + rng.normal(0, 20, n)
    lp = capped_shift(ci, phi, cap, 24)
    b = battery_shift(ci, energy_h=24 * phi, power=phi, round_trip_eff=1.0, cap=cap)
    # identical up to the boundary condition (empty backlog at start vs cyclic): O(1/days)
    assert b.baseline_g - b.optimised_g == pytest.approx(lp.baseline_g - lp.optimised_g, rel=0.01)
