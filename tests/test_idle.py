import numpy as np
import pytest

from carbonshift import idle
from carbonshift.facility import capped_shift
from carbonshift.waterfill import saving_sinc

T = np.arange(24 * 60)
M, A = 400.0, 120.0
SINE = M - A * np.cos(2 * np.pi * T / 24)


def test_first_unit_value_is_amplitude_for_a_sine():
    assert idle.first_unit_value(SINE) == pytest.approx(A, rel=1e-9)
    # the sinc law agrees: dS/dh at h -> 0 equals A for any phi
    for phi in (0.2, 0.5, 0.9):
        eps = 1e-4
        assert saving_sinc(A, phi, 1 + eps) / eps == pytest.approx(A, rel=2e-3)


def test_breakeven_closed_form():
    assert idle.breakeven_idle(SINE) == pytest.approx(A / (A + M))


def test_first_unit_value_matches_lp_shadow_price_on_a_sine():
    """Independent derivation: the LP's dual at a tiny cap equals A on a pure sine."""
    r = capped_shift(SINE, 0.5, 1.001, 24)
    assert r.annual_marginal_saving / len(SINE) == pytest.approx(A, rel=0.02)


def test_lp_first_unit_between_daily_law_and_bound():
    """With noise the LP plans across days: daily D <= LP value <= max - mean."""
    rng = np.random.default_rng(1)
    ci = SINE + rng.normal(0, 30, SINE.size)
    lp = capped_shift(ci, 0.5, 1.001, 24).annual_marginal_saving / len(ci)
    assert idle.first_unit_value(ci) * 0.99 <= lp <= ci.max() - ci.mean()
    p, _ = idle.breakeven_idle_lp(ci, 0.5, hs=(0.01, 0.1))
    assert p <= idle.breakeven_bound(ci)


def test_net_saving_sign_flips_at_breakeven():
    p_star = idle.breakeven_idle(SINE)
    h = 0.01
    assert idle.net_build_saving(SINE, 0.5, h, p_star * 0.8, exact=True) > 0
    assert idle.net_build_saving(SINE, 0.5, h, p_star * 1.2, exact=True) < 0


def test_energy_proportional_limit_and_optimum():
    # p0 = 0: net saving is the plain capped saving
    r = capped_shift(SINE, 0.5, 1.3, 24)
    assert idle.net_build_saving(SINE, 0.5, 0.3, 0.0, exact=True) == pytest.approx(r.baseline_g - r.optimised_g)
    # above break-even nothing is worth building
    h, v = idle.optimal_headroom(SINE, 0.5, 0.5, grid=[0, 0.01, 0.1, 0.5], exact=True)
    assert h == 0.0 and v == 0.0
    h, v = idle.optimal_headroom(SINE, 0.5, 0.05, grid=[0, 0.01, 0.1, 0.5], exact=True)
    assert h > 0 and v > 0


def test_existing_spare_accounting():
    out = idle.existing_spare(SINE, 0.5, 0.4, 0.25)
    assert out["sleep_shift"] == pytest.approx(out["sleep_only"] + out["S"])
    assert out["shift_only"] == pytest.approx(0.75 * out["S"])
    assert out["sleep_only"] == pytest.approx(0.25 * 0.4 * SINE.mean())
