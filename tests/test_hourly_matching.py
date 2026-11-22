import numpy as np
import pandas as pd
import pytest

from carbonshift.facility import capped_shift
from carbonshift.hourly_matching import (
    cfe_score, clear_sky_solar, hm_attribution, hm_shift, ledger, portfolio, toy_grid_inventories,
)


def test_clear_sky_night_noon_and_season():
    utc = pd.date_range("2022-01-01", periods=8760, freq="h")
    g = clear_sky_solar(utc, 36.0, -119.5)
    assert g.min() == 0 and g.max() > 800
    # local solar noon at lon -119.5 is ~20:00 UTC
    jun = g[(utc.month == 6)].reshape(-1, 24).mean(axis=0)
    assert int(jun.argmax()) in (19, 20)
    assert jun[8] == 0  # 08 UTC = midnight local
    assert g[utc.month == 6].sum() > 1.5 * g[utc.month == 12].sum()


def test_portfolio_sizing_and_cfe():
    P = portfolio(np.r_[np.zeros(12), np.ones(12)], 1.0)
    assert P.mean() == pytest.approx(1.0)
    assert cfe_score(np.ones(24), P) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        portfolio(np.zeros(4))


def test_hm_shift_without_portfolio_equals_carbon_shift():
    rng = np.random.default_rng(0)
    c = 300 + 100 * rng.random(24 * 5)
    draw = hm_shift(c, np.zeros_like(c), 0.5, 1.5, 6)
    ref = capped_shift(c, 0.5, 1.5, 6)
    assert (c * draw).mean() == pytest.approx(ref.optimised_g, rel=1e-6)


def test_hm_shift_moves_work_into_surplus_and_respects_constraints():
    n = 48
    c = np.full(n, 400.0)
    shape = np.zeros(n)
    shape[np.r_[10:15, 34:39]] = 1.0
    P = portfolio(shape, 1.0)
    draw = hm_shift(c, P, 0.5, 2.0, 24, tiebreak=c)
    assert draw.mean() == pytest.approx(1.0)
    assert draw.max() <= 2.0 + 1e-9 and draw.min() >= 0.5 - 1e-9
    assert hm_attribution(c, draw, P) < hm_attribution(c, np.ones(n), P) - 1
    assert cfe_score(draw, P) > cfe_score(np.ones(n), P)
    # cumulative service never exceeds arrivals
    assert np.all(np.cumsum(draw - 0.5) <= np.cumsum(np.full(n, 0.5)) + 1e-9)


def test_ledger_identity_and_flat_grid_real_zero():
    n = 48
    c = np.full(n, 400.0)
    m = np.full(n, 450.0)        # flat marginal rate: no shift has a real effect
    shape = np.zeros(n)
    shape[np.r_[10:15, 34:39]] = 1.0
    P = portfolio(shape)
    draw = hm_shift(c, P, 0.5, 2.0, 24, tiebreak=c)
    L = ledger(c, m, P, draw)
    assert L["real"] == pytest.approx(0.0, abs=1e-9)
    assert L["reported"] > 50
    assert L["transfer"] == pytest.approx(L["reported"] - L["real"])


def test_toy_grid_transfer_matches_first_order_ledger():
    """Exact inventories on a toy grid: others absorb (reported - real), as the ledger says."""
    rng = np.random.default_rng(1)
    n = 72
    hours = np.arange(n) % 24
    D0 = 1e4 * (1 + 0.1 * np.sin(2 * np.pi * hours / 24))
    R = 3e3 * np.clip(np.sin(np.pi * (hours - 6) / 12), 0, None) + 500 * rng.random(n)
    eps = 1.0                                    # consumer of mean 1 MW on a 10 GW grid
    P = eps * portfolio(np.clip(np.sin(np.pi * (hours - 6) / 12), 0, None))
    flat = eps * np.ones(n)
    draw = eps * hm_shift(np.ones(n), P / eps, 0.5, 2.0, 24, tiebreak=-P)
    a, b = toy_grid_inventories(D0, R, P, flat), toy_grid_inventories(D0, R, P, draw)
    # exact budget balance of hourly market-based accounting
    assert a["self"] + a["others"] == pytest.approx(a["E"], rel=1e-12)
    dE, dself, dothers = b["E"] - a["E"], b["self"] - a["self"], b["others"] - a["others"]
    assert dself + dothers == pytest.approx(dE, rel=1e-9, abs=1e-9)
    # first-order ledger with c = residual-mix rate and m = d emissions / d fossil
    f = np.maximum(D0 + flat - R - P, 0)
    m = 400.0 + 0.1 * f
    c = (400.0 * f + 0.05 * f ** 2) / D0
    led = ledger(c, m, P / eps, draw / eps)
    assert -dself / n == pytest.approx(led["reported"] * eps, rel=0.02)
    assert -dE / n == pytest.approx(led["real"] * eps, rel=0.02, abs=1e-3)
    assert dothers / n == pytest.approx(led["transfer"] * eps, rel=0.02)
    assert dothers > 0
