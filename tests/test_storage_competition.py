import numpy as np
import pytest

from carbonshift import storage_competition as sc
from carbonshift.curtailment import coordinated_profile, fleet_emissions

H = 24
HOUR = np.arange(H)
BIG = 1e6


def _days(n=6, mw=60.0, seed=0):
    rng = np.random.default_rng(seed)
    S = np.where((HOUR >= 10) & (HOUR < 15), mw, 0.0)[None, :] * rng.uniform(0.3, 1.5, (n, 1))
    G = np.where(S > 0, 0.0, BIG)
    return S, G


def test_no_storage_matches_f24_coordinated_lp():
    S, G = _days()
    P, phi, cap = 40.0, 0.5, 1.2
    ours = sc.fleet_year(S, G, P, phi, cap, 0.0, 0.0, 0.85)
    prof = coordinated_profile(S.ravel() / P, np.ones(S.size), phi, cap, 24)
    f24 = fleet_emissions(P * prof, S.ravel(), np.ones(S.size)) * 1000.0  # MWh of gas above the surplus
    # ours counts gas *change*: overflow minus nothing displaced (no battery)
    assert ours == pytest.approx(f24, rel=1e-6)


def test_no_arbitrage_marginal_rates():
    """Interior battery: marginal rate where it charges = eta x marginal rate where it discharges."""
    S = np.where((HOUR >= 10) & (HOUR < 15), 50.0, 0.0)
    G = np.where(S > 0, 0.0, BIG)
    eta = 0.8
    r = sc._day_lp(S, G, base=0.0, flex_total=0.0, ycap=0.0, Pb=1000.0, Eb=1e4, eta=eta)
    assert r.charge[10:15].sum() == pytest.approx(250.0)
    assert r.marginal[12] == pytest.approx(eta, abs=1e-9)
    assert r.marginal[19] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("eta", [0.7, 0.85, 1.0])
def test_saving_collapses_to_round_trip_loss(eta):
    S, G = _days()
    kw = dict(P=20.0, phi=0.5, cap=1.5)
    v0 = sc.shifting_saving(S, G, Pb=0.0, Eb=0.0, eta=eta, **kw)
    vinf = sc.shifting_saving(S, G, Pb=1e4, Eb=1e5, eta=eta, **kw)
    assert v0 > 0
    assert vinf == pytest.approx((1 - eta) * v0, rel=1e-6, abs=1e-9)


def test_saving_non_increasing_in_storage():
    S, G = _days(n=8, seed=2)
    vals = [sc.shifting_saving(S, G, 30.0, 0.5, 1.4, pb, 4 * pb, 0.85) for pb in (0, 10, 25, 50, 100)]
    assert all(b <= a + 1e-9 for a, b in zip(vals, vals[1:]))


@pytest.mark.parametrize("pb", [0.0, 10.0, 40.0, 1e4])
def test_greedy_equals_lp_when_timing_is_free(pb):
    S, G = _days(seed=4)
    a = sc.shifting_saving(S, G, 30.0, 0.5, 1.4, pb, 4 * pb, 0.85)
    b = sc.greedy_saving(S, G, 30.0, 0.5, 1.4, pb, 4 * pb, 0.85)
    assert a == pytest.approx(b, rel=1e-6, abs=1e-9)
