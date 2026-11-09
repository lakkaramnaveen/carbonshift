"""Check the compact cumulative LP against a brute-force per-arrival transportation LP."""

import numpy as np
import pytest
from scipy.optimize import linprog

from carbonshift.facility import battery_shift, capped_shift
from carbonshift.temporal import shift_one_zone


def brute_force(ci, phi, cap, slack):
    """x[t, s] = flexible work from arrival t served at hour s in [t, min(t+slack, L-1)]."""
    L = len(ci)
    pairs = [(t, s) for t in range(L) for s in range(t, min(t + slack, L - 1) + 1)]
    c = np.array([ci[s] for _, s in pairs])
    A_eq = np.zeros((L, len(pairs)))
    A_ub = np.zeros((L, len(pairs)))
    for k, (t, s) in enumerate(pairs):
        A_eq[t, k] = 1
        A_ub[s, k] = 1
    res = linprog(c, A_ub=A_ub, b_ub=np.full(L, cap - (1 - phi)), A_eq=A_eq, b_eq=np.full(L, phi),
                  bounds=(0, None), method="highs")
    assert res.status == 0
    return (res.fun + (1 - phi) * ci.sum()) / L


@pytest.mark.parametrize("phi,cap,slack", [(0.5, 1.0, 4), (0.5, 1.2, 4), (0.3, 1.5, 6), (0.8, 1.1, 10), (0.5, 3.0, 3)])
def test_compact_lp_equals_brute_force(phi, cap, slack):
    ci = np.random.default_rng(int(cap * 10 + slack)).uniform(50, 600, 60)
    assert capped_shift(ci, phi, cap, slack).optimised_g == pytest.approx(brute_force(ci, phi, cap, slack), rel=1e-7)


def test_no_headroom_means_no_saving():
    ci = np.random.default_rng(1).uniform(50, 600, 500)
    r = capped_shift(ci, 0.5, 1.0, 24)
    assert r.optimised_g == pytest.approx(r.baseline_g)


def test_large_cap_equals_uncapped_and_monotone():
    ci = np.random.default_rng(2).uniform(50, 600, 800)
    inf = capped_shift(ci, 0.5, np.inf, 24).optimised_g
    big = capped_shift(ci, 0.5, 1e4, 24).optimised_g
    assert big == pytest.approx(inf, rel=1e-9)
    vals = [capped_shift(ci, 0.5, c, 24).optimised_g for c in (1.0, 1.1, 1.3, 1.6, 2.0)]
    assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:]))


def test_uncapped_nests_paper_one_hour_jobs():
    """Saving per flexible kWh with no cap = the paper's 1-hour-job deferral saving."""
    ci = np.random.default_rng(3).uniform(50, 600, 8760)
    paper = shift_one_zone(ci, 1, 24)
    paper_saving = (paper.baseline - paper.defer).mean()
    ours = capped_shift(ci, 1.0, np.inf, 24).saving_per_flex_kwh
    assert ours == pytest.approx(paper_saving, rel=2e-3)  # differs only in the final arrival hour


def test_battery_bounds():
    t = np.arange(24 * 30)
    ci = 300 + 100 * np.sin(2 * np.pi * t / 24)
    small = battery_shift(ci, energy_h=1 / 6)
    large = battery_shift(ci, energy_h=4)
    lossy = battery_shift(ci, energy_h=4, round_trip_eff=0.5)
    assert 0 < small.reduction_pct < large.reduction_pct
    assert lossy.reduction_pct < large.reduction_pct
    assert battery_shift(ci, energy_h=1 / 6, reserve_frac=1.0).reduction_pct == 0
