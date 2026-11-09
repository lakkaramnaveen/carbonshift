import numpy as np
import pytest

from carbonshift.embodied import BUILDS, LIFETIME, per_kw, payback_years
from carbonshift.facility import capped_shift
from carbonshift.redundancy import (OutagePlan, cap_profile, carbon_aware_windows, failure_windows,
                                    installed_cap, random_windows)


def test_installed_cap():
    assert installed_cap(0.5, "design_only") == pytest.approx(2.0)
    assert installed_cap(0.5, "2N") == pytest.approx(4.0)
    assert installed_cap(0.8, "N+1 (N=4)") == pytest.approx(1.25 * 1.25)


def test_payback():
    assert payback_years(100, 50_000) == pytest.approx(2.0)          # 100 kg at 50 kg/yr
    assert payback_years(100, 0) == float("inf")
    assert payback_years(100, 50_000, decline_per_year=0.6) == float("inf")  # 50+20+8+... -> 83 < 100
    assert payback_years(10, 50_000, decline_per_year=0.1) < payback_years(10, 45_000)


def test_embodied_builds_consistent():
    for b in BUILDS:
        lo, mid, hi = per_kw(b)
        assert 0 <= lo <= mid <= hi
        assert LIFETIME[b] > 0
    assert per_kw("grid_only")[1] < per_kw("backed")[1] < per_kw("gpu_servers")[1] < per_kw("cpu_servers")[1]


def test_windows_do_not_overlap_and_aware_picks_cheapest():
    rng = np.random.default_rng(0)
    s = random_windows(rng, 8760, 3, 24)
    assert len(s) == 3 and all(abs(a - b) >= 24 for i, a in enumerate(s) for b in s[i + 1:])
    shadow = np.ones(1000)
    shadow[500:524] = 0.0
    assert carbon_aware_windows(shadow, 1, 24) == [500]


def test_outages_never_increase_saving_and_redundancy_helps():
    t = np.arange(24 * 60)
    ci = 300 + 120 * np.sin(2 * np.pi * t / 24) + np.random.default_rng(3).normal(0, 10, len(t))
    u = 0.65
    design = capped_shift(ci, 0.5, installed_cap(u, "design_only"), 24).optimised_g
    normal = installed_cap(u, "2N")
    perfect = capped_shift(ci, 0.5, normal, 24)
    caps = cap_profile(len(ci), normal, 1 / u, [(100, 48), (700, 48)])
    degraded = capped_shift(ci, 0.5, caps, 24).optimised_g
    assert perfect.optimised_g <= degraded + 1e-9 <= design + 1e-9


def test_failure_windows_rate():
    rng = np.random.default_rng(1)
    counts = [len(failure_windows(rng, 8760, OutagePlan(failure_rate_per_year=2.0))) for _ in range(400)]
    assert np.mean(counts) == pytest.approx(2.0, rel=0.15)
