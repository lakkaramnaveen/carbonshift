import numpy as np
import pytest

from carbonshift.spatial_capacity import spatial_shift


def _setup(T=48, R=4, seed=0):
    rng = np.random.default_rng(seed)
    ci = rng.uniform(50, 700, (T, R))
    lat = rng.uniform(10, 200, (R, R))
    lat = (lat + lat.T) / 2
    np.fill_diagonal(lat, 0)
    return ci, [f"r{i}" for i in range(R)], lat


def test_uncapacitated_goes_to_cheapest_reachable():
    ci, regs, lat = _setup()
    r = spatial_shift(ci, regs, lat, 0.5, 100.0, limit_ms=120)
    R = len(regs)
    reach = (lat <= 120) | np.eye(R, dtype=bool)
    expect = sum(0.5 * ci[t, reach[o]].min() + 0.5 * ci[t, o] for t in range(len(ci)) for o in range(R)) / ci.size
    assert r.optimised_g == pytest.approx(expect)
    assert np.allclose(r.shadow, 0, atol=1e-9)  # capacity never binds


def test_no_capacity_means_no_migration_and_shadow_matches_fd():
    ci, regs, lat = _setup(seed=1)
    tight = spatial_shift(ci, regs, lat, 0.5, 1.0, limit_ms=300)
    assert tight.optimised_g == pytest.approx(tight.baseline_g)
    eps = 1e-4
    a = spatial_shift(ci, regs, lat, 0.5, 1.1, limit_ms=300)
    caps = np.full(len(regs), 1.1)
    caps[2] += eps
    b = spatial_shift(ci, regs, lat, 0.5, caps, limit_ms=300)
    fd = (a.optimised_g - b.optimised_g) * ci.size / eps
    assert a.annual_shadow[2] == pytest.approx(fd, rel=1e-3, abs=1e-6)
    assert np.all(a.shadow >= -1e-9)


def test_latency_limit_monotone():
    ci, regs, lat = _setup(seed=2)
    vals = [spatial_shift(ci, regs, lat, 0.5, 1.5, limit_ms=L).optimised_g for L in (0, 50, 100, 300)]
    assert all(x >= y - 1e-9 for x, y in zip(vals, vals[1:]))
