import numpy as np
import pytest

from carbonshift.facility import capped_shift
from carbonshift.hedge import frontier, saving


def _signals(n=24 * 30, seed=0):
    t = np.arange(n)
    rng = np.random.default_rng(seed)
    avg = 400 + 100 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 10, n)            # cleanest mid-day
    mar = 600 + 80 * np.sin(2 * np.pi * (t + 9) / 24) + rng.normal(0, 10, n)       # a different, shifted pattern
    return avg, mar


def test_saving_matches_capped_shift():
    avg, _ = _signals()
    r = capped_shift(avg, 0.5, 1.4, 24)
    assert saving(avg, r.served, 0.5) == pytest.approx(r.baseline_g - r.optimised_g, rel=1e-9)


def test_endpoints_are_each_signals_optimum():
    avg, mar = _signals()
    f = frontier(avg, mar, 0.5, 1.4, 24, [0.0, 1.0]).set_index("w_avg")
    assert f.loc[1.0, "avg_share"] == pytest.approx(1.0, abs=1e-6)
    assert f.loc[0.0, "marg_share"] == pytest.approx(1.0, abs=1e-6)


def test_frontier_is_monotone_and_bounded():
    avg, mar = _signals(seed=3)
    f = frontier(avg, mar, 0.5, 1.4, 24, np.linspace(0, 1, 11))
    assert (np.diff(f.avg_saving_g) >= -1e-6).all()   # more weight on avg never lowers avg saving
    assert (np.diff(f.marg_saving_g) <= 1e-6).all()
    assert (f[["avg_share", "marg_share"]] <= 1 + 1e-6).all().all()


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        frontier(np.ones(48), np.ones(24), 0.5, 1.4, 24, [0.5])
