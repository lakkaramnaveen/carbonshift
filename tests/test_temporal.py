"""The vectorised temporal code must match a line-by-line port of the original artifact loop."""

import numpy as np
import pandas as pd
import pytest

from carbonshift.temporal import rolling_sum, shift_one_zone


def reference_loop(ci: np.ndarray, job: int, slack: int, year_hours: int):
    """Port of sim_temporal/experiment_vary_job_slack.py::task (absolute slack)."""
    s = pd.Series(ci)
    L = len(s)
    inf = slack == year_hours
    ending = L if inf else L - job
    base, defer, inter = [], [], []
    for i in range(ending):
        w = s.iloc[i:slack] if inf else s.iloc[i : i + slack + job]
        if len(w) < job:
            break
        sorted_w = w.sort_values().values
        rolling = w.rolling(job).sum().dropna().reset_index(drop=True)
        inter.append(sorted_w[:job].sum())
        defer.append(rolling.min())
        base.append(rolling.loc[0])
    return np.array(base), np.array(defer), np.array(inter)


@pytest.mark.parametrize("job,slack", [(1, 5), (3, 5), (4, 7), (6, 24), (1, 200), (5, 200)])
def test_matches_reference(job, slack):
    rng = np.random.default_rng(job * 100 + slack)
    L = 200  # a short "year" so the one-year-slack branch (slack == L) is exercised
    ci = rng.uniform(10, 800, L)
    base, defer, inter = reference_loop(ci, job, slack, year_hours=L)
    r = shift_one_zone(ci, job, slack)
    np.testing.assert_allclose(r.baseline, base)
    np.testing.assert_allclose(r.defer, defer)
    np.testing.assert_allclose(r.interrupt, inter)


def test_ordering_invariant():
    ci = np.random.default_rng(0).uniform(0, 500, 1000)
    r = shift_one_zone(ci, 12, 48)
    assert np.all(r.interrupt <= r.defer + 1e-9)
    assert np.all(r.defer <= r.baseline + 1e-9)


def test_rolling_sum():
    np.testing.assert_allclose(rolling_sum(np.arange(5.0), 2), [1, 3, 5, 7])
