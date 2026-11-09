import numpy as np
import pandas as pd
import pytest

from carbonshift.eia930 import marginal_factors


def test_mef_recovers_known_marginal_unit():
    """If gas (490 g/kWh) follows load at night and coal (820) by day, the regression must recover them."""
    t = pd.date_range("2022-01-01", periods=24 * 60, freq="h")
    rng = np.random.default_rng(0)
    base = 1000 + 200 * np.sin(np.arange(len(t)) * 2 * np.pi / 24) + rng.normal(0, 30, len(t))
    day = (t.hour >= 8) & (t.hour < 20)
    mef_true = np.where(day, 820.0, 490.0)
    emis = np.cumsum(np.r_[0, np.diff(base) * mef_true[1:]]) + 500_000  # emissions change at the marginal rate
    m = marginal_factors(pd.DataFrame({"utc": t, "gen_mwh": base, "emis_kg": emis}))
    assert m.loc[(0, 12)] == pytest.approx(820, rel=1e-6)
    assert m.loc[(0, 2)] == pytest.approx(490, rel=1e-6)
