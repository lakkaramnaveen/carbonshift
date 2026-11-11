import numpy as np
import pandas as pd
import pytest

from carbonshift import curtailment as cu
from carbonshift.curtailment import absorbable_daily, coordinated_profile, fleet_emissions, herd_profile

DAYS = 20
L = 24 * DAYS
HOUR = np.arange(L) % 24


def _midday_curtailment(mw=300.0):
    return np.where((HOUR >= 10) & (HOUR < 15), mw, 0.0)


def test_no_curtailment_constant_margin_makes_shifting_worthless():
    C, m = np.zeros(L), np.full(L, 490.0)
    avg = 300 + 100 * np.cos(2 * np.pi * HOUR / 24)
    flat = fleet_emissions(np.full(L, 100.0), C, m)
    for prof in (herd_profile(avg, 0.5, 1.2, 24), coordinated_profile(C / 100, m, 0.5, 1.2, 24)):
        assert fleet_emissions(100 * prof, C, m) == pytest.approx(flat, rel=1e-9)


@pytest.mark.parametrize("cap", [1.2, 1.5, np.inf])
def test_coordinated_is_never_worse(cap):
    rng = np.random.default_rng(3)
    C = _midday_curtailment() * rng.uniform(0, 1.5, L)
    m = rng.uniform(350, 550, L)
    P = 120.0
    coord = fleet_emissions(P * coordinated_profile(C / P, m, 0.5, cap, 24), C, m)
    for prof in (np.ones(L), herd_profile(m - 1000 * (C > 0), 0.5, cap, 24)):
        assert coord <= fleet_emissions(P * prof, C, m) + 1e-6


def test_uncapped_herd_abandons_curtailment_and_raises_emissions():
    """Many hours with a little curtailment: a flat fleet uses all of it, an uncapped herd piles into one hour."""
    C = np.where(HOUR < 20, 60.0, 0.0)
    m = np.full(L, 490.0)
    signal = np.where(C > 0, 0.0, 490.0) + 1e-3 * HOUR  # 'free when curtailed', ties broken by hour
    P = 100.0
    flat = fleet_emissions(np.full(L, P), C, m)
    herd = fleet_emissions(P * herd_profile(signal, 0.5, np.inf, 24), C, m)
    capped = fleet_emissions(P * herd_profile(signal, 0.5, 1.2, 24), C, m)
    assert herd > flat * 1.05
    assert capped - flat < (herd - flat) / 4  # a cap limits the overflow (it does not rule it out)


def test_closed_form_absorption_matches_lp_for_daily_blocks():
    C = _midday_curtailment(80.0)
    P = 100.0
    prof = coordinated_profile(C / P, np.full(L, 490.0), 0.5, 1.5, 24)
    lp_share = np.minimum(P * prof, C).sum() / (P * prof).sum()
    assert absorbable_daily(C / P, 0.5, 1.5) == pytest.approx(lp_share, rel=0.02)


def test_caiso_hourly_aggregates_five_minute_data(tmp_path, monkeypatch):
    day = pd.Timestamp("2030-01-01")
    n = 24 * 12
    prod = pd.DataFrame({"Date": [day + pd.Timedelta(minutes=5 * i) for i in range(n)], "Hour": np.repeat(np.arange(1, 25), 12),
                         "Interval": np.tile(np.arange(1, 13), 24), "Load": 1000.0, "Solar": 10.0, "Wind": 5.0,
                         "Thermal": 400.0, "Imports": 200.0})
    cur = pd.DataFrame({"Date": [day] * 3, "Hour": [12, 12, 13], "Interval": [1, 2, 1],
                        "Wind Curtailment": [np.nan, 6.0, np.nan], "Solar Curtailment": [120.0, 60.0, 24.0], "Reason": ["x"] * 3})
    with pd.ExcelWriter(tmp_path / "ProductionAndCurtailmentsData_2030.xlsx") as w:
        prod.to_excel(w, sheet_name="Production", index=False)
        cur.to_excel(w, sheet_name="Curtailments", index=False)
    monkeypatch.setattr(cu, "RAW", tmp_path)
    monkeypatch.setattr(cu, "DERIVED", tmp_path / "hourly_{year}.csv.gz")
    cu.caiso_hourly.cache_clear()
    h = cu.caiso_hourly(2030)
    assert len(h) == 24 and h.load.eq(1000.0).all()
    assert h.set_index("hour").curtail.to_dict()[12] == pytest.approx((120 + 66) / 12)
    assert h.set_index("hour").curtail.to_dict()[13] == pytest.approx(2.0)
    assert h.curtail.sum() == pytest.approx((120 + 66 + 24) / 12)
    cu.caiso_hourly.cache_clear()
