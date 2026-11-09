"""US balancing-authority emissions from EIA-930: average and marginal carbon signals.

Source: EIA Hourly Electric Grid Monitor six-month balance files (public domain),
https://www.eia.gov/electricity/gridmonitor/ . Uses the *adjusted* hourly net
generation by fuel for each balancing authority (BA), timestamps in UTC.

Average intensity: generation-weighted lifecycle factors (IPCC 2014 medians, as
used by Electricity Maps: coal 820, gas 490, oil 650, nuclear 12, hydro 24,
solar 45, wind 11 g/kWh; "other" taken as 230, biomass-like, and "unknown" as
the gas value 490 — both assumptions). This is production-based: imports are
ignored.

Marginal emission factors (MEF) follow Siler-Evans et al. (Environ. Sci.
Technol., 2012): regress the hour-to-hour change in emissions on the change in
generation, separately for each (season, hour-of-day) bin. The slope is the
MEF for that bin, and the marginal signal for hour t is the MEF of its bin.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "eia930" / "raw"
DERIVED = Path(__file__).resolve().parents[2] / "data" / "eia930" / "signals_2022.csv.gz"
FUELS = {"Coal": 820, "Natural Gas": 490, "All Petroleum Products": 650, "Nuclear": 12,
         "Hydropower and Pumped Storage": 24, "Solar": 45, "Wind": 11, "Other Fuel Sources": 230,
         "Unknown Fuel Sources": 490}
SEASON = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}


def _read_raw() -> pd.DataFrame:
    cols = ["Balancing Authority", "UTC Time at End of Hour"] + [f"Net Generation (MW) from {f} (Adjusted)" for f in FUELS]
    frames = [pd.read_csv(f, usecols=cols, thousands=",", low_memory=False) for f in sorted(RAW.glob("EIA930_BALANCE_2022_*.csv"))]
    d = pd.concat(frames, ignore_index=True)
    d["utc"] = pd.to_datetime(d["UTC Time at End of Hour"], format="%m/%d/%Y %I:%M:%S %p") - pd.Timedelta(hours=1)  # start of hour
    d = d.rename(columns={"Balancing Authority": "ba"})
    gen = d[[f"Net Generation (MW) from {f} (Adjusted)" for f in FUELS]].clip(lower=0).fillna(0.0)
    d["gen_mwh"] = gen.sum(axis=1)
    d["emis_kg"] = (gen * np.array(list(FUELS.values()))).sum(axis=1)  # MWh x g/kWh = kg
    return d[["ba", "utc", "gen_mwh", "emis_kg"]]


def marginal_factors(df: pd.DataFrame) -> pd.Series:
    """MEF (g/kWh) per (season, hour) bin from first differences, OLS through the origin."""
    d = df.sort_values("utc")
    dg, de = d.gen_mwh.diff(), d.emis_kg.diff()
    ok = (d.utc.diff() == pd.Timedelta(hours=1)) & dg.abs().gt(1e-6)
    x = pd.DataFrame({"dg": dg[ok], "de": de[ok], "s": d.utc[ok].dt.month.map(SEASON), "h": d.utc[ok].dt.hour})
    return x.groupby(["s", "h"]).apply(lambda b: (b.dg * b.de).sum() / (b.dg ** 2).sum(), include_groups=False)


@lru_cache(maxsize=1)
def signals() -> pd.DataFrame:
    """Hourly UTC average and marginal signals (g/kWh) per BA for 2022; cached to a small derived file."""
    if DERIVED.exists():
        return pd.read_csv(DERIVED, parse_dates=["utc"])
    raw = _read_raw()
    out = []
    for ba, d in raw.groupby("ba"):
        d = d.drop_duplicates("utc").set_index("utc").sort_index()[["gen_mwh", "emis_kg"]]
        d = d[(d.gen_mwh > 0)].reindex(pd.date_range("2022-01-01", "2022-12-31 23:00", freq="h"))
        if d.gen_mwh.notna().mean() < 0.9:
            continue  # too sparse
        d = d.interpolate(limit=6).ffill().bfill()
        avg = d.emis_kg / d.gen_mwh
        mef = marginal_factors(d.rename_axis("utc").reset_index())
        key = list(zip(d.index.month.map(SEASON), d.index.hour))
        out.append(pd.DataFrame({"utc": d.index, "ba": ba, "average": avg.values, "marginal": [mef.get(k, np.nan) for k in key]}))
    df = pd.concat(out, ignore_index=True)
    DERIVED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DERIVED, index=False, compression="gzip")
    return df
