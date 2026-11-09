"""Google PowerData2019: real power utilisation of 57 data-centre power domains (May 2019).

Source: github.com/google/cluster-data (PowerData2019.md), CC-BY 4.0; files from
the public bucket gs://powerdata_2019. Each file holds 5-minute averages of
``measured_power_util``: power drawn / power-domain capacity, including floor
cooling. Rows with ``bad_measurement_data == True`` are dropped. (The
documentation describes the flag the other way round, but only 24 of 508,896
rows are True, so True must mark the bad rows.) Timestamps are microseconds since
600 s before 2019-05-01 00:00 US/Pacific.

Reference: V. Sakalkar et al., "Data Center Power Oversubscription with a Medium
Voltage Power Plane and Priority-Aware Capping", ASPLOS 2020.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path(__file__).resolve().parents[2] / "data" / "google_power"
START = pd.Timestamp("2019-05-01 00:00")  # US/Pacific wall clock


@lru_cache(maxsize=1)
def load_all() -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in sorted(DIR.glob("*.csv.gz"))]
    d = pd.concat(frames, ignore_index=True)
    d = d[~d.bad_measurement_data.astype(bool)]
    d["ts"] = START + pd.to_timedelta(d.time / 1e6 - 600, unit="s")
    d["domain"] = d.cell + "_" + d.pdu
    return d[["ts", "domain", "measured_power_util"]]


def domain_stats() -> pd.DataFrame:
    """Per power domain: mean utilisation, hourly peak / mean, diurnal amplitude."""
    d = load_all()
    hourly = d.set_index("ts").groupby("domain").measured_power_util.resample("1h").mean().dropna()
    out = []
    for dom, s in hourly.groupby(level=0):
        s = s.droplevel(0)
        hod = s.groupby(s.index.hour).mean()
        out.append({"domain": dom, "mean_util": s.mean(), "peak_to_mean": s.max() / s.mean(),
                    "diurnal_amplitude": (hod.max() - hod.min()) / hod.mean(), "peak_hour_pt": int(hod.idxmax())})
    return pd.DataFrame(out).set_index("domain")


def week_profile(domain: str | None = None) -> np.ndarray:
    """Hour-of-week load shape (Mon 00:00 = index 0), normalised to mean 1.

    ``domain=None`` gives the fleet median of the normalised per-domain profiles.
    """
    d = load_all()
    hourly = d.set_index("ts").groupby("domain").measured_power_util.resample("1h").mean().dropna().reset_index()
    hourly["how"] = hourly.ts.dt.dayofweek * 24 + hourly.ts.dt.hour
    prof = hourly.groupby(["domain", "how"]).measured_power_util.mean().unstack()
    prof = prof.div(prof.mean(axis=1), axis=0)
    p = prof.median() if domain is None else prof.loc[domain]
    p = p.reindex(range(168)).interpolate().to_numpy()
    return p / p.mean()


def tile(profile168: np.ndarray, index: pd.DatetimeIndex, shift_hours: int = 0) -> np.ndarray:
    """Lay an hour-of-week profile on a time index, shifted by ``shift_hours``; renormalised to mean 1."""
    how = ((index.dayofweek * 24 + index.hour + shift_hours) % 168).to_numpy()
    x = profile168[how]
    return x / x.mean()
