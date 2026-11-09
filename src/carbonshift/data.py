"""Loading the Electricity Maps carbon-intensity traces and region metadata.

The traces are the ones released with Sukprasert et al., EuroSys 2024
(github.com/umassos/decarbonization-potential): hourly average carbon intensity
in gCO2eq/kWh for 123 grid zones, 2020-2022. Electricity Maps data is licensed
under the ODbL (see data/README.md).

Preprocessing matches the original artifact exactly: gaps are back-filled then
forward-filled over the whole 3-year trace, and only then is a year selected.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CARBON_FILE = DATA_DIR / "combined_carbon.csv"
GROUPS_FILE = DATA_DIR / "region_groups.csv"

GEO_GROUPINGS = ("Global", "Americas", "Europe", "Asia", "Oceania")
DEFAULT_YEAR = 2022


@lru_cache(maxsize=None)
def _load_all() -> pd.DataFrame:
    df = pd.read_csv(CARBON_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.bfill().ffill()
    return df.set_index("datetime")


def load_carbon(year: int | None = DEFAULT_YEAR) -> pd.DataFrame:
    """Hourly carbon intensity, one column per zone. ``year=None`` returns all years."""
    df = _load_all()
    if year is not None:
        df = df[df.index.year == year]
    return df.copy()


@lru_cache(maxsize=None)
def region_groups() -> pd.DataFrame:
    return pd.read_csv(GROUPS_FILE, encoding="utf-8-sig").set_index("zone_code")


def grouping_members(grouping: str) -> list[str]:
    """Zones in a geographic grouping as defined by the original artifact."""
    groups = region_groups()
    if grouping == "Global":
        return groups.index.tolist()
    return groups.index[groups[grouping] == 1].tolist()


def as_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """(hours x zones) float array plus zone names."""
    return df.to_numpy(dtype=float), list(df.columns)
