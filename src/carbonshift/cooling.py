"""The cooling penalty of carbon-aware scheduling (F26).

A data centre's grid draw is IT power times PUE, and PUE rises with outdoor
temperature because chillers lose efficiency as the heat-rejection temperature
climbs. So the carbon cost of one IT kWh in hour t is c(t) * PUE(t), not c(t).
On solar-heavy grids the cleanest hours are also the hottest, so a scheduler
that looks only at c(t) moves work into hours where cooling costs more.

Cooling model (hourly, steady state, no thermal lag), two archetypes:

* ``hyperscale``: warm chilled water (18 C), cooling tower on wet-bulb
  temperature, water-side economiser (full free cooling at wet bulb <= 10 C,
  mechanical above 14 C, linear blend between), 4% fixed overhead (power
  conversion, lighting) and 3% fans and pumps.
* ``legacy``: 7 C chilled water, air-cooled condenser on dry-bulb temperature,
  no economiser, 12% fixed overhead and 10% air handlers.

Mechanical COP = eta * T_chw / (T_cond - T_chw) (fraction of Carnot), with
T_cond = outdoor temperature + approach, capped at ``cop_max``. These are
standard first-principles chiller models; the parameters are assumptions
chosen to give annual PUEs of about 1.1 (hyperscale) and 1.5 (legacy), and the
results are reported for both.

Accounting identity (extends F25's saving = -cov(c, d)): with the IT draw d
scaled to mean 1, the true saving against running on arrival is
    -cov(c * PUE, d),
while an operator who multiplies a carbon-only saving by the annual PUE claims
    -mean(PUE) * cov(c, d).
The gap is -cov(c * (PUE - mean PUE), d): the cooling penalty.

Weather: Open-Meteo historical archive (ERA5 reanalysis, CC BY 4.0), hourly
2-m dry-bulb and wet-bulb temperature in UTC at one data-centre hub per zone.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

WEATHER = Path(__file__).resolve().parents[2] / "data" / "weather" / "hub_weather_{year}.csv.gz"

# One representative data-centre hub per cloud-hosting zone (lat, lon).
HUBS: dict[str, tuple[str, float, float]] = {
    "SE-SE1": ("Lulea", 65.58, 22.15),
    "CA-QC": ("Montreal", 45.50, -73.57),
    "NO-NO1": ("Oslo", 59.91, 10.75),
    "CA-ON": ("Toronto", 43.65, -79.38),
    "FR": ("Paris", 48.86, 2.35),
    "NZ": ("Auckland", -36.85, 174.76),
    "US-NW-BPAT": ("Quincy WA", 47.23, -119.85),
    "CH": ("Zurich", 47.38, 8.54),
    "BE": ("St Ghislain", 50.45, 3.82),
    "BR-CS": ("Sao Paulo", -23.55, -46.63),
    "ES": ("Madrid", 40.42, -3.70),
    "FI": ("Hamina", 60.57, 27.20),
    "DK-DK1": ("Fredericia", 55.57, 9.75),
    "AT": ("Vienna", 48.21, 16.37),
    "GB": ("London", 51.51, -0.13),
    "CL-SEN": ("Santiago", -33.45, -70.67),
    "US-CAL-CISO": ("Santa Clara", 37.35, -121.95),
    "US-NW-PACW": ("Prineville", 44.30, -120.83),
    "IT-NO": ("Milan", 45.46, 9.19),
    "IE": ("Dublin", 53.35, -6.26),
    "US-TEX-ERCO": ("Dallas", 32.78, -96.80),
    "US-MIDA-PJM": ("Ashburn", 39.04, -77.49),
    "HK": ("Hong Kong", 22.32, 114.17),
    "US-CAR-SCEG": ("Berkeley County SC", 33.06, -80.04),
    "GR": ("Athens", 37.98, 23.73),
    "NL": ("Eemshaven", 53.44, 6.83),
    "DE": ("Frankfurt", 50.11, 8.68),
    "US-SW-AZPS": ("Phoenix", 33.45, -112.07),
    "JP-KN": ("Osaka", 34.69, 135.50),
    "US-SE-SOCO": ("Atlanta", 33.75, -84.39),
    "KR": ("Seoul", 37.57, 126.98),
    "SG": ("Singapore", 1.35, 103.82),
    "US-NW-NEVP": ("Las Vegas", 36.17, -115.14),
    "TW": ("Changhua", 24.08, 120.54),
    "JP-TK": ("Tokyo", 35.68, 139.69),
    "US-MIDW-MISO": ("Chicago", 41.88, -87.63),
    "AU-VIC": ("Melbourne", -37.81, 144.96),
    "US-NW-PACE": ("Salt Lake City", 40.76, -111.89),
    "AU-NSW": ("Sydney", -33.87, 151.21),
    "US-MIDW-AECI": ("Council Bluffs", 41.26, -95.86),
    "ZA": ("Johannesburg", -26.20, 28.05),
    "IN-MH": ("Mumbai", 19.08, 72.88),
    "IN-UP": ("Noida", 28.54, 77.39),
    "US-NW-WACM": ("Cheyenne", 41.14, -104.82),
    "PL": ("Warsaw", 52.23, 21.01),
    "ID": ("Jakarta", -6.21, 106.85),
    "IL": ("Tel Aviv", 32.09, 34.78),
}


@dataclass(frozen=True)
class Plant:
    t_chw: float          # chilled-water supply temperature, C
    approach: float       # condenser temperature above the driving outdoor temperature, K
    eta: float            # fraction of Carnot COP achieved
    cop_max: float        # cap on mechanical COP
    fixed: float          # overhead independent of weather (fraction of IT)
    fans: float           # fans and pumps, always on (fraction of IT)
    wet_bulb: bool        # True: heat rejected on wet bulb (cooling tower); False: dry bulb
    econ_full: float | None = None   # full free cooling at or below this temperature (C)
    econ_none: float | None = None   # no free cooling at or above this temperature (C)


PLANTS = {
    "hyperscale": Plant(t_chw=18.0, approach=10.0, eta=0.5, cop_max=25.0, fixed=0.04, fans=0.03,
                        wet_bulb=True, econ_full=10.0, econ_none=14.0),
    "legacy": Plant(t_chw=7.0, approach=15.0, eta=0.45, cop_max=12.0, fixed=0.12, fans=0.10, wet_bulb=False),
}


def chiller_cop(t_out: np.ndarray, plant: Plant) -> np.ndarray:
    """Mechanical-cooling COP at outdoor temperature ``t_out`` (C)."""
    t_chw = plant.t_chw + 273.15
    lift = np.maximum(np.asarray(t_out, float) + plant.approach + 273.15 - t_chw, 1e-6)
    return np.minimum(plant.eta * t_chw / lift, plant.cop_max)


def pue(t_out: np.ndarray, plant: Plant) -> np.ndarray:
    """Hourly PUE from the outdoor temperature that drives heat rejection."""
    t = np.asarray(t_out, float)
    mech = 1.0 / chiller_cop(t, plant)
    if plant.econ_full is not None:
        share = np.clip((t - plant.econ_full) / (plant.econ_none - plant.econ_full), 0.0, 1.0)
        mech = share * mech
    return 1.0 + plant.fixed + plant.fans + mech


def plant_pue(weather: pd.DataFrame, zone: str, plant_name: str) -> np.ndarray:
    plant = PLANTS[plant_name]
    col = f"{zone}|{'wb' if plant.wet_bulb else 'db'}"
    return pue(weather[col].to_numpy(), plant)


def decompose(ci: np.ndarray, p: np.ndarray, draw: np.ndarray) -> dict:
    """True vs claimed saving of an IT draw ``draw`` (mean 1) against running flat.

    Returns g CO2 per kWh of mean IT load: ``true`` = -cov(c*PUE, d),
    ``claimed`` = -mean(PUE) cov(c, d), ``penalty`` = claimed - true.
    """
    ci, p, d = (np.asarray(x, float) for x in (ci, p, draw))
    if abs(d.mean() - 1.0) > 1e-6:
        raise ValueError("draw must have mean 1")
    true = float((ci * p).mean() - (ci * p * d).mean())
    claimed = float(p.mean() * (ci.mean() - (ci * d).mean()))
    return {"true": true, "claimed": claimed, "penalty": claimed - true}


def _detrend(x: np.ndarray, window: int = 25) -> np.ndarray:
    s = pd.Series(np.asarray(x, float))
    return (s - s.rolling(window, center=True, min_periods=1).mean()).to_numpy()


def penalty_law(ci: np.ndarray, p: np.ndarray) -> float:
    """Closed-form cooling penalty as a fraction of the claimed saving.

    If the schedule d moves against the carbon signal, cov(PUE, d) / cov(c, d)
    is about rho * sd(PUE) / sd(c), where rho and the sds are of within-day
    (25 h detrended) hourly variation. Then
        penalty / claimed = -rho * (mean(c) / sd(c)) * (sd(PUE) / mean(PUE)).
    Returns nan for a carbon signal with no within-day variation.
    """
    rc, rp = _detrend(ci), _detrend(p)
    if rc.std() < 1e-9 or rp.std() < 1e-12:
        return float("nan") if rc.std() < 1e-9 else 0.0
    rho = float(np.corrcoef(rc, rp)[0, 1])
    return -rho * (float(np.mean(ci)) / rc.std()) * (rp.std() / float(np.mean(p)))


def fetch_weather(year: int = 2022, batch: int = 12) -> pd.DataFrame:
    """Download hourly UTC dry/wet-bulb temperature for every hub (Open-Meteo archive)."""
    zones = list(HUBS)
    cols = {}
    for i in range(0, len(zones), batch):
        part = zones[i:i + batch]
        lat = ",".join(f"{HUBS[z][1]:.2f}" for z in part)
        lon = ",".join(f"{HUBS[z][2]:.2f}" for z in part)
        url = ("https://archive-api.open-meteo.com/v1/archive?"
               f"latitude={lat}&longitude={lon}&start_date={year}-01-01&end_date={year}-12-31"
               "&hourly=temperature_2m,wet_bulb_temperature_2m&timezone=GMT")
        with urllib.request.urlopen(url, timeout=120) as r:
            out = json.load(r)
        out = out if isinstance(out, list) else [out]
        for z, o in zip(part, out):
            cols[f"{z}|db"] = o["hourly"]["temperature_2m"]
            cols[f"{z}|wb"] = o["hourly"]["wet_bulb_temperature_2m"]
            cols.setdefault("datetime", o["hourly"]["time"])
        time.sleep(1.0)
    df = pd.DataFrame(cols)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").astype(float).interpolate(limit_direction="both").round(1)
    path = Path(str(WEATHER).format(year=year))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def load_weather(year: int = 2022) -> pd.DataFrame:
    path = Path(str(WEATHER).format(year=year))
    if not path.exists():
        return fetch_weather(year)
    df = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")
    return df
