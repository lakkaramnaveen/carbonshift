"""Embodied (manufacturing, EN 15804 A1-A3) carbon of data-centre electrical plant, per kW.

I found no peer-reviewed per-kW factors for this equipment, so they are derived
here from manufacturer Environmental Product Declarations (EPDs). Each factor
is in kg CO2eq per kW of *facility* power capacity and comes as (low, central,
high). Per-kVA values are converted at the stated power factor.
Sources (accessed 2026-11-09):

transformer
  Trafo Elettro TES-R2-2 2500 kVA cast-resin, EPDItaly1073 rev04 (EN 15804+A2):
  A1-A3 = 4.07e4 kg CO2eq per transformer -> 16.3 kg/kVA. (Use stage, mostly losses
  over 35 years: 2.42e6 kg, i.e. 59x manufacturing.)
  ABB 16/20 MVA ONAN distribution transformer EPD S-P-00056: manufacturing 7.65 kg/kVA.
ups (power electronics, excl. batteries)
  ABB MegaFlex DPA 1500 kW PEP (OneClickLCA listing): 25.4 t / 1500 kW = 17 kg/kW.
  ABB DPA 250 S4 (INIES): 9.27 t / 250 kW = 37 kg/kW.
  Schneider Galaxy VX PEP ENVPEP2311001_V2: 92.4 t A1-A3 for a 3600 kg unit;
  1250-1500 kW frames weigh ~3450 kg (Schneider spec) -> 62-74 kg/kW.
ups_battery
  Li-ion LFP 54-69 kg CO2eq/kWh (Peiseler 2024, as cited by REEL LCI; secondary
  source) x 10 min ride-through = 1/6 kWh per kW -> 9-11.5 kg/kW.
generator
  Cummins QSK95 genset EPD-IES-0025774:003: upstream + core = 5.32e4 kg per MVA
  (1 MVA standby, 25-year RSL) -> 53 kg/kVA -> 66 kg/kW at pf 0.8.
  mtu 10V1600 DS500 EPD-IES-0017705:002: A1-A3 2.8e4 kg for 459 kVA -> 61 kg/kVA
  -> 76 kg/kW at pf 0.8.
switchgear
  No primary per-rating EPD found; treated as a sensitivity range 0-20 kg/kW.
cpu_servers (compute headroom: extra servers to absorb shifted work)
  Dell PowerEdge R650 PCF (2022): 9,860 kg CO2eq lifetime, ~11% manufacturing -> ~1,085 kg;
  Dell PowerEdge R760 LCA: 4,177 kg (US use), manufacturing 10-15% -> 420-630 kg.
  Assumed 0.35 kW average draw per server (assumption) -> 1,200-3,100 kg/kW.
gpu_servers
  NVIDIA HGX H100 baseboard PCF: 1,312 kg cradle-to-gate for 8 x H100 SXM (8 x 700 W TDP)
  -> 234 kg/kW for the baseboard alone; the full server (CPUs, DRAM, storage, chassis)
  is assumed to double it (assumption) -> 230-700 kg/kW.

Two build options matter:
  * "backed" capacity (transformer + switchgear + UPS + battery + generator) is
    what a tier-rated facility adds for critical load;
  * "grid-only" capacity (transformer + switchgear) is enough for sheddable,
    deferrable work, because on a grid outage that work is postponed, not lost.
"""

from __future__ import annotations

PF_TRANSFORMER = 0.9
PF_GENERATOR = 0.8

# (low, central, high) kg CO2eq per kW
FACTORS: dict[str, tuple[float, float, float]] = {
    "transformer": (7.65 / PF_TRANSFORMER, 16.3 / PF_TRANSFORMER, 16.3 / PF_TRANSFORMER),
    "switchgear": (0.0, 10.0, 20.0),
    "ups": (17.0, 40.0, 74.0),
    "ups_battery": (9.0, 10.3, 11.5),
    "generator": (53.0 / PF_GENERATOR, 57.0 / PF_GENERATOR, 61.0 / PF_GENERATOR),
}

FACTORS.update({
    "cpu_servers": (1200.0, 2000.0, 3100.0),
    "gpu_servers": (234.0, 470.0, 700.0),
})

BUILDS = {
    "grid_only": ("transformer", "switchgear"),
    "backed": ("transformer", "switchgear", "ups", "ups_battery", "generator"),
    "cpu_servers": ("cpu_servers",),
    "gpu_servers": ("gpu_servers",),
}

# Service life over which each build must repay its embodied carbon (years).
LIFETIME = {"grid_only": 25, "backed": 25, "cpu_servers": 5, "gpu_servers": 5}


def per_kw(build: str) -> tuple[float, float, float]:
    """(low, central, high) embodied kg CO2eq per kW of added capacity for a build option."""
    parts = [FACTORS[p] for p in BUILDS[build]]
    return tuple(sum(p[i] for p in parts) for i in range(3))  # type: ignore[return-value]


def payback_years(embodied_kg_per_kw: float, annual_saving_g_per_kw: float, decline_per_year: float = 0.0,
                  max_years: int = 100) -> float:
    """Years until cumulative operational savings repay the embodied carbon.

    ``decline_per_year`` shrinks the saving geometrically (grid decarbonisation
    reduces both the level and, often, the value of shifting). Returns inf if
    it never pays back within ``max_years``.
    """
    if annual_saving_g_per_kw <= 0:
        return float("inf")
    need, acc = embodied_kg_per_kw * 1000.0, 0.0
    for year in range(1, max_years + 1):
        s = annual_saving_g_per_kw * (1 - decline_per_year) ** (year - 1)
        if acc + s >= need:
            return year - 1 + (need - acc) / s
        acc += s
    return float("inf")
