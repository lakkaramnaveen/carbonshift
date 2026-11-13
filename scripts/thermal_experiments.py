"""Chilled-water thermal storage as a third virtual battery (F27).

47 cloud-hosting zones, 2022 carbon and ERA5 weather, both cooling plants.
Flat IT load (heat = 1), tank of 1-12 h of mean heat load, chiller 1.5x peak,
1%/day standing loss. For each zone and plant:

  * TES saving vs no tank (g per kWh of mean IT, and % of facility emissions)
  * what drives it: dispatch on carbon only (c * mean m), on weather only
    (mean c * m) or on both (c * m), each scored on the true c * m
  * exchange rate: kWh of 4 h, 90% Li-ion battery (F23 spread law) saving the
    same carbon as one kWh of tank (thermal)
  * stacking with IT shifting (F26): cooling-aware IT schedule (phi 0.5,
    +20% headroom, 24 h slack), then the tank on the resulting heat load

Outputs: results/thermal_*.csv, figures/thermal.png
Run: python scripts/thermal_experiments.py   (~2 min)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift import plotting as P
from carbonshift.cooling import HUBS, PLANTS, load_weather, plant_pue
from carbonshift.data import load_carbon
from carbonshift.exchange import spread_saving_per_kwh
from carbonshift.facility import capped_shift
from carbonshift.thermal import mech_per_heat, tes_shift

RESULTS = Path(__file__).resolve().parents[1] / "results"
CAPS = (1, 2, 4, 8, 12)
PHI, HEAD, SLACK, CAP_MAIN = 0.5, 0.2, 24, 4


def _run(args):
    zone, ci, temps = args
    rows = []
    batt = spread_saving_per_kwh(ci, 4.0, 0.90)
    for plant_name, t in temps.items():
        plant = PLANTS[plant_name]
        m = mech_per_heat(t, plant)
        p = 1 + plant.fixed + plant.fans + m
        facility = float((ci * p).mean())
        base = {"zone": zone, "plant": plant_name, "facility_g": facility, "mech_mean": float(m.mean()),
                "cooling_share_pct": float((ci * m).mean() / facility * 100), "battery_g_per_kwh": batt}
        for cap in CAPS:
            r = tes_shift(ci, m, cap)
            row = {**base, "capacity_h": cap, "tes_saving_g": r.saving_g,
                   "tes_saving_pct": r.saving_g / facility * 100,
                   "battery_kwh_per_tes_kwh": (r.saving_g / cap) / batt if batt > 0 else np.nan}
            if cap == CAP_MAIN:
                price = ci * m
                for name, sig_c, sig_m in (("carbon_only", ci, np.full_like(m, m.mean())),
                                           ("weather_only", np.full_like(ci, ci.mean()), m)):
                    q = tes_shift(sig_c, sig_m, cap).produced
                    row[f"tes_{name}_g"] = r.baseline_g - float((price * q).mean())
                # stacking with cooling-aware IT shifting
                d = (1 - PHI) + capped_shift(ci * p, PHI, 1 + HEAD, SLACK).served
                it_only = facility - float((ci * p * d).mean())
                rt = tes_shift(ci, m, cap, heat=d, q_max=1.5 * (1 + HEAD))
                both = facility - (float((ci * (1 + plant.fixed + plant.fans) * d).mean()) + rt.optimised_g)
                row.update({"it_shift_g": it_only, "it_plus_tes_g": both,
                            "stack_ratio": both / (it_only + r.saving_g) if it_only + r.saving_g > 1e-9 else np.nan})
            rows.append(row)
    return rows


def main() -> None:
    ci, w = load_carbon(2022), load_weather(2022)
    jobs = []
    for z in HUBS:
        temps = {k: w[f"{z}|{'wb' if PLANTS[k].wet_bulb else 'db'}"].to_numpy() for k in PLANTS}
        jobs.append((z, ci[z].to_numpy(), temps))
    with ProcessPoolExecutor() as ex:
        df = pd.DataFrame([r for rows in ex.map(_run, jobs) for r in rows])
    df.to_csv(RESULTS / "thermal_per_zone.csv", index=False)

    s = {"zones": df.zone.nunique()}
    for plant in PLANTS:
        g = df[df.plant == plant]
        s[f"{plant}_cooling_share_pct_median"] = float(g.drop_duplicates("zone").cooling_share_pct.median())
        for cap in CAPS:
            x = g[g.capacity_h == cap]
            s[f"{plant}_cap{cap}_saving_pct_median"] = float(x.tes_saving_pct.median())
            s[f"{plant}_cap{cap}_saving_g_median"] = float(x.tes_saving_g.median())
            s[f"{plant}_cap{cap}_battery_kwh_per_tes_kwh_median"] = float(x.battery_kwh_per_tes_kwh.median())
        x = g[g.capacity_h == CAP_MAIN]
        ok = x.tes_saving_g > 0.05
        s[f"{plant}_share_carbon_only_pooled"] = float(x.tes_carbon_only_g[ok].sum() / x.tes_saving_g[ok].sum())
        s[f"{plant}_share_weather_only_pooled"] = float(x.tes_weather_only_g[ok].sum() / x.tes_saving_g[ok].sum())
        s[f"{plant}_zones_weather_beats_carbon"] = int((x.tes_weather_only_g > x.tes_carbon_only_g)[ok].sum())
        s[f"{plant}_zones_ok"] = int(ok.sum())
        s[f"{plant}_zones_carbon_only_raises"] = int((x.tes_carbon_only_g < 0).sum())
        s[f"{plant}_it_shift_g_median"] = float(x.it_shift_g.median())
        s[f"{plant}_tes_over_it_pooled"] = float(x.tes_saving_g.sum() / x.it_shift_g.sum())
        s[f"{plant}_stack_ratio_pooled"] = float(x.it_plus_tes_g.sum() / (x.it_shift_g + x.tes_saving_g).sum())
        s[f"{plant}_stack_ratio_min"] = float(x.stack_ratio.min())
    pd.Series(s).to_csv(RESULTS / "thermal_summary.csv", header=["value"])
    for k, v in s.items():
        print(f"{k:50s} {v:.4g}")
    figure(df)


def figure(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    P.setup()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for plant, col in (("legacy", P.ORANGE), ("hyperscale", P.BLUE)):
        g = df[df.plant == plant]
        med = g.groupby("capacity_h").tes_saving_pct.median()
        q1 = g.groupby("capacity_h").tes_saving_pct.quantile(0.25)
        q3 = g.groupby("capacity_h").tes_saving_pct.quantile(0.75)
        a.plot(med.index, med.values, color=col, marker="o", ms=3, label=plant)
        a.fill_between(med.index, q1.values, q3.values, color=col, alpha=0.15, lw=0)
        x = g[g.capacity_h == CAP_MAIN]
        b.scatter(x.tes_carbon_only_g / x.tes_saving_g, x.tes_weather_only_g / x.tes_saving_g, s=10, color=col,
                  lw=0, alpha=0.8, label=plant)
    a.set(xlabel="Tank size (hours of mean heat load)", ylabel="Facility emissions cut (%)",
          title="A chilled-water tank as a battery")
    a.legend(fontsize=7)
    b.plot([0, 1], [0, 1], color=P.INK_2, lw=0.8)
    b.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), xlabel="Share kept dispatching on carbon only",
          ylabel="Share kept dispatching on weather only", title=f"What drives the tank ({CAP_MAIN} h)")
    fig.tight_layout()
    P.save(fig, "thermal")


if __name__ == "__main__":
    main()
