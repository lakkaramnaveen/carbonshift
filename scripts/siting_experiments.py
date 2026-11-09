"""Where should the next kW of capacity go? Spatial vs temporal shadow prices across a cloud's regions.

Regions: GCP regions whose grid zones are in the carbon dataset (32 zones), with
measured inter-region latency. Every region has mean load 1, capacity 1/0.70
(the median utilisation of Google power domains, F12), and phi = 0.5 flexible.

  spatial value  : annual dual of each region's capacity in the hourly
                   migration LP (latency limit 50 / 100 / 250 ms), with no time shifting.
  temporal value : annual dual of the region's own capacity in the capped
                   time-shifting LP (24 h slack), with no migration.

Outputs: results/siting_*.csv, figures/siting_spatial_vs_temporal.png
Run: python scripts/siting_experiments.py   (~1 min)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift import plotting as P
from carbonshift.data import DATA_DIR, load_carbon
from carbonshift.facility import capped_shift
from carbonshift.spatial_capacity import spatial_shift

RESULTS = Path(__file__).resolve().parents[1] / "results"
PHI, CAP, SLACK = 0.5, 1 / 0.70, 24
LIMITS = (50, 100, 250)
ALIASES = {"AUS-NSW": "AU-NSW", "AUS-VIC": "AU-VIC"}  # latency-file codes -> carbon-data codes


def regions_and_latency(ci: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    lat = pd.read_csv(DATA_DIR / "gcp_latency_matrix.csv", index_col="origin").rename(index=ALIASES, columns=ALIASES)
    lat = lat[~lat.index.duplicated()]
    lat = lat.loc[:, ~lat.columns.duplicated()]
    zones = [z for z in lat.columns if z in ci.columns and z in lat.index]
    m = lat.loc[zones, zones].to_numpy(dtype=float)
    return zones, (m + m.T) / 2  # symmetrise (max asymmetry in the raw matrix is 2 ms)


def _spatial(args):
    limit, ci, zones, lat = args
    r = spatial_shift(ci, zones, lat, PHI, CAP, limit)
    return limit, r.annual_shadow, r.reduction_pct


def _temporal(args):
    zone, x = args
    r = capped_shift(x, PHI, CAP, SLACK)
    return zone, r.annual_marginal_saving, r.facility_reduction_pct


def main() -> None:
    P.setup()
    ci = load_carbon(2022)
    zones, lat = regions_and_latency(ci)
    X = ci[zones].to_numpy()
    with ProcessPoolExecutor() as ex:
        sp = list(ex.map(_spatial, [(L, X, zones, lat) for L in LIMITS]))
        tm = list(ex.map(_temporal, [(z, ci[z].to_numpy()) for z in zones]))

    df = pd.DataFrame({"zone": zones, "mean_ci": ci[zones].mean().values})
    g = ci[zones].groupby(ci.index.date)
    df["daily_cv"] = (g.std() / g.mean()).mean().values
    df["temporal_kg_per_kw_yr"] = [t[1] / 1000 for t in tm]
    for limit, shadow, _ in sp:
        df[f"spatial_{limit}ms_kg_per_kw_yr"] = shadow / 1000
    df.to_csv(RESULTS / "siting_per_region.csv", index=False)

    rows = []
    for limit, _, red in sp:
        col = f"spatial_{limit}ms_kg_per_kw_yr"
        rows.append({"latency_ms": limit, "spatial_reduction_pct": red,
                     "spearman_spatial_vs_temporal": df[col].corr(df.temporal_kg_per_kw_yr, method="spearman"),
                     "spearman_spatial_vs_mean_ci": df[col].corr(df.mean_ci, method="spearman"),
                     "spatial_median": df[col].median(), "spatial_max": df[col].max(),
                     "top3_spatial": ", ".join(df.nlargest(3, col).zone)})
    summ = pd.DataFrame(rows)
    summ["temporal_median"] = df.temporal_kg_per_kw_yr.median()
    summ["top3_temporal"] = ", ".join(df.nlargest(3, "temporal_kg_per_kw_yr").zone)
    summ["spearman_temporal_vs_daily_cv"] = df.temporal_kg_per_kw_yr.corr(df.daily_cv, method="spearman")
    summ["temporal_reduction_pct_mean"] = np.mean([t[2] for t in tm])
    summ.to_csv(RESULTS / "siting_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False))
    print(df.sort_values("spatial_100ms_kg_per_kw_yr", ascending=False).round(1).to_string(index=False))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    col = "spatial_100ms_kg_per_kw_yr"
    ax.scatter(df.temporal_kg_per_kw_yr, df[col], s=22, color=P.BLUE)
    label = set(df.nlargest(6, col).zone) | set(df.nlargest(3, "temporal_kg_per_kw_yr").zone)
    for i, (_, r) in enumerate(df[df.zone.isin(label)].sort_values("temporal_kg_per_kw_yr").iterrows()):
        dy = 2 if r[col] > 300 else (10 if i % 2 else -11)  # stagger labels near the x axis
        ax.annotate(r.zone, (r.temporal_kg_per_kw_yr, r[col]), xytext=(4, dy), textcoords="offset points", fontsize=7, color=P.INK_2)
    ax.set(xlabel="Temporal value of +1 kW (kg CO$_2$/yr)", ylabel="Spatial value of +1 kW\n(kg CO$_2$/yr, 100 ms SLO)",
           title="Where should the next kW go? (32 GCP regions)", ylim=(-220, None))
    P.save(fig, "siting_spatial_vs_temporal")


if __name__ == "__main__":
    main()
