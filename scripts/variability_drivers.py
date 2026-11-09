"""What drives the carbon value of capacity? Absolute swing, not level.

Lemma (tests/test_facility.py::test_affine_invariance...): for ci' = a + b*ci the
capped schedule is unchanged and savings and shadow prices scale by b, so only
*absolute* within-window variation matters, not the mean intensity. This script
checks the implication empirically on 120 zones x 2020-2022, using the
per-zone shadow prices from scripts/robustness_years.py:

  * cross-section: Spearman correlation of the shadow price with daily absolute
    std (g/kWh), mean intensity, and daily coefficient of variation;
  * change: does the 2020 -> 2022 change in shadow price follow the change in
    daily absolute std?

Outputs: results/variability_*.csv, figures/variability_drivers.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift import plotting as P
from carbonshift.data import load_carbon

RESULTS = Path(__file__).resolve().parents[1] / "results"


def zone_stats(year: int) -> pd.DataFrame:
    ci = load_carbon(year)
    g = ci.groupby(ci.index.date)
    return pd.DataFrame({"mean_ci": ci.mean(), "daily_std": g.std().mean(), "daily_cv": (g.std() / g.mean()).mean()}).assign(year=year)


def main() -> None:
    P.setup()
    shadow = pd.read_csv(RESULTS / "robustness_years_per_zone.csv")[["year", "zone", "kg_per_kw_year"]]
    stats = pd.concat([zone_stats(y).rename_axis("zone").reset_index() for y in (2020, 2021, 2022)])
    df = shadow.merge(stats, on=["year", "zone"])
    df.to_csv(RESULTS / "variability_per_zone_year.csv", index=False)

    rows = []
    for y, d in df.groupby("year"):
        rows.append({"year": y, **{f"rho_{c}": d.kg_per_kw_year.corr(d[c], method="spearman") for c in ("daily_std", "mean_ci", "daily_cv")}})
    cross = pd.DataFrame(rows)

    w = df.pivot(index="zone", columns="year")
    dlt = pd.DataFrame({
        "d_shadow": w[("kg_per_kw_year", 2022)] - w[("kg_per_kw_year", 2020)],
        "d_std": w[("daily_std", 2022)] - w[("daily_std", 2020)],
        "d_mean": w[("mean_ci", 2022)] - w[("mean_ci", 2020)],
    }).dropna()
    # log-log elasticity of shadow price w.r.t. daily std (cross-section, pooled years)
    pos = df[(df.kg_per_kw_year > 0) & (df.daily_std > 0)]
    slope, intercept = np.polyfit(np.log(pos.daily_std), np.log(pos.kg_per_kw_year), 1)
    summ = {
        **{f"{k}_{int(r.year)}": v for _, r in cross.iterrows() for k, v in r.items() if k != "year"},
        "rho_change_shadow_vs_change_std": dlt.d_shadow.corr(dlt.d_std, method="spearman"),
        "rho_change_shadow_vs_change_mean": dlt.d_shadow.corr(dlt.d_mean, method="spearman"),
        "median_daily_std_2020": df[df.year == 2020].daily_std.median(),
        "median_daily_std_2022": df[df.year == 2022].daily_std.median(),
        "median_mean_ci_2020": df[df.year == 2020].mean_ci.median(),
        "median_mean_ci_2022": df[df.year == 2022].mean_ci.median(),
        "loglog_elasticity_shadow_vs_std": slope,
        "loglog_r2": np.corrcoef(np.log(pos.daily_std), np.log(pos.kg_per_kw_year))[0, 1] ** 2,
    }
    pd.Series(summ).to_csv(RESULTS / "variability_summary.csv", header=["value"])
    print(pd.Series(summ).round(3).to_string())

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    d = df[df.year == 2022]
    axes[0].scatter(d.daily_std, d.kg_per_kw_year, s=14, color=P.BLUE)
    axes[0].set(xscale="log", yscale="log", xlabel="Daily std of carbon intensity (g/kWh)", ylabel="Value of +1 kW (kg CO$_2$/yr)",
                title=f"Swing drives value (2022; ρ = {d.kg_per_kw_year.corr(d.daily_std, method='spearman'):.2f})")
    axes[1].scatter(d.mean_ci, d.kg_per_kw_year, s=14, color=P.ORANGE)
    axes[1].set(yscale="log", xlabel="Mean carbon intensity (g/kWh)",
                title=f"Level does not (ρ = {d.kg_per_kw_year.corr(d.mean_ci, method='spearman'):.2f})")
    P.save(fig, "variability_drivers")


if __name__ == "__main__":
    main()
