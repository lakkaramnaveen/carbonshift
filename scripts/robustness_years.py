"""Robustness: do the facility findings (F4, F7, F8, F9) hold for 2020, 2021 and 2022?

For every zone and year (phi = 0.5, 24 h slack) this computes:
  * share of the ideal saving kept at +20% and +50% headroom          (F4)
  * shadow carbon price of capacity at +20% headroom, kg/kW-yr       (F7)
  * payback of grid-only and CPU-server headroom                     (F8)
  * redundancy dividend at u = 0.65 (design / N+1 / 2N)              (F9)

Outputs: results/robustness_years_per_zone.csv, results/robustness_years_summary.csv
Run: python scripts/robustness_years.py   (~1 min on 14 cores)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift.data import load_carbon
from carbonshift.embodied import LIFETIME, per_kw, payback_years
from carbonshift.facility import capped_shift
from carbonshift.redundancy import installed_cap

RESULTS = Path(__file__).resolve().parents[1] / "results"
YEARS = (2020, 2021, 2022)
PHI, SLACK, U = 0.5, 24, 0.65


def _task(args):
    year, zone, ci = args
    if np.ptp(ci) < 1e-9:  # constant (or fully gap-filled) trace: no shifting possible
        return None
    ideal = capped_shift(ci, PHI, np.inf, SLACK)
    ideal_save = ideal.baseline_g - ideal.optimised_g
    row = {"year": year, "zone": zone, "ideal_pct": ideal.facility_reduction_pct, "ideal_save_g": ideal_save}
    for cap in (1.2, 1.5):
        r = capped_shift(ci, PHI, cap, SLACK)
        row[f"save_{cap}_g"] = r.baseline_g - r.optimised_g
        if cap == 1.2:
            row["kg_per_kw_year"] = r.annual_marginal_saving / 1000 * 8760 / len(ci)  # normalise leap years
    for topo in ("design_only", "N+1 (N=4)", "2N"):
        row[f"red_{topo}"] = capped_shift(ci, PHI, installed_cap(U, topo), SLACK).facility_reduction_pct
    g = row["kg_per_kw_year"] * 1000
    row["payback_grid_only_y"] = payback_years(per_kw("grid_only")[1], g)
    row["payback_cpu_y"] = payback_years(per_kw("cpu_servers")[1], g, 0.03)
    return row


def main() -> None:
    jobs = []
    for y in YEARS:
        ci = load_carbon(y)
        jobs += [(y, z, ci[z].to_numpy()) for z in ci.columns]
    with ProcessPoolExecutor() as ex:
        rows = [r for r in ex.map(_task, jobs, chunksize=4) if r is not None]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "robustness_years_per_zone.csv", index=False)

    def gap(d, topo):
        return ((d[f"red_{topo}"] - d["red_design_only"]).mean() / (d["ideal_pct"] - d["red_design_only"]).mean()) * 100

    summ = []
    for y, d in df.groupby("year"):
        summ.append({
            "year": y, "zones": len(d),
            "ideal_pct_mean": d.ideal_pct.mean(),
            # same definition as facility_summary.csv: ratio of mean absolute savings
            "kept_20pct_headroom": d["save_1.2_g"].mean() / d.ideal_save_g.mean() * 100,
            "kept_50pct_headroom": d["save_1.5_g"].mean() / d.ideal_save_g.mean() * 100,
            "kg_per_kw_year_median": d.kg_per_kw_year.median(),
            "grid_only_payback_median_y": d.payback_grid_only_y.median(),
            "cpu_pays_back_in_life_pct": (d.payback_cpu_y < LIFETIME["cpu_servers"]).mean() * 100,
            "gap_closed_N+1_pct": gap(d, "N+1 (N=4)"),
            "gap_closed_2N_pct": gap(d, "2N"),
        })
    summ = pd.DataFrame(summ)
    summ.to_csv(RESULTS / "robustness_years_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(summ.round(2).to_string(index=False))

    # Year-to-year stability of the per-zone shadow price (rank correlation)
    piv = df.pivot(index="zone", columns="year", values="kg_per_kw_year").dropna()
    print("Spearman rank correlation of per-zone shadow price between years:")
    print(piv.corr(method="spearman").round(3))
    piv.corr(method="spearman").to_csv(RESULTS / "robustness_years_shadow_rankcorr.csv")


if __name__ == "__main__":
    main()
