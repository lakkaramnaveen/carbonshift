"""The water-filling law: closed forms that replace the capped-shifting LP (F21).

Checks, against LP results already computed for 123 zones (2022, 24 h slack):
  1. Homogeneity theorem: retention depends on (phi, h) only through h/phi.
     Pairs of (phi, cap) with equal h/phi must give identical per-zone retention.
  2. Closed form S = phi * A * sinc(pi * phi/(phi+h)), A = mean daily (mean - min),
     vs the LP saving for 3 phi x 13 finite caps x 123 zones, plus the uncapped case.
  3. Value of a kW = daily "water level" area, vs LP shadow prices for 120 zones x 3 years.
  4. Peak draw of the uncapped schedule = 1 + slack * phi (slack + 1 arrivals can pile into one hour).

Inputs: results/facility_capped_per_zone.csv, results/robustness_years_per_zone.csv
Outputs: results/waterfill_*.csv, figures/waterfill_law.png
Run: python scripts/waterfill_experiments.py   (~30 s)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift import plotting as P
from carbonshift.data import load_carbon
from carbonshift.waterfill import daily_amplitude, retained_sinc, saving_sinc, value_waterfill

RESULTS = Path(__file__).resolve().parents[1] / "results"


def r2(pred, obs) -> float:
    pred, obs = np.asarray(pred), np.asarray(obs)
    return float(1 - ((pred - obs) ** 2).sum() / ((obs - obs.mean()) ** 2).sum())


def main() -> None:
    P.setup()
    lp = pd.read_csv(RESULTS / "facility_capped_per_zone.csv")
    lp = lp[lp.slack == 24].assign(save=lambda d: d.baseline_g - d.optimised_g)
    ci = load_carbon(2022)
    amp = {z: daily_amplitude(ci[z].to_numpy(), "min") for z in lp.zone.unique()}
    lp["A"] = lp.zone.map(amp)
    ideal = lp[np.isinf(lp.cap)].set_index(["zone", "phi"]).save
    lp["ideal"] = [ideal[(z, p)] for z, p in zip(lp.zone, lp.phi)]
    f = lp[~np.isinf(lp.cap)].copy()
    f["h_over_phi"] = (f.cap - 1) / f.phi
    f["retained"] = np.where(f.ideal > 1e-9, f.save / f.ideal, np.nan)
    f["pred_save"] = saving_sinc(f.A, f.phi, f.cap)
    f["pred_retained"] = retained_sinc(f.phi, f.cap)
    f.to_csv(RESULTS / "waterfill_saving_per_zone.csv", index=False)

    # 1. homogeneity: equal h/phi across phi -> equal retention, per zone
    f["key"] = f.h_over_phi.round(6)
    grp = f.groupby(["zone", "key"]).retained.agg(["max", "min", "count"])
    grp = grp[grp["count"] >= 2]
    homog_max_diff = float((grp["max"] - grp["min"]).max())

    # 2. closed-form saving
    m = f[f.save > 1e-6]
    uc = lp[np.isinf(lp.cap) & (lp.save > 1e-6)]

    # 3. value of a kW: water level vs LP dual
    s = pd.read_csv(RESULTS / "robustness_years_per_zone.csv")
    vals = []
    for y in sorted(s.year.unique()):
        c = load_carbon(int(y))
        for r in s[s.year == y].itertuples():
            if r.zone in c:
                vals.append({"year": y, "zone": r.zone, "lp_kg_per_kw_year": r.kg_per_kw_year,
                             "waterlevel_kg_per_kw_year": value_waterfill(c[r.zone].to_numpy(), 0.5, 1.2) / 1000})
    v = pd.DataFrame(vals)
    v.to_csv(RESULTS / "waterfill_value_per_zone.csv", index=False)
    vm = v[v.lp_kg_per_kw_year > 1]

    peak = pd.read_csv(RESULTS / "facility_capped_per_zone.csv")
    peak = peak[(peak.slack == 24) & np.isinf(peak.cap)]
    summ = {
        "homogeneity_pairs": int(len(grp)), "homogeneity_max_abs_diff_retained": homog_max_diff,
        "saving_cases": int(len(m) + len(uc)), "saving_r2_finite_caps": r2(m.pred_save, m.save),
        "saving_median_rel_err": float(((m.pred_save - m.save) / m.save).median()),
        "saving_median_abs_rel_err": float((abs(m.pred_save - m.save) / m.save).median()),
        "uncapped_r2_phiA_vs_lp": r2(uc.phi * uc.A, uc.save),
        "value_cases": int(len(vm)), "value_r2": r2(vm.waterlevel_kg_per_kw_year, vm.lp_kg_per_kw_year),
        "value_median_ratio": float((vm.waterlevel_kg_per_kw_year / vm.lp_kg_per_kw_year).median()),
        "value_median_abs_rel_err": float((abs(vm.waterlevel_kg_per_kw_year - vm.lp_kg_per_kw_year) / vm.lp_kg_per_kw_year).median()),
        "value_spearman": float(vm.waterlevel_kg_per_kw_year.corr(vm.lp_kg_per_kw_year, method="spearman")),
        **{f"peak_median_phi{p}_lp": float(peak[peak.phi == p].peak_draw.median()) for p in (0.25, 0.5, 0.75)},
        **{f"peak_law_phi{p}": 1 + 24 * p for p in (0.25, 0.5, 0.75)},
    }
    pd.Series(summ).to_csv(RESULTS / "waterfill_summary.csv", header=["value"])
    print(pd.Series(summ).round(4).to_string())

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    for phi, c in ((0.25, P.ORANGE), (0.5, P.BLUE), (0.75, P.INK_2)):
        g = f[(f.phi == phi) & f.retained.notna()].groupby("h_over_phi").retained.median()
        ax[0].plot(g.index, g.values, "o", ms=4, color=c, label=f"LP median, φ = {phi}")
    x = np.logspace(-1.4, 1.15, 300)
    ax[0].plot(x, retained_sinc(1.0, 1 + x), color="black", lw=1.2, label="sinc(π/(1+h/φ))")
    ax[0].set(xscale="log", xlabel="Headroom per unit of flexible load, h/φ", ylabel="Share of uncapped saving kept",
              title="Three curves, one variable")
    ax[0].legend(frameon=False, fontsize=7)
    ax[1].scatter(m.save, m.pred_save, s=3, alpha=0.35, color=P.BLUE)
    lim = [0, float(max(m.save.max(), m.pred_save.max())) * 1.05]
    ax[1].plot(lim, lim, color=P.INK_2, lw=0.8, ls="--")
    ax[1].set(xlabel="LP saving (g/kWh)", ylabel="φ·A·sinc(πq) (g/kWh)", title=f"Closed form vs LP (R² = {summ['saving_r2_finite_caps']:.3f})")
    ax[2].scatter(vm.lp_kg_per_kw_year, vm.waterlevel_kg_per_kw_year, s=8, alpha=0.6, color=P.ORANGE)
    lim = [1, float(vm[["lp_kg_per_kw_year", "waterlevel_kg_per_kw_year"]].max().max()) * 1.2]
    ax[2].plot(lim, lim, color=P.INK_2, lw=0.8, ls="--")
    ax[2].set(xscale="log", yscale="log", xlabel="LP shadow price (kg/kW-yr)", ylabel="Water-level area (kg/kW-yr)",
              title=f"Value of a kW (R² = {summ['value_r2']:.3f})")
    P.save(fig, "waterfill_law")


if __name__ == "__main__":
    main()
