"""Do the findings survive a *marginal* carbon signal? US balancing authorities, EIA-930, 2022.

1. Validation: EIA-derived average intensity vs Electricity Maps (matching zones).
2. F15 under marginal signals: does the shadow price of capacity still track the
   signal's absolute daily swing?
3. Cross-signal: a schedule optimised on the average signal, evaluated in
   marginal emissions. What share of the marginal-optimal saving does it get?
4. Siting (F13): do BA rankings by mean intensity agree between average and marginal?

phi = 0.5, 24 h slack, cap = 1/0.70 (Google-real headroom, F12).
Outputs: results/marginal_*.csv, figures/marginal_signals.png
Run: python scripts/marginal_experiments.py   (~1 min)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift import plotting as P
from carbonshift.data import load_carbon
from carbonshift.eia930 import signals
from carbonshift.facility import capped_shift

RESULTS = Path(__file__).resolve().parents[1] / "results"
PHI, SLACK, CAP = 0.5, 24, 1 / 0.70


def daily_std(x: np.ndarray) -> float:
    n = len(x) // 24 * 24
    return float(x[:n].reshape(-1, 24).std(axis=1, ddof=1).mean())


def _task(args):
    ba, avg, mar = args
    ra = capped_shift(avg, PHI, CAP, SLACK)
    rm = capped_shift(mar, PHI, CAP, SLACK)
    base_m = float(np.mean(mar))
    got_m = base_m - float((mar * ((1 - PHI) + ra.served)).mean())  # average-optimal schedule, marginal accounting
    best_m = rm.baseline_g - rm.optimised_g
    return {"ba": ba, "mean_avg": float(avg.mean()), "mean_marg": base_m, "std_avg": daily_std(avg), "std_marg": daily_std(mar),
            "shadow_avg_kg": ra.annual_marginal_saving / 1000, "shadow_marg_kg": rm.annual_marginal_saving / 1000,
            "saving_avg_signal_pct": (ra.baseline_g - ra.optimised_g) / ra.baseline_g * 100,
            "marg_saving_best_g": best_m, "marg_saving_from_avg_schedule_g": got_m,
            "share_of_marg_optimum": got_m / best_m if abs(best_m) > 1e-9 else np.nan}


def main() -> None:
    P.setup()
    s = signals()
    piv_a = s.pivot(index="utc", columns="ba", values="average")
    piv_m = s.pivot(index="utc", columns="ba", values="marginal")
    bas = [b for b in piv_a.columns if piv_m[b].notna().all() and np.ptp(piv_a[b]) > 1e-9]

    # 1. validation against Electricity Maps
    em = load_carbon(2022)
    val = []
    for z in [c for c in em.columns if c.startswith("US-")]:
        ba = z.split("-")[-1]
        if ba in piv_a.columns:
            e, a = em[z].to_numpy(), piv_a[ba].to_numpy()[: len(em)]
            val.append({"zone": z, "ba": ba, "em_mean": e.mean(), "eia_mean": a.mean(), "hourly_r": np.corrcoef(e, a)[0, 1]})
    val = pd.DataFrame(val)
    val.to_csv(RESULTS / "marginal_validation.csv", index=False)

    with ProcessPoolExecutor() as ex:
        df = pd.DataFrame(list(ex.map(_task, [(b, piv_a[b].to_numpy(), piv_m[b].to_numpy()) for b in bas])))
    df.to_csv(RESULTS / "marginal_per_ba.csv", index=False)
    pos = df[df.marg_saving_best_g > 1e-6]
    summ = {
        "bas": len(df), "validation_zones": len(val),
        "validation_mean_r": float(np.corrcoef(val.em_mean, val.eia_mean)[0, 1]),
        "validation_hourly_r_median": float(val.hourly_r.median()),
        "rho_shadow_marg_vs_std_marg": df.shadow_marg_kg.corr(df.std_marg, method="spearman"),
        "rho_shadow_marg_vs_mean_marg": df.shadow_marg_kg.corr(df.mean_marg, method="spearman"),
        "rho_shadow_avg_vs_shadow_marg": df.shadow_avg_kg.corr(df.shadow_marg_kg, method="spearman"),
        "median_shadow_avg_kg": df.shadow_avg_kg.median(), "median_shadow_marg_kg": df.shadow_marg_kg.median(),
        "share_of_marg_optimum_pooled": pos.marg_saving_from_avg_schedule_g.sum() / pos.marg_saving_best_g.sum(),
        "share_of_marg_optimum_median": float(pos.share_of_marg_optimum.median()),
        "bas_where_avg_schedule_raises_marginal_pct": float((df.marg_saving_from_avg_schedule_g < 0).mean() * 100),
        "rho_siting_rank_mean_avg_vs_mean_marg": df.mean_avg.corr(df.mean_marg, method="spearman"),
    }
    pd.Series(summ).to_csv(RESULTS / "marginal_summary.csv", header=["value"])
    print(pd.Series(summ).round(3).to_string())
    print(df.sort_values("mean_avg")[["ba", "mean_avg", "mean_marg", "std_avg", "std_marg", "share_of_marg_optimum"]].round(2).head(12).to_string(index=False))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    d0 = df[(df.shadow_marg_kg > 0.1) & (df.std_marg > 0.1)]  # drops flat-hydro BAs with ~zero value
    axes[0].scatter(d0.std_marg, d0.shadow_marg_kg, s=14, color=P.BLUE)
    axes[0].set(xscale="log", yscale="log", xlabel="Daily std of marginal signal (g/kWh)", ylabel="Value of +1 kW (kg CO$_2$/yr)",
                title=f"Swing law, marginal signal (ρ = {summ['rho_shadow_marg_vs_std_marg']:.2f})")
    axes[1].scatter(df.mean_avg, df.mean_marg, s=14, color=P.ORANGE)
    lim = [0, max(df.mean_avg.max(), df.mean_marg.max()) * 1.05]
    axes[1].plot(lim, lim, color=P.INK_2, lw=0.8, ls="--")
    axes[1].set(xlabel="Mean average intensity (g/kWh)", ylabel="Mean marginal intensity (g/kWh)",
                title=f"Siting rank agreement (ρ = {summ['rho_siting_rank_mean_avg_vs_mean_marg']:.2f})")
    P.save(fig, "marginal_signals")


if __name__ == "__main__":
    main()
