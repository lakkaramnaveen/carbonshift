"""Can one schedule serve both carbon signals? Average-vs-marginal Pareto frontier (US BAs, EIA-930 2022).

F17 showed an average-optimised capped schedule gets only 7% of the marginal
optimum and raises marginal emissions in 44% of BAs. Here each BA schedules on
w*avg + (1-w)*marginal for w in [0, 1] and is scored on both signals.

phi = 0.5, 24 h slack, cap = 1/0.70 (as F17).

Out-of-sample check (F21): plan on w*avg + (1-w)*marginal with marginal factors
estimated on even days, score on factors from odd days (and vice versa).
Outputs: results/hedge_*.csv, figures/hedge_frontier.png
Run: python scripts/hedge_experiments.py   (~3 min)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift import plotting as P
from carbonshift.eia930 import signals, split_half_marginal
from carbonshift.facility import capped_shift
from carbonshift.hedge import frontier, saving

RESULTS = Path(__file__).resolve().parents[1] / "results"
PHI, SLACK, CAP = 0.5, 24, 1 / 0.70
W = np.round(np.linspace(0, 1, 11), 2)


def _task(args):
    ba, avg, mar = args
    return frontier(avg, mar, PHI, CAP, SLACK, W).assign(ba=ba)


def _oos_task(args):
    ba, avg, me, mo = args
    rows = []
    for w in W:
        xe = capped_shift(w * avg + (1 - w) * me, PHI, CAP, SLACK).served
        xo = capped_shift(w * avg + (1 - w) * mo, PHI, CAP, SLACK).served
        rows.append({"ba": ba, "w_avg": float(w),
                     "avg_saving_g": (saving(avg, xe, PHI) + saving(avg, xo, PHI)) / 2,
                     "marg_in_sample_g": (saving(me, xe, PHI) + saving(mo, xo, PHI)) / 2,
                     "marg_out_of_sample_g": (saving(mo, xe, PHI) + saving(me, xo, PHI)) / 2})
    return rows


def main() -> None:
    P.setup()
    s = signals()
    pa = s.pivot(index="utc", columns="ba", values="average")
    pm = s.pivot(index="utc", columns="ba", values="marginal")
    bas = [b for b in pa.columns if pm[b].notna().all() and np.ptp(pa[b]) > 1e-9]
    with ProcessPoolExecutor() as ex:
        df = pd.concat(ex.map(_task, [(b, pa[b].to_numpy(), pm[b].to_numpy()) for b in bas]), ignore_index=True)
    df.to_csv(RESULTS / "hedge_frontier_per_ba.csv", index=False)

    # keep BAs where both optima are meaningful (drops flat-hydro BAs with ~zero marginal swing)
    ok = df.groupby("ba").filter(lambda d: d.avg_saving_g.max() > 0.5 and d.marg_saving_g.max() > 0.5)
    pooled = ok.groupby("w_avg")[["avg_saving_g", "marg_saving_g"]].sum()
    best = ok.groupby("ba")[["avg_saving_g", "marg_saving_g"]].max().sum()
    pooled_share = pooled / best.values
    ok = ok.assign(worst=ok[["avg_share", "marg_share"]].min(axis=1))
    bal = ok.loc[ok.groupby("ba").worst.idxmax()]  # max-min weight per BA
    raise_avg = ok[(ok.w_avg == 0) & (ok.avg_saving_g < 0)].ba.nunique()
    raise_mar = ok[(ok.w_avg == 1) & (ok.marg_saving_g < 0)].ba.nunique()
    nba = ok.ba.nunique()
    summ = {
        "bas": nba,
        **{f"pooled_avg_share_w{w}": pooled_share.loc[w, "avg_saving_g"] for w in (0.0, 0.3, 0.5, 0.7, 1.0)},
        **{f"pooled_marg_share_w{w}": pooled_share.loc[w, "marg_saving_g"] for w in (0.0, 0.3, 0.5, 0.7, 1.0)},
        "balanced_worst_share_median": bal.worst.median(),
        "balanced_w_median": bal.w_avg.median(),
        "bas_with_some_w_both_ge_80pct_pct": (bal.worst >= 0.8).mean() * 100,
        "bas_with_some_w_both_ge_50pct_pct": (bal.worst >= 0.5).mean() * 100,
        "bas_marginal_schedule_raises_avg_pct": raise_avg / nba * 100,
        "bas_avg_schedule_raises_marginal_pct": raise_mar / nba * 100,
    }
    pd.Series(summ).to_csv(RESULTS / "hedge_summary.csv", header=["value"])
    bal.to_csv(RESULTS / "hedge_balanced_per_ba.csv", index=False)
    print(pd.Series(summ).round(3).to_string())
    print(pooled_share.round(3).to_string())

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    for b, d in ok.groupby("ba"):
        axes[0].plot(d.avg_share, d.marg_share, color=P.INK_2, lw=0.5, alpha=0.35)
    axes[0].plot(pooled_share.avg_saving_g, pooled_share.marg_saving_g, color=P.BLUE, lw=2, marker="o", ms=3, label="pooled")
    axes[0].axhline(0, color=P.INK_2, lw=0.6); axes[0].axvline(0, color=P.INK_2, lw=0.6)
    axes[0].set(xlabel="Share of average-signal optimum", ylabel="Share of marginal-signal optimum",
                ylim=(-1.0, 1.05), xlim=(-1.0, 1.05), title="Pareto frontier per BA (w = 0 → 1)")
    axes[0].legend(frameon=False, loc="lower left")
    axes[1].hist(bal.worst, bins=np.linspace(0, 1, 21), color=P.ORANGE)
    axes[1].set(xlabel="Best achievable min(avg share, marginal share)", ylabel="Balancing authorities",
                title=f"Balanced schedule (median {summ['balanced_worst_share_median']:.0%})")
    P.save(fig, "hedge_frontier")

    sp = split_half_marginal()
    pe = sp.pivot(index="utc", columns="ba", values="m_even")
    po = sp.pivot(index="utc", columns="ba", values="m_odd")
    keep = sorted(ok.ba.unique())
    keep = [b for b in keep if b in pe.columns and pe[b].notna().all() and po[b].notna().all()]
    with ProcessPoolExecutor() as ex:
        oos = pd.DataFrame([r for rows in ex.map(_oos_task, [(b, pa[b].to_numpy(), pe[b].to_numpy(), po[b].to_numpy()) for b in keep]) for r in rows])
    oos.to_csv(RESULTS / "hedge_oos_per_ba.csv", index=False)
    pool = oos.groupby("w_avg")[["avg_saving_g", "marg_in_sample_g", "marg_out_of_sample_g"]].sum()
    ref_a = pool.loc[1.0, "avg_saving_g"]
    ref_m_in = pool.loc[0.0, "marg_in_sample_g"]          # marginal-only plan, scored on its own half
    ref_m_oos = pool.loc[0.0, "marg_out_of_sample_g"]     # marginal-only plan, scored on the held-out half
    pool = pool.assign(avg_share=pool.avg_saving_g / ref_a, marg_oos_share_of_in_sample=pool.marg_out_of_sample_g / ref_m_in,
                       marg_oos_vs_marginal_only_oos=pool.marg_out_of_sample_g / ref_m_oos)
    best_w = oos.loc[oos.groupby("ba").marg_out_of_sample_g.idxmax()]
    pool.to_csv(RESULTS / "hedge_oos_pooled.csv")
    so = {"bas": len(keep), "pooled_best_w_for_oos_marginal": float(pool.marg_out_of_sample_g.idxmax()),
          "pooled_oos_marginal_gain_at_best_w_vs_w0_pct": (pool.marg_out_of_sample_g.max() / ref_m_oos - 1) * 100,
          "bas_where_best_oos_w_gt0_pct": float((best_w.w_avg > 0).mean() * 100),
          "median_best_oos_w": float(best_w.w_avg.median())}
    pd.Series(so).to_csv(RESULTS / "hedge_oos_summary.csv", header=["value"])
    print(pool.round(3).to_string())
    print(pd.Series(so).round(3).to_string())


if __name__ == "__main__":
    main()
