"""Can one schedule serve both carbon signals? Average-vs-marginal Pareto frontier (US BAs, EIA-930 2022).

F17 showed an average-optimised capped schedule gets only 7% of the marginal
optimum and raises marginal emissions in 44% of BAs. Here each BA schedules on
w*avg + (1-w)*marginal for w in [0, 1] and is scored on both signals.

phi = 0.5, 24 h slack, cap = 1/0.70 (as F17).
Outputs: results/hedge_*.csv, figures/hedge_frontier.png
Run: python scripts/hedge_experiments.py   (~3 min)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift import plotting as P
from carbonshift.eia930 import signals
from carbonshift.hedge import frontier

RESULTS = Path(__file__).resolve().parents[1] / "results"
PHI, SLACK, CAP = 0.5, 24, 1 / 0.70
W = np.round(np.linspace(0, 1, 11), 2)


def _task(args):
    ba, avg, mar = args
    return frontier(avg, mar, PHI, CAP, SLACK, W).assign(ba=ba)


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


if __name__ == "__main__":
    main()
