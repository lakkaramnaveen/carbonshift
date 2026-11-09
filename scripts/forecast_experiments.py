"""Does forecast error hurt power-capped shifting more than uncapped shifting?

All zones (2022), phi = 0.5, 24 h slack. Schedules are optimised on an AR(1)
noisy forecast (MAPE 5, 10, 15 and 30%) and charged the true carbon intensity.
Lost share = (perfect saving - forecast saving) / perfect saving, pooled over
zones (ratio of sums) and over 3 seeds.

Outputs: results/forecast_per_zone.csv, results/forecast_summary.csv, figures/forecast_regret.png
Run: python scripts/forecast_experiments.py   (~2 min on 14 cores)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift import plotting as P
from carbonshift.data import load_carbon
from carbonshift.forecast import regret

RESULTS = Path(__file__).resolve().parents[1] / "results"
MAPES = (0.05, 0.10, 0.15, 0.30)
CAPS = (1.2, 1.5, np.inf)
SEEDS = 3
PHI, SLACK = 0.5, 24


def _zone(args):
    k, zone, ci = args
    if np.ptp(ci) < 1e-9:
        return []
    rows = []
    for mape in MAPES:
        for cap in CAPS:
            for seed in range(SEEDS):
                rng = np.random.default_rng([k, seed, int(mape * 1000)])  # same noise across caps
                rows.append({"zone": zone, "mape": mape, "cap": cap, "seed": seed, **regret(ci, mape, PHI, cap, SLACK, rng)})
    return rows


def main() -> None:
    P.setup()
    ci = load_carbon(2022)
    with ProcessPoolExecutor() as ex:
        out = ex.map(_zone, [(k, z, ci[z].to_numpy()) for k, z in enumerate(ci.columns)])
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv(RESULTS / "forecast_per_zone.csv", index=False)
    g = df.groupby(["mape", "cap"])
    summ = pd.DataFrame({
        "lost_share_pooled_pct": (g.perfect_saving_g.sum() - g.forecast_saving_g.sum()) / g.perfect_saving_g.sum() * 100,
        "lost_share_median_pct": g.lost_share.median() * 100,
        "zones_where_shifting_backfires_pct": g.apply(lambda d: (d.forecast_saving_g < 0).mean() * 100, include_groups=False),
    }).reset_index()
    summ["lost_pct_of_emissions"] = ((g.perfect_saving_g.sum() - g.forecast_saving_g.sum()) / g.baseline_g.sum() * 100).values
    # share of the (equally mis-forecast) uncapped saving that a capped facility keeps
    unc = df[np.isinf(df.cap)].groupby("mape")[["perfect_saving_g", "forecast_saving_g"]].sum()
    tot = df.groupby(["mape", "cap"])[["perfect_saving_g", "forecast_saving_g"]].sum()
    summ["kept_vs_uncapped_perfect_pct"] = [tot.loc[(m, c), "perfect_saving_g"] / unc.loc[m, "perfect_saving_g"] * 100 for m, c in zip(summ.mape, summ.cap)]
    summ["kept_vs_uncapped_forecast_pct"] = [tot.loc[(m, c), "forecast_saving_g"] / unc.loc[m, "forecast_saving_g"] * 100 for m, c in zip(summ.mape, summ.cap)]
    summ.to_csv(RESULTS / "forecast_summary.csv", index=False)
    print(summ.round(2).to_string(index=False))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for k, cap in enumerate(CAPS):
        d = summ[summ.cap == cap]
        label = "no cap (ideal)" if np.isinf(cap) else f"+{round((cap - 1) * 100)}% headroom"
        ax.plot(d.mape * 100, d.lost_share_pooled_pct, "o-", color=(P.BLUE, P.ORANGE, P.AQUA)[k], label=label)
    ax.set(xlabel="Forecast error, MAPE (%)", ylabel="Share of perfect-forecast\nsaving lost (%)",
           title="Forecast error vs power cap (φ = 0.5, 24 h slack)", ylim=(0, None))
    ax.legend(fontsize=8, loc="upper left")
    P.save(fig, "forecast_regret")


if __name__ == "__main__":
    main()
