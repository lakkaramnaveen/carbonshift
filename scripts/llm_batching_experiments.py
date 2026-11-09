"""How much does carbon-aware (time-varying) batching save over the best static batching?

Measured M4 Pro curves (Qwen2.5 1.5B and 7B, 4-bit), 2022 carbon traces for all
zones with non-constant traces, flat demand. For each latency budget (mean time
per output token), compare the best static batch mix with an hourly carbon-aware
mix under the same yearly budget.

Outputs: results/llm_batching_*.csv, figures/llm_batching.png
Run: python scripts/llm_batching_experiments.py   (~1 min on 14 cores)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift import plotting as P
from carbonshift.data import load_carbon
from carbonshift.llm_batching import carbon_aware, measured_curve, static_best

RESULTS = Path(__file__).resolve().parents[1] / "results"
BUDGETS_MS = {"Qwen2.5-1.5B": (6, 10, 15, 20), "Qwen2.5-7B": (25, 50, 75)}


def _task(args):
    zone, ci = args
    rows = []
    for model, budgets in BUDGETS_MS.items():
        curve = measured_curve(model)
        b1 = float(np.mean(ci) * curve.energy_j[0])  # always batch 1
        for ms in budgets:
            s, _ = static_best(ci, curve, ms / 1000)
            d, x = carbon_aware(ci, curve, ms / 1000)
            rows.append({"zone": zone, "model": model, "budget_ms": ms, "batch1": b1, "static": s, "aware": d,
                         "mean_batch_aware": float((x @ curve.batch).mean())})
    return rows


def main() -> None:
    P.setup()
    ci = load_carbon(2022)
    ci = ci.loc[:, ci.max() - ci.min() > 1e-9]
    with ProcessPoolExecutor() as ex:
        df = pd.DataFrame([r for rows in ex.map(_task, [(z, ci[z].to_numpy()) for z in ci.columns]) for r in rows])
    df.to_csv(RESULTS / "llm_batching_per_zone.csv", index=False)
    df["aware_gain_pct"] = (df.static - df.aware) / df.static * 100
    g = df.groupby(["model", "budget_ms"])
    summ = pd.DataFrame({
        "static_vs_batch1_saving_pct": (1 - g.static.sum() / g.batch1.sum()) * 100,
        "aware_vs_static_saving_pct_pooled": (1 - g.aware.sum() / g.static.sum()) * 100,
        "aware_vs_static_saving_pct_median": g.aware_gain_pct.median(),
        "aware_vs_static_saving_pct_max": g.aware_gain_pct.max(),
        "best_zone": g.apply(lambda d: d.loc[d.aware_gain_pct.idxmax(), "zone"], include_groups=False),
    }).reset_index()
    summ.to_csv(RESULTS / "llm_batching_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(summ.round(2).to_string(index=False))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for k, (model, d) in enumerate(summ.groupby("model")):
        ax.plot(d.budget_ms, d.static_vs_batch1_saving_pct, "o-", color=(P.BLUE, P.ORANGE)[k], label=f"{model}: static batching vs batch 1")
        ax.plot(d.budget_ms, d.aware_vs_static_saving_pct_pooled, "s--", color=(P.BLUE, P.ORANGE)[k], label=f"{model}: carbon-aware vs static")
    ax.set(xlabel="Latency budget (ms per output token)", ylabel="CO$_2$ saving (%)", ylim=(0, None),
           title="Batch for efficiency first; carbon-aware timing adds little")
    ax.legend(fontsize=6.8, loc="center right")
    P.save(fig, "llm_batching")


if __name__ == "__main__":
    main()
