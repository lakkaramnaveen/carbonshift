"""Real data-centre power: how much headroom do facilities have, and does the daily load shape matter?

Uses Google PowerData2019 (57 power domains, May 2019) with the 2022 carbon traces
(120 zones with non-constant traces), phi = 0.5, 24 h slack.

(A) Real headroom: each domain's cap = 1 / mean utilisation. Share of the ideal
    saving kept (ratio of savings pooled over zones), per domain.
(B) Load shape: with the fleet-median utilisation, compare a flat load with the
    real hour-of-week shape (fleet median, and the most diurnal domain). The
    shape is laid over each grid at all 24 daily alignments, giving
    worst / mean / best case without assuming a time zone.

Outputs: results/diurnal_*.csv, figures/diurnal_headroom.png
Run: python scripts/diurnal_load_experiments.py   (~2 min on 14 cores)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift import plotting as P
from carbonshift.data import load_carbon
from carbonshift.facility import capped_shift
from carbonshift.google_power import domain_stats, tile, week_profile

RESULTS = Path(__file__).resolve().parents[1] / "results"
PHI, SLACK = 0.5, 24


def _headroom_task(args):
    zone, ci, caps = args
    ideal = capped_shift(ci, PHI, np.inf, SLACK)
    best = ideal.baseline_g - ideal.optimised_g
    return [{"zone": zone, "domain": dom, "cap": cap, "ideal_save": best,
             "save": (lambda r: r.baseline_g - r.optimised_g)(capped_shift(ci, PHI, cap, SLACK))}
            for dom, cap in caps]


def _shape_task(args):
    zone, ci, idx, cap, profiles = args
    rows = []
    flat = capped_shift(ci, PHI, cap, SLACK)
    for name, prof in profiles.items():
        for k in range(24):
            load = tile(prof, idx, k)
            r = capped_shift(ci, PHI, max(cap, load.max()), SLACK, load=load)
            rows.append({"zone": zone, "profile": name, "shift": k, "flat_pct": flat.facility_reduction_pct,
                         "shaped_pct": r.facility_reduction_pct})
    return rows


def main() -> None:
    P.setup()
    stats = domain_stats()
    stats.to_csv(RESULTS / "diurnal_domain_stats.csv")
    ci_df = load_carbon(2022)
    ci_df = ci_df.loc[:, ci_df.max() - ci_df.min() > 1e-9]
    caps = [(dom, 1.0 / u) for dom, u in stats.mean_util.items()]

    with ProcessPoolExecutor() as ex:
        a = [r for rows in ex.map(_headroom_task, [(z, ci_df[z].to_numpy(), caps) for z in ci_df.columns]) for r in rows]
    a = pd.DataFrame(a)
    per_dom = a.groupby("domain").agg(save=("save", "sum"), ideal=("ideal_save", "sum"), cap=("cap", "first"))
    per_dom["kept_pct"] = per_dom.save / per_dom.ideal * 100
    per_dom["headroom_pct"] = (per_dom.cap - 1) * 100
    per_dom.to_csv(RESULTS / "diurnal_real_headroom_per_domain.csv")

    u_med = stats.mean_util.median()
    most = stats.diurnal_amplitude.idxmax()
    profiles = {"fleet median": week_profile(None), f"most diurnal ({most})": week_profile(most)}
    idx = ci_df.index
    with ProcessPoolExecutor() as ex:
        b = [r for rows in ex.map(_shape_task, [(z, ci_df[z].to_numpy(), idx, 1 / u_med, profiles) for z in ci_df.columns])
             for r in rows]
    b = pd.DataFrame(b)
    b.to_csv(RESULTS / "diurnal_shape_per_zone.csv", index=False)
    zone_level = b.groupby(["profile", "zone"]).agg(flat=("flat_pct", "first"), worst=("shaped_pct", "min"),
                                                    mean=("shaped_pct", "mean"), best=("shaped_pct", "max"))
    shape_sum = zone_level.groupby("profile").mean()
    shape_sum["worst_loss_pp"] = shape_sum.flat - shape_sum.worst
    shape_sum["worst_loss_rel_pct"] = shape_sum.worst_loss_pp / shape_sum.flat * 100
    shape_sum.to_csv(RESULTS / "diurnal_shape_summary.csv")

    summary = {
        "domains": len(stats), "mean_util_median": u_med, "mean_util_min": stats.mean_util.min(), "mean_util_max": stats.mean_util.max(),
        "headroom_median_pct": (1 / u_med - 1) * 100, "peak_to_mean_median": stats.peak_to_mean.median(),
        "diurnal_amplitude_median_pct": stats.diurnal_amplitude.median() * 100, "diurnal_amplitude_max_pct": stats.diurnal_amplitude.max() * 100,
        "kept_median_pct": per_dom.kept_pct.median(), "kept_min_pct": per_dom.kept_pct.min(), "kept_max_pct": per_dom.kept_pct.max(),
    }
    pd.Series(summary).to_csv(RESULTS / "diurnal_summary.csv", header=["value"])
    print(pd.Series(summary).round(3).to_string())
    print(shape_sum.round(3).to_string())

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.scatter(per_dom.headroom_pct, per_dom.kept_pct, s=18, color=P.BLUE, label="Google power domain (May 2019)")
    ax.set(xlabel="Real headroom above mean load (%)", ylabel="Share of ideal carbon\nsaving kept (%)",
           title="Real facility headroom (57 domains x 120 grids)", ylim=(0, 100))
    ax.legend(fontsize=7.5, loc="lower right")
    P.save(fig, "diurnal_headroom")


if __name__ == "__main__":
    main()
