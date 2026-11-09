"""Extension 3: shadow carbon price of capacity, embodied payback, and the redundancy dividend.

Part A, marginal value of headroom: for each zone and cap level, the LP dual
variables give the grams of CO2 saved per year by one more kW of headroom.
Dividing EPD-derived embodied carbon per kW (``carbonshift.embodied``) by that
saving gives the carbon payback time of building headroom for carbon-aware shifting.

Part B, redundancy dividend: tier-rated redundant paths (N+1, 2N) used as
headroom for sheddable flexible work, with planned maintenance and random
failures returning the facility to design capacity. Maintenance is placed
either at random or in the hours where capacity is worth least (lowest
summed shadow price).

Outputs: results/redundancy_*.csv, figures/redundancy_*.png
Run: python scripts/redundancy_experiments.py   (~3 min on 14 cores)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift import plotting as P
from carbonshift.data import load_carbon
from carbonshift.embodied import BUILDS, LIFETIME, per_kw, payback_years
from carbonshift.facility import capped_shift
from carbonshift.redundancy import (OutagePlan, cap_profile, carbon_aware_windows, failure_windows,
                                    installed_cap, random_windows)

RESULTS = Path(__file__).resolve().parents[1] / "results"
PHI = 0.5
SLACKS = (24, 168)
MARGINAL_CAPS = (1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0)
UTILISATIONS = (0.5, 0.65, 0.8)
TOPOLOGIES = ("design_only", "N+1 (N=4)", "2N")
PLAN = OutagePlan()
SEEDS = 10
LIFETIME_Y = 25
DECLINE = 0.03  # sensitivity: savings shrink 3%/yr as grids decarbonise


def _zone(args):
    zone, ci = args
    L = len(ci)
    marg, red = [], []
    for slack in SLACKS:
        for cap in MARGINAL_CAPS:
            r = capped_shift(ci, PHI, cap, slack)
            marg.append({"zone": zone, "slack": slack, "cap": cap, "g_per_kw_year": r.annual_marginal_saving,
                         "reduction_pct": r.facility_reduction_pct})
    slack = 24
    ideal = capped_shift(ci, PHI, np.inf, slack).facility_reduction_pct
    for u in UTILISATIONS:
        design = 1 / u
        for topo in TOPOLOGIES:
            normal = installed_cap(u, topo)
            perfect = capped_shift(ci, PHI, normal, slack)
            row = {"zone": zone, "u": u, "topology": topo, "cap": normal, "ideal_pct": ideal,
                   "perfect_pct": perfect.facility_reduction_pct}
            if topo != "design_only":
                rand, aware = [], []
                aware_starts = carbon_aware_windows(perfect.shadow, PLAN.maintenance_windows, PLAN.maintenance_hours)
                for seed in range(SEEDS):
                    rng = np.random.default_rng(seed)
                    fails = failure_windows(rng, L, PLAN)
                    rs = random_windows(rng, L, PLAN.maintenance_windows, PLAN.maintenance_hours)
                    for starts, sink in ((rs, rand), (aware_starts, aware)):
                        w = [(s, PLAN.maintenance_hours) for s in starts] + fails
                        sink.append(capped_shift(ci, PHI, cap_profile(L, normal, design, w), slack).facility_reduction_pct)
                row.update(random_maint_pct=float(np.mean(rand)), aware_maint_pct=float(np.mean(aware)))
            red.append(row)
    return marg, red


def main() -> None:
    import sys

    P.setup()
    if "--figures-only" in sys.argv:
        figures(pd.read_csv(RESULTS / "redundancy_marginal_per_zone.csv"), pd.read_csv(RESULTS / "redundancy_dividend_per_zone.csv"))
        return
    ci = load_carbon(2022)
    with ProcessPoolExecutor() as ex:
        out = list(ex.map(_zone, [(z, ci[z].to_numpy()) for z in ci.columns]))
    marg = pd.DataFrame([r for m, _ in out for r in m])
    red = pd.DataFrame([r for _, rr in out for r in rr])

    # ---- Part A: payback of building headroom
    for build in BUILDS:
        lo, mid, hi = per_kw(build)
        marg[f"payback_{build}_y"] = [payback_years(mid, g) for g in marg.g_per_kw_year]
        marg[f"payback_{build}_decline_y"] = [payback_years(mid, g, DECLINE) for g in marg.g_per_kw_year]
        marg[f"payback_{build}_hi_y"] = [payback_years(hi, g) for g in marg.g_per_kw_year]
    marg.to_csv(RESULTS / "redundancy_marginal_per_zone.csv", index=False)

    aggs = {"g_per_kw_year_median": ("g_per_kw_year", "median"),
            "kg_per_kw_year_p90": ("g_per_kw_year", lambda s: s.quantile(0.9) / 1000)}
    for build in BUILDS:
        life = LIFETIME[build]
        aggs[f"{build}_payback_median_y"] = (f"payback_{build}_y", "median")
        aggs[f"{build}_pays_back_in_life_pct"] = (f"payback_{build}_decline_y", lambda s, life=life: float((s < life).mean() * 100))
    summ = marg.groupby(["slack", "cap"]).agg(**aggs).reset_index()
    summ["g_per_kw_year_median"] /= 1000
    summ = summ.rename(columns={"g_per_kw_year_median": "kg_per_kw_year_median"})
    summ.to_csv(RESULTS / "redundancy_payback_summary.csv", index=False)

    # ---- Part B: redundancy dividend
    red.to_csv(RESULTS / "redundancy_dividend_per_zone.csv", index=False)
    rsum = red.groupby(["u", "topology"])[["cap", "ideal_pct", "perfect_pct", "random_maint_pct", "aware_maint_pct"]].mean().reset_index()
    rsum.to_csv(RESULTS / "redundancy_dividend_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print("embodied kg/kW (low, central, high):", {b: tuple(round(x, 1) for x in per_kw(b)) for b in BUILDS})
    print(summ.round(2).to_string(index=False))
    print(rsum.round(3).to_string(index=False))
    figures(marg, red)


def figures(marg: pd.DataFrame, red: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    # A) marginal value of headroom (median + IQR across zones) and embodied breakeven lines
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for k, slack in enumerate(SLACKS):
        d = marg[marg.slack == slack].groupby("cap").g_per_kw_year
        x = (np.array(sorted(marg.cap.unique())) - 1) * 100
        med, q1, q3 = d.median() / 1000, d.quantile(0.25) / 1000, d.quantile(0.75) / 1000
        col = (P.BLUE, P.ORANGE)[k]
        ax.plot(x, med, "o-", color=col, label=f"median zone, {slack} h slack")
        ax.fill_between(x, q1, q3, color=col, alpha=0.15, linewidth=0)
    names = {"backed": "electrical plant, fully backed", "gpu_servers": "GPU servers", "cpu_servers": "CPU servers"}
    for build, ls in (("backed", "--"), ("gpu_servers", "-."), ("cpu_servers", ":")):
        need = per_kw(build)[1] / LIFETIME[build]
        ax.axhline(need, color=P.INK_2, lw=1, ls=ls)
        left = build == "backed"  # near zero: label at the left, clear of the curves
        text = (f"electrical plant: {need:.0f} kg/yr" if left else f"breakeven: {names[build]} ({LIFETIME[build]} y life)")
        ax.annotate(text,
                    (x.min() if left else x.max(), need), xytext=(4 if left else 0, 3), textcoords="offset points",
                    ha="left" if left else "right", fontsize=7, color=P.INK_2)
    ax.set(xlabel="Existing headroom above mean load (%)", ylabel="CO$_2$ saved by +1 kW headroom\n(kg per year)",
           title="Shadow carbon price of facility capacity", ylim=(0, None))
    ax.legend(fontsize=7.5, loc="upper right", bbox_to_anchor=(1, 0.88))
    P.save(fig, "redundancy_marginal")

    # B) redundancy dividend at u = 0.65
    d = red[red.u == 0.65].groupby("topology")[["perfect_pct", "random_maint_pct", "aware_maint_pct", "ideal_pct"]].mean()
    d = d.reindex(TOPOLOGIES)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = np.arange(len(d))
    ax.bar(x - 0.2, d.perfect_pct, 0.38, color=P.BLUE, label="no outages")
    ax.bar(x + 0.2, d.random_maint_pct.fillna(d.perfect_pct), 0.38, color=P.ORANGE, label="with maintenance + failures (spare path only)")
    ax.axhline(d.ideal_pct.iloc[0], color=P.INK_2, lw=1, ls="--")
    ax.annotate("unlimited power (ideal)", (len(d) - 0.6, d.ideal_pct.iloc[0]), xytext=(0, 3), textcoords="offset points",
                ha="right", fontsize=7.5, color=P.INK_2)
    labels = ["Design capacity only\n(u = 65%)", "N+1 spare used\nfor flex work", "2N spare used\nfor flex work"]
    ax.set_xticks(x, labels)
    ax.set(ylabel="Facility CO$_2$ reduction (%)", title="Redundancy dividend (mean of 123 zones)")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax.set_ylim(0, d.ideal_pct.iloc[0] * 1.15)
    P.save(fig, "redundancy_dividend")


if __name__ == "__main__":
    main()
