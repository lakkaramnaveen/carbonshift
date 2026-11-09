"""Reproduce the headline results of Sukprasert et al., EuroSys'24 (2022 traces, 123 zones).

Outputs:
  results/repro_temporal.csv   per (slack, job, grouping): defer / interrupt_extra / combined, g per job-hour
  results/repro_relative.csv   Figure 9: % of baseline
  results/repro_spatial.csv    Figure 5(a,b) per grouping (artifact metric and corrected metric)
  results/repro_vs_paper.csv   every numeric claim checked, paper vs ours
  figures/repro_*.png

Run: python scripts/reproduce.py   (about 1 minute on 8+ cores)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift import plotting as P
from carbonshift.data import DATA_DIR, GEO_GROUPINGS, as_matrix, grouping_members, load_carbon
from carbonshift.spatial import capacity_latency, global_idle_curve, grouping_reduction, one_vs_inf_migration
from carbonshift.temporal import JOB_LENGTHS, JOB_WEIGHTS, SLACKS, relative_reduction, shift_one_zone

RESULTS = Path(__file__).resolve().parents[1] / "results"
YEAR = 2022


def _zone_task(args):
    zone, ci = args
    rows = []
    for slack in SLACKS:
        for job in JOB_LENGTHS:
            rows.append({"zone": zone, "slack": slack, "job": job, **shift_one_zone(ci, job, slack).per_unit(job)})
    return rows


def temporal(ci_df: pd.DataFrame) -> pd.DataFrame:
    with ProcessPoolExecutor() as ex:
        chunks = ex.map(_zone_task, [(z, ci_df[z].to_numpy()) for z in ci_df.columns])
    per_zone = pd.DataFrame([r for chunk in chunks for r in chunk])
    per_zone.to_csv(RESULTS / "repro_temporal_per_zone.csv", index=False)
    out = []
    for g in GEO_GROUPINGS:
        sub = per_zone[per_zone.zone.isin(grouping_members(g))]
        m = sub.groupby(["slack", "job"])[["defer", "interrupt_extra", "combined"]].mean().reset_index()
        m.insert(0, "grouping", g)
        out.append(m)
    return pd.concat(out, ignore_index=True), per_zone


def weighted_mix(per_zone: pd.DataFrame, slack: int, weights) -> pd.Series:
    """Per-zone combined saving under a job-length mix (Figure 10a-c)."""
    sub = per_zone[per_zone.slack == slack].pivot(index="zone", columns="job", values="combined")
    return (sub[list(JOB_LENGTHS)] * np.array(weights)).sum(axis=1) / sum(weights)


def provider_members(provider: str) -> list[str]:
    df = pd.read_csv(DATA_DIR / "provider_regions.csv", encoding="utf-8-sig")
    return df.loc[df[provider] == 1, "zone_code"].dropna().tolist()


def daily_cv(ci_df: pd.DataFrame) -> pd.Series:
    g = ci_df.groupby(ci_df.index.date)
    return (g.std() / g.mean()).mean()


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    P.setup()
    ci = load_carbon(YEAR)
    means = ci.mean()
    claims = []  # (section, claim, paper, ours, unit)

    def claim(sec, text, paper, ours, unit="g"):
        claims.append({"section": sec, "claim": text, "paper": paper, "ours": round(float(ours), 2), "unit": unit})

    # ---- Section 4: trace characteristics
    cv = daily_cv(ci)
    claim("§3.1.3", "Global average carbon intensity", 368.39, means.mean(), "g/kWh")
    claim("§4.1", "Share of zones with mean CI > 400", 46, (means > 400).mean() * 100, "%")
    claim("§4.1", "Max/min ratio of zone mean CI", 40, means.max() / means.min(), "x")
    claim("§1", "Share of zones with daily CV < 0.1 ('>70%')", 70, (cv < 0.1).mean() * 100, "%")
    claim("§5.1.1", "Greenest zone mean CI (Sweden)", 16, means.min(), "g/kWh")

    # ---- Section 5.2: temporal
    tab, per_zone = temporal(ci)
    tab.to_csv(RESULTS / "repro_temporal.csv", index=False)
    glob = tab[tab.grouping == "Global"].set_index(["slack", "job"])
    yr, day = 24 * 365, 24
    claim("Fig 7a", "Deferral saving, 1 h job, 1-year slack", 154, glob.loc[(yr, 1), "defer"])
    claim("Fig 7a", "Deferral saving, 168 h job, 1-year slack", 70, glob.loc[(yr, 168), "defer"])
    claim("Fig 7b", "Deferral saving, 1 h job, 24 h slack", 57, glob.loc[(day, 1), "defer"])
    claim("Fig 7b", "Deferral saving, 168 h job, 24 h slack", 3, glob.loc[(day, 168), "defer"])
    claim("Fig 8a", "Extra saving from interruption, 168 h job, 1-year slack", 43, glob.loc[(yr, 168), "interrupt_extra"])
    claim("Fig 8b", "Extra saving from interruption, 24 h job, 24 h slack (peak)", 18.4, glob.loc[(day, 24), "interrupt_extra"])

    m, _ = as_matrix(ci)
    rel = pd.DataFrame({(s, j): relative_reduction(m, j, s) for s in (yr, day) for j in JOB_LENGTHS}).T
    rel.index.names = ["slack", "job"]
    rel.to_csv(RESULTS / "repro_relative.csv")
    claim("Fig 9a", "Deferral, 168 h job, 1-year slack (% of baseline)", 19.0, rel.loc[(yr, 168), "defer_pct"], "%")
    claim("Fig 9a", "Interruption adds, 168 h job, 1-year slack", 11, rel.loc[(yr, 168), "interrupt_pct"], "%")
    claim("Fig 9b", "Defer+interrupt, 168 h job, 24 h slack", 3, rel.loc[(day, 168)].sum(), "%")

    mix = {}
    for name, w in JOB_WEIGHTS.items():
        s = weighted_mix(per_zone, yr, w)
        members = s.index if name == "equal" else s.index.intersection(provider_members(name))
        mix[name] = s.loc[members].mean()
    claim("Fig 10a", "Equal job mix, 1-year slack, global", 135, mix["equal"])
    claim("Fig 10b", "Azure job mix, 1-year slack", 100, mix["azure"])
    claim("Fig 10c", "Google job mix, 1-year slack", 112, mix["gcp"])
    vary = glob.groupby(level="slack")["combined"].mean()
    claim("Fig 10d", "Equal mix, 24 h slack", 31, vary.loc[day])
    claim("Fig 10d", "Equal mix, 1-year slack", 127, vary.loc[yr])

    # ---- Section 5.1: spatial
    sp = []
    for g in GEO_GROUPINGS:
        mem = grouping_members(g)
        r_inf, r_50 = grouping_reduction(means, mem, 0.99), grouping_reduction(means, mem, 0.5)
        sp.append({"grouping": g, "zones": len(mem), "mean_ci": means[mem].mean(),
                   "inf_artifact": r_inf["artifact"], "inf_corrected": r_inf["corrected"],
                   "idle50_artifact": r_50["artifact"], "idle50_corrected": r_50["corrected"],
                   **{f"migr_{k}": v for k, v in one_vs_inf_migration(ci, mem).items()}})
    sp = pd.DataFrame(sp).set_index("grouping")
    sp.to_csv(RESULTS / "repro_spatial.csv")
    claim("Fig 5a", "Global, infinite capacity", 352, sp.loc["Global", "inf_artifact"])
    claim("Fig 5a", "Europe, infinite capacity", 281, sp.loc["Europe", "inf_artifact"])
    claim("Fig 5a", "Asia, infinite capacity", 556, sp.loc["Asia", "inf_artifact"])
    claim("Fig 5b", "Global, 50% idle capacity", 190, sp.loc["Global", "idle50_artifact"])
    claim("Fig 5b", "Asia, 50% idle capacity (artifact metric)", 556, sp.loc["Asia", "idle50_artifact"])
    claim("Fig 6b", "Max extra from inf-migration over 1-migration (<10)", 10,
          (sp.migr_inf - sp.migr_one).max())

    idle = global_idle_curve(means, [i / 100 for i in range(0, 100, 10)] + [0.99])
    idle.to_csv(RESULTS / "repro_global_idle.csv")
    claim("Fig 5c", "Global reduction at 50% idle", 51.5, idle.loc[50, "artifact_pct"], "%")
    claim("Fig 5c", "Global reduction at 99% idle", 95.68, idle.loc[99, "artifact_pct"], "%")

    lat = pd.read_csv(DATA_DIR / "gcp_latency_matrix.csv", index_col="origin")
    cl = capacity_latency(means, lat, [5, 50, 100, 150, 200, 250, 300], [0, 0.5, 0.99])
    cl_pct = (1 - cl.div(cl[0], axis=0)) * 100
    cl_pct.to_csv(RESULTS / "repro_capacity_latency.csv")
    claim("Fig 6a", "250 ms SLO, infinite capacity", 92.5, cl_pct.loc[250, 99], "%")
    claim("Fig 6a", "250 ms SLO, 50% utilisation", 45.7, cl_pct.loc[250, 50], "%")
    claim("Fig 6a", "50 ms SLO, 50% utilisation", 31, cl_pct.loc[50, 50], "%")

    cmp = pd.DataFrame(claims)
    cmp["abs_diff"] = (cmp.ours - cmp.paper).abs().round(2)
    cmp.to_csv(RESULTS / "repro_vs_paper.csv", index=False)
    print(cmp.to_string(index=False))

    figures(glob, vary, idle, cl_pct, sp)


def figures(glob, vary, idle, cl_pct, sp) -> None:
    import matplotlib.pyplot as plt

    yr, day = 24 * 365, 24
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    x = np.arange(len(JOB_LENGTHS))
    for ax, slack, title in ((axes[0], yr, "One-year slack (ideal)"), (axes[1], day, "24-hour slack (practical)")):
        d = glob.loc[slack]
        ax.bar(x, d.defer, 0.62, color=P.BLUE, label="Deferral")
        ax.bar(x, d.interrupt_extra, 0.62, bottom=d.defer, color=P.ORANGE, label="+ Interruption",
               edgecolor="white", linewidth=1)
        ax.set_xticks(x, [f"{j}" for j in JOB_LENGTHS])
        ax.set_xlabel("Job length (hours)")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Carbon reduction\n(g·CO$_2$eq per job-hour)")
    axes[1].legend(loc="upper right")
    P.save(fig, "repro_temporal")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].plot(idle.index, idle.artifact_pct, "o-", color=P.BLUE)
    axes[0].set(xlabel="Global idle capacity (%)", ylabel="Global carbon reduction (%)", ylim=(0, 100),
                title="Spatial: capacity limits (Fig. 5c)")
    for k, (col, lab) in enumerate(((99, "Infinite capacity"), (50, "50% utilisation"))):
        axes[1].plot(cl_pct.index, cl_pct[col], "o-", color=(P.BLUE, P.ORANGE)[k], label=lab)
    axes[1].set(xlabel="Latency SLO (ms)", ylabel="Global carbon reduction (%)", ylim=(0, 100),
                title="Spatial: latency + capacity (Fig. 6a)")
    axes[1].legend(loc="upper left")
    P.save(fig, "repro_spatial")


if __name__ == "__main__":
    main()
