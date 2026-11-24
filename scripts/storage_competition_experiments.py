"""F32: grid batteries compete with carbon-aware shifting (CAISO 2021-2024).

For each year, a new data-centre fleet (mean load P, phi = 0.5, cap 1/0.70, 24 h slack)
is added to CAISO with the F24 model (curtailment is free, otherwise a 490 g/kWh gas
margin). We then add extra grid storage beyond the fleet already behind the observed
curtailment, dispatched to minimise emissions, and measure the *real* saving of
carbon-aware shifting (flat vs shifted fleet, storage re-optimised in both).

Storage cases: 1, 2, 10, 20 GW of 4 h storage, and CAISO's 2025 additions
(4.7 GW / 16.5 GWh, Modo Energy Q4-2025 fleet report). eta = 0.85 round trip.
Second method: greedy closed form (`greedy_saving`), no LP.
Third check: unlimited storage, saving / saving(no storage) vs 1 - eta.

Outputs: results/storage_competition_{grid,eta,summary}.csv, figures/storage_competition.png,\npaper/sigmetrics/fig_storage_competition.pdf. `--plot-only` redraws from the CSVs.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift.curtailment import caiso_hourly
from carbonshift import storage_competition as sc

OUT = Path(__file__).resolve().parents[1] / "results"
PHI, CAP, ETA = 0.5, 1 / 0.70, 0.85
CASES = [("none", 0, 0), ("1GW_4h", 1000, 4000), ("2GW_4h", 2000, 8000),
         ("caiso_2025_adds", 4700, 16500), ("10GW_4h", 10000, 40000), ("20GW_4h", 20000, 80000)]


def main():
    rows, erows = [], []
    for year in (2021, 2022, 2023, 2024):
        S, G = sc.caiso_days(caiso_hourly(year))
        for P in (100, 1000):
            for name, pb, eb in CASES:
                lp = sc.shifting_saving(S, G, P, PHI, CAP, pb, eb, ETA)
                gr = sc.greedy_saving(S, G, P, PHI, CAP, pb, eb, ETA)
                rows.append(dict(year=year, fleet_mw=P, storage=name, storage_mw=pb, storage_mwh=eb,
                                 saving_g_per_flex_kwh=lp, greedy_g_per_flex_kwh=gr))
                print(year, P, name, round(lp, 2), round(gr, 2), flush=True)
        if year == 2024:
            v0 = [r for r in rows if r["year"] == 2024 and r["fleet_mw"] == 100 and r["storage"] == "none"][0]["saving_g_per_flex_kwh"]
            for eta in (0.7, 0.8, 0.85, 0.9, 0.95, 1.0):
                v = sc.shifting_saving(S, G, 100, PHI, CAP, 1e5, 1e6, eta)
                erows.append(dict(year=year, fleet_mw=100, eta=eta, saving_unlimited=v, saving_none=v0,
                                  ratio=v / v0, one_minus_eta=1 - eta))
    df = pd.DataFrame(rows)
    base = df[df.storage == "none"].set_index(["year", "fleet_mw"]).saving_g_per_flex_kwh
    df["share_of_no_storage"] = [r.saving_g_per_flex_kwh / base[(r.year, r.fleet_mw)] for r in df.itertuples()]
    df.to_csv(OUT / "storage_competition_grid.csv", index=False)
    e = pd.DataFrame(erows)
    e.to_csv(OUT / "storage_competition_eta.csv", index=False)
    s = {}
    for (y, P), g in df.groupby(["year", "fleet_mw"]):
        g = g.set_index("storage")
        for k in ("caiso_2025_adds", "20GW_4h"):
            s[f"share_kept_{k}_{y}_P{P}"] = g.loc[k, "share_of_no_storage"]
        s[f"saving_none_{y}_P{P}"] = g.loc["none", "saving_g_per_flex_kwh"]
    s["greedy_minus_lp_max_abs"] = float((df.greedy_g_per_flex_kwh - df.saving_g_per_flex_kwh).abs().max())
    s["greedy_over_lp_min"] = float((df.greedy_g_per_flex_kwh / df.saving_g_per_flex_kwh).min())
    s["eta_ratio_minus_one_minus_eta_max"] = float((e.ratio - e.one_minus_eta).max())
    pd.Series(s).to_csv(OUT / "storage_competition_summary.csv", header=["value"])
    print(pd.Series(s).round(3).to_string())
    print(e.round(3).to_string())


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(OUT / "storage_competition_grid.csv")
    e = pd.read_csv(OUT / "storage_competition_eta.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for (P, y), g in df.groupby(["fleet_mw", "year"]):
        if y not in (2022, 2024):
            continue
        ax[0].plot(g.storage_mw / 1000, g.share_of_no_storage, marker="o", ls="-" if P == 100 else "--",
                   label=f"{y}, {P/1000:g} GW fleet")
    ax[0].axhline(1 - ETA, color="grey", lw=0.8, ls=":")
    ax[0].text(13, 1 - ETA - 0.07, "1 - eta (theorem floor)", fontsize=8, color="grey")
    ax[0].axvline(4.7, color="grey", lw=0.8)
    ax[0].text(4.9, 0.9, "CAISO 2025\nadditions", fontsize=8, color="grey")
    ax[0].set_xlabel("extra grid storage (GW, ~4 h)")
    ax[0].set_ylabel("real saving kept (vs no extra storage)")
    ax[0].set_ylim(0, 1.05)
    ax[0].legend(fontsize=7, frameon=False)
    ax[1].plot(e.one_minus_eta, e.ratio, marker="o", label="LP, unlimited storage (2024)")
    ax[1].plot([0, 0.32], [0, 0.32], color="grey", lw=0.8, ls=":", label="theorem: 1 - eta")
    ax[1].set_xlabel("battery round-trip loss, 1 - eta")
    ax[1].set_ylabel("saving kept")
    ax[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(Path(__file__).resolve().parents[1] / "figures" / "storage_competition.png", dpi=200)
    fig.savefig(Path(__file__).resolve().parents[1] / "paper" / "sigmetrics" / "fig_storage_competition.pdf")


if __name__ == "__main__":
    import sys
    if "--plot-only" not in sys.argv:
        main()
    plot()
