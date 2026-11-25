"""F33: the idle-power wall. Does capacity built for carbon-aware shifting pay for its idle draw?

For every zone (Electricity Maps average signal, 123 zones x 2020-2022) and every US
balancing authority (EIA-930 2022, average and marginal signals), with phi = 0.5 and
24 h slack:

* break-even idle fraction p0* (largest idle/full-load power ratio at which *some*
  always-on build h pays), exact from the LP over h in HS, plus the daily closed form
  D / (D + cbar) and the horizon-free bound 1 - cbar / max;
* the share of zones where always-on spare capacity built for shifting raises
  emissions at the SPECpower x86 idle fractions (15.7% = 2017 best year, 25.7% = 2024
  mean) and at 5% / 10%;
* for capacity that already exists (Google-real +42%, F12): shift only (servers stay on)
  versus sleep only (power the spare down, no shifting) versus both.

Outputs: results/idle_{zones,bas,summary}.csv, figures/idle_wall.png,
paper/sigmetrics/fig_idle_wall.pdf. `--plot-only` redraws from the CSVs.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift import idle
from carbonshift.data import load_carbon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
PHI, H_REAL = 0.5, 0.42
HS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.42, 1.0)
P0S = (0.05, 0.10, 0.157, 0.257)


def zone_row(ci, phi=PHI):
    ci = np.asarray(ci, dtype=float)
    cbar = float(ci.mean())
    S = {h: idle.capped_saving(ci, phi, h) for h in HS}
    v = {h: S[h] / h for h in HS}
    pstar = max(v[h] / (v[h] + cbar) for h in HS)
    hstar = max(HS, key=lambda h: v[h])
    row = dict(cbar=cbar, D=idle.first_unit_value(ci), p0_star_lp=pstar, h_at_p0_star=hstar,
               p0_star_daily=idle.breakeven_idle(ci), p0_bound=idle.breakeven_bound(ci))
    for h in HS:
        row[f"S_h{h}"] = S[h]
    for p0 in P0S:
        nets = {h: (1 - p0) * S[h] - p0 * h * cbar for h in HS}
        hb = max(HS, key=lambda h: nets[h])
        row[f"net_best_p{p0}"] = max(0.0, nets[hb])
        row[f"hopt_p{p0}"] = hb if nets[hb] > 0 else 0.0
        row[f"net_h042_pct_p{p0}"] = 100 * nets[H_REAL] / (cbar * (1 + p0 * H_REAL))
        sp = S[H_REAL]
        base = cbar * (1 + p0 * H_REAL)
        row[f"shift_only_pct_p{p0}"] = 100 * (1 - p0) * sp / base
        row[f"sleep_only_pct_p{p0}"] = 100 * p0 * H_REAL * cbar / base
        row[f"sleep_shift_pct_p{p0}"] = 100 * (p0 * H_REAL * cbar + sp) / base
    return row


def main():
    rows = []
    for year in (2020, 2021, 2022):
        df = load_carbon(year)
        for z in df.columns:
            r = zone_row(df[z].to_numpy())
            rows.append(dict(year=year, zone=z, **r))
        print(year, "done", flush=True)
    Z = pd.DataFrame(rows)
    Z.to_csv(OUT / "idle_zones.csv", index=False)

    sig = pd.read_csv(ROOT / "data" / "eia930" / "signals_2022.csv.gz")
    brows = []
    for ba, g in sig.groupby("ba"):
        g = g.sort_values("utc")
        for s in ("average", "marginal"):
            ci = g[s].to_numpy()
            if len(ci) < 24 * 300 or np.any(~np.isfinite(ci)) or ci.mean() <= 0:
                continue
            brows.append(dict(ba=ba, signal=s, **zone_row(ci)))
    B = pd.DataFrame(brows)
    B.to_csv(OUT / "idle_bas.csv", index=False)
    summarise(Z, B)
    plot(Z, B)


def summarise(Z, B):
    s = {}
    for year, g in Z.groupby("year"):
        s[f"zones_{year}"] = len(g)
        s[f"p0_star_median_{year}"] = g.p0_star_lp.median()
        s[f"p0_star_p90_{year}"] = g.p0_star_lp.quantile(0.9)
        s[f"p0_star_max_{year}"] = g.p0_star_lp.max()
        s[f"p0_star_daily_median_{year}"] = g.p0_star_daily.median()
        s[f"p0_bound_median_{year}"] = g.p0_bound.median()
        for p0 in P0S:
            s[f"share_zones_build_raises_p{p0}_{year}"] = float((g.p0_star_lp < p0).mean())
            s[f"share_zones_h042_raises_p{p0}_{year}"] = float((g[f"net_h042_pct_p{p0}"] < 0).mean())
            s[f"median_net_h042_pct_p{p0}_{year}"] = g[f"net_h042_pct_p{p0}"].median()
            s[f"median_shift_only_pct_p{p0}_{year}"] = g[f"shift_only_pct_p{p0}"].median()
            s[f"median_sleep_only_pct_p{p0}_{year}"] = g[f"sleep_only_pct_p{p0}"].median()
            s[f"share_sleep_beats_shift_p{p0}_{year}"] = float((g[f"sleep_only_pct_p{p0}"] > g[f"shift_only_pct_p{p0}"]).mean())
        s[f"spearman_lp_vs_daily_{year}"] = g.p0_star_lp.corr(g.p0_star_daily, method="spearman")
        s[f"max_lp_minus_bound_{year}"] = float((g.p0_star_lp - g.p0_bound).max())
    for sg, g in B.groupby("signal"):
        s[f"ba_{sg}_n"] = len(g)
        s[f"ba_{sg}_p0_star_median"] = g.p0_star_lp.median()
        for p0 in P0S:
            s[f"ba_{sg}_share_build_raises_p{p0}"] = float((g.p0_star_lp < p0).mean())
            s[f"ba_{sg}_share_sleep_beats_shift_p{p0}"] = float((g[f"sleep_only_pct_p{p0}"] > g[f"shift_only_pct_p{p0}"]).mean())
    for name, g in (("zones2022", Z[Z.year == 2022]), ("ba_average", B[B.signal == "average"]),
                    ("ba_marginal", B[B.signal == "marginal"])):
        for h in HS:
            p = pstar_at(g, h)
            s[f"{name}_p0_star_h{h}_median"] = p.median()
            s[f"{name}_p0_star_h{h}_share_below_0.257"] = float((p < 0.257).mean())
            s[f"{name}_p0_star_h{h}_share_below_0.157"] = float((p < 0.157).mean())
    pd.Series(s).to_csv(OUT / "idle_summary.csv", header=["value"])
    for k, v in s.items():
        print(f"{k:55s} {v:.4g}")


def pstar_at(g, h):
    """Break-even idle share for a build of exactly h: v / (v + cbar), v = S(h) / h."""
    v = g[f"S_h{h}"] / h
    return v / (v + g.cbar)


def plot(Z, B):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    hs = np.array(HS)
    for g, lab, col in ((Z[Z.year == 2022], "123 zones, average signal (2022)", "#2a6fbb"),
                        (B[B.signal == "marginal"], "53 US BAs, marginal signal", "#e08a1e")):
        P = np.array([pstar_at(g, h).to_numpy() for h in HS])
        ax.fill_between(100 * hs, 100 * np.quantile(P, 0.25, axis=1), 100 * np.quantile(P, 0.75, axis=1),
                        color=col, alpha=0.18, lw=0)
        ax.plot(100 * hs, 100 * np.median(P, axis=1), color=col, lw=1.6, marker="o", ms=2.5, label=lab)
    for p0, lab in ((0.257, "SPECpower x86 mean, 2024"), (0.157, "SPECpower x86 mean, 2017 (best)")):
        ax.axhline(100 * p0, color="#c0392b", lw=0.9, ls="--")
        ax.text(1.05, 100 * p0 + 0.7, f"{lab}: {100 * p0:.1f}%", fontsize=5.5, color="#c0392b")
    ax.axvline(100 * H_REAL, color="grey", lw=0.6, ls=":")
    ax.text(100 * H_REAL * 1.05, 36, "Google-real\n+42%", fontsize=5.5, color="grey")
    ax.set_xscale("log")
    ax.set_xlabel("always-on headroom built for shifting, h (% of mean load)", fontsize=6.5)
    ax.set_ylabel("break-even idle share $p_0^*$ (%)", fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.5, loc="lower left", frameon=False)
    ax.set_ylim(0, 42)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "idle_wall.png", dpi=200)
    fig.savefig(ROOT / "paper" / "sigmetrics" / "fig_idle_wall.pdf")
    print("figure written")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        Z, B = pd.read_csv(OUT / "idle_zones.csv"), pd.read_csv(OUT / "idle_bas.csv")
        summarise(Z, B)
        plot(Z, B)
    else:
        main()
