"""F35: who produced the saving? Euler attribution between flexibility and headroom.

For every zone (Electricity Maps, 123 zones x 2020-2022) and US balancing authority
(EIA-930 2022, average and marginal), with phi = 0.5 and 24 h slack, split the capped
shifting saving S(phi, h) into the marginal products of flexible load and headroom
(attribution.euler_*), three ways:

* LP (facility.capped_shift, sliding 24 h windows): S_h from the cap duals, S_phi by
  finite differences; S_h re-derived by a finite difference in h (independent check);
* daily water-filling, analytic order statistics (closed form, calendar days);
* the sinusoidal closed form sigma_h = (1 - q)(1 - pi q cot pi q).

Slack 6 h and 168 h are run for 2022 at h = 0.2 and 0.42.

Outputs: results/attribution_{zones,bas,slack,summary}.csv,
figures/attribution_shares.png, paper/sigmetrics/fig_attribution.pdf.
`--plot-only` recomputes the summary and figures from the CSVs.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift import attribution as at
from carbonshift.data import load_carbon
from carbonshift.facility import capped_shift

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
PHI = 0.5
HS = (0.05, 0.2, 0.3, 0.42, 0.6, 0.9)   # 24 phi / (phi + h) never an integer


def rows_for(ci, label: dict, slack=24, hs=HS, check_fd=True):
    ci = np.asarray(ci, dtype=float)
    out = []
    for h in hs:
        lp = at.euler_lp(ci, PHI, h, slack)
        row = dict(label, slack=slack, h=h, S_lp=lp["S"], S_phi_lp=lp["S_phi"], S_h_lp=lp["S_h"],
                   resid_lp=lp["residual"], sigma_lp=lp["sigma_h"],
                   sigma_sinc=float(at.capacity_share_sinc(PHI, h)))
        if check_fd and h in (0.2, 0.42):
            e = 1e-3
            sp = capped_shift(ci, PHI, 1 + h + e, slack)
            sm = capped_shift(ci, PHI, 1 + h - e, slack)
            row["S_h_lp_fd"] = ((sp.baseline_g - sp.optimised_g) - (sm.baseline_g - sm.optimised_g)) / (2 * e)
        if slack == 24:
            w = at.euler_waterfill(ci, PHI, h)
            row.update(S_wf=w["S"], S_phi_wf=w["S_phi"], S_h_wf=w["S_h"], sigma_wf=w["sigma_h"])
        out.append(row)
    return out


def _zone_job(args):
    year, z, slack, hs, fd = args
    ci = load_carbon(year)[z].to_numpy()
    if np.ptp(ci) < 1e-9:
        return []
    return rows_for(ci, dict(year=year, zone=z), slack=slack, hs=hs, check_fd=fd)


def _ba_job(args):
    ba, s, ci = args
    return rows_for(ci, dict(ba=ba, signal=s), hs=(0.2, 0.42), check_fd=False)


def run(workers=4):
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(y, z, 24, HS, True) for y in (2020, 2021, 2022) for z in load_carbon(y).columns]
    sjobs = [(2022, z, s, (0.2, 0.42), False) for z in load_carbon(2022).columns for s in (6, 168)]
    sig = pd.read_csv(ROOT / "data" / "eia930" / "signals_2022.csv.gz")
    bjobs = []
    for ba, g in sig.groupby("ba"):
        g = g.sort_values("utc")
        for s in ("average", "marginal"):
            ci = g[s].to_numpy()
            if len(ci) >= 24 * 300 and np.all(np.isfinite(ci)) and np.ptp(ci) > 1e-9:
                bjobs.append((ba, s, ci))
    with ProcessPoolExecutor(workers) as ex:
        Z = pd.DataFrame([r for rs in ex.map(_zone_job, jobs, chunksize=4) for r in rs])
        Z.to_csv(OUT / "attribution_zones.csv", index=False)
        print("zones done", flush=True)
        pd.DataFrame([r for rs in ex.map(_ba_job, bjobs) for r in rs]).to_csv(OUT / "attribution_bas.csv", index=False)
        print("bas done", flush=True)
        pd.DataFrame([r for rs in ex.map(_zone_job, sjobs, chunksize=4) for r in rs]).to_csv(
            OUT / "attribution_slack.csv", index=False)
        print("slack done", flush=True)


def shape_shares(year):
    """Per zone: share of hourly variance that is the average daily shape (clock-like part)."""
    df = load_carbon(year)
    out = {}
    for z in df.columns:
        c = df[z].to_numpy()
        c = c[: len(c) // 24 * 24].reshape(-1, 24)
        tot = c.var()
        if tot <= 0:
            continue
        out[z] = dict(diurnal_share=c.mean(axis=0).var() / tot, daily_std=c.std(axis=1).mean())
    return pd.DataFrame(out).T


def summarise():
    Z = pd.read_csv(OUT / "attribution_zones.csv")
    S = pd.read_csv(OUT / "attribution_slack.csv")
    B = pd.read_csv(OUT / "attribution_bas.csv")
    s = {}
    Z = Z[Z.S_lp > 1e-6]
    for (year, h), g in Z.groupby(["year", "h"]):
        p = f"zones{year}_h{h}"
        s[f"{p}_n"] = len(g)
        s[f"{p}_sigma_lp_median"] = g.sigma_lp.median()
        s[f"{p}_sigma_lp_p10"] = g.sigma_lp.quantile(0.1)
        s[f"{p}_sigma_lp_p90"] = g.sigma_lp.quantile(0.9)
        s[f"{p}_sigma_wf_median"] = g.sigma_wf.median()
        s[f"{p}_sigma_sinc"] = g.sigma_sinc.iloc[0]
        s[f"{p}_overclaim_lp_median"] = float(np.median(at.overclaim_factor(g.sigma_lp)))
        s[f"{p}_marginal_over_average_flex_median"] = (g.S_phi_lp / (g.S_lp / PHI)).median()
        s[f"{p}_max_rel_euler_residual_lp"] = (g.resid_lp.abs() / g.S_lp).max()
        s[f"{p}_median_rel_dual_vs_fd_S_h"] = ((g.S_h_lp - g.S_h_lp_fd).abs() / g.S_h_lp_fd).median()
        s[f"{p}_share_capacity_majority"] = float((g.sigma_lp > 0.5).mean())
    # What sets the split: how much of the signal's variance is the average daily shape
    from scipy.stats import spearmanr
    for year in (2020, 2021, 2022):
        sh = shape_shares(year)
        for h in (0.2, 0.42):
            g = Z[(Z.year == year) & (Z.h == h)].set_index("zone").join(sh, how="inner")
            s[f"zones{year}_h{h}_spearman_sigma_vs_diurnal_share"] = spearmanr(g.sigma_lp, g.diurnal_share)[0]
            s[f"zones{year}_h{h}_spearman_sigma_vs_mean_daily_std"] = spearmanr(g.sigma_lp, g.daily_std)[0]
            reg = g.diurnal_share >= g.diurnal_share.quantile(0.75)
            irr = g.diurnal_share <= g.diurnal_share.quantile(0.25)
            s[f"zones{year}_h{h}_sigma_median_top_quartile_diurnal"] = g.sigma_lp[reg].median()
            s[f"zones{year}_h{h}_sigma_median_bottom_quartile_diurnal"] = g.sigma_lp[irr].median()
    for (slack, h), g in S[S.S_lp > 1e-6].groupby(["slack", "h"]):
        s[f"slack{slack}_h{h}_sigma_lp_median"] = g.sigma_lp.median()
        s[f"slack{slack}_h{h}_median_rel_euler_residual"] = (g.resid_lp.abs() / g.S_lp).median()
        s[f"slack{slack}_h{h}_max_rel_euler_residual"] = (g.resid_lp.abs() / g.S_lp).max()
    for (sig_, h), g in B[B.S_lp > 1e-6].groupby(["signal", "h"]):
        s[f"ba_{sig_}_h{h}_n"] = len(g)
        s[f"ba_{sig_}_h{h}_sigma_lp_median"] = g.sigma_lp.median()
        s[f"ba_{sig_}_h{h}_sigma_wf_median"] = g.sigma_wf.median()
    pd.Series(s).to_csv(OUT / "attribution_summary.csv", header=["value"])
    for k, v in s.items():
        print(f"{k:60s} {v:.4g}")
    return Z


def plot(Z):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = Z[Z.year == 2022]
    hh = np.linspace(0.01, 3, 300)
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    x = hh / PHI
    ax.plot(x, at.capacity_share_sinc(PHI, hh), color="#2a6f97", lw=1.6, label="sinusoidal closed form")
    med = g.groupby("h").sigma_lp.agg(["median", lambda v: v.quantile(0.1), lambda v: v.quantile(0.9)])
    med.columns = ["med", "lo", "hi"]
    hx = med.index.to_numpy() / PHI
    ax.fill_between(hx, med.lo, med.hi, color="#e07a5f", alpha=0.25, lw=0, label="LP, 120 zones, 10-90%")
    ax.plot(hx, med.med, "o-", color="#e07a5f", ms=3.5, lw=1.2, label="LP, median zone (2022)")
    ax.axvline(0.42 / PHI, color="0.5", lw=0.8, ls=":")
    ax.text(0.42 / PHI + 0.04, 0.93, "Google-real\n+42%", fontsize=7, color="0.35", va="top")
    ax.axhline(0.5, color="0.8", lw=0.6)
    ax.set_xlabel("headroom per unit of flexible load, h/phi")
    ax.set_ylabel("capacity share of the saving")
    ax.set_xlim(0, 3 / PHI * 0.5 + 0.2)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.8))
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "attribution_shares.png", dpi=200)
    fig.savefig(ROOT / "paper" / "sigmetrics" / "fig_attribution.pdf")
    print("figures written")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run()
    plot(summarise())
