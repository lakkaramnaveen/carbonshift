"""Smooth knobs vs deferral: carbon-aware speed scaling is a Harberger triangle (F29).

For 123 zones x 2020-2022 (average signal, 24 h windows) compare
  * carbon-aware speed scaling (DVFS / GPU power limits), P = s**alpha, alpha in {1.5, 2, 3},
    uncapped and under a power cap of +20% / +50% (s_max = cap**(1/alpha));
  * deferral of the same work on an energy-proportional server (water-filling, phi = 1)
    under the same power caps.
Checks: exact solver vs the power-mean closed form (uncapped) and vs the Harberger
law (kappa/2) CV**2; log-log slope of saving against within-day CV (2 = second order,
1 = first order); and the same slope for the independent F14 batching model
(results/llm_batching_per_zone.csv), which uses measured GPU curves.

Outputs: results/smoothknob_per_zone.csv, results/smoothknob_summary.csv,
         results/smoothknob_f14_slopes.csv, figures/smoothknob.png
Run: python scripts/smoothknob_experiments.py   (~1 min)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift import plotting as P
from carbonshift.data import load_carbon
from carbonshift.smoothknob import (loglog_slope, saving_harberger, saving_power_mean,
                                    saving_speed_scaling, within_window_cv)
from carbonshift.waterfill import saving_waterfill

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
ALPHAS = (1.5, 2.0, 3.0)
CAPS = (1.2, 1.5, np.inf)


def r2(pred, obs) -> float:
    pred, obs = np.asarray(pred), np.asarray(obs)
    return float(1 - ((pred - obs) ** 2).sum() / ((obs - obs.mean()) ** 2).sum())


def cap_label(cap) -> str:
    return "uncapped" if np.isinf(cap) else f"+{round((cap - 1) * 100)}%"


def per_zone() -> pd.DataFrame:
    rows = []
    for year in (2020, 2021, 2022):
        ci = load_carbon(year)
        for z in ci.columns:
            c = ci[z].to_numpy(dtype=float)
            if np.nanmean(c) <= 0 or np.isnan(c).any():
                continue
            mean = c.mean()
            row = dict(zone=z, year=year, mean=mean, cv=within_window_cv(c))
            for cap in CAPS:
                lab = cap_label(cap)
                row[f"defer_{lab}"] = saving_waterfill(c, 1.0, cap) / mean
                for a in ALPHAS:
                    smax = np.inf if np.isinf(cap) else cap ** (1 / a)
                    row[f"dvfs_a{a}_{lab}"] = saving_speed_scaling(c, a, s_max=smax)
                    # same *speed* headroom as deferral (power cap cap**alpha)
                    if not np.isinf(cap):
                        row[f"dvfs_speedcap_a{a}_{lab}"] = saving_speed_scaling(c, a, s_max=cap)
            for a in ALPHAS:
                row[f"powermean_a{a}"] = saving_power_mean(c, a)
                row[f"harberger_a{a}"] = saving_harberger(c, a)
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for cap in CAPS:
        lab = cap_label(cap)
        d = df[f"defer_{lab}"]
        out.append(dict(knob="deferral (phi=1, energy-proportional)", alpha=1.0, power_cap=lab,
                        median_saving_pct=100 * d.median(), pooled_saving_pct=100 * (d * df["mean"]).sum() / df["mean"].sum(),
                        loglog_slope_vs_cv=loglog_slope(df.cv, d), median_ratio_defer_over_knob=1.0))
        for a in ALPHAS:
            s = df[f"dvfs_a{a}_{lab}"]
            row = dict(knob="speed scaling (same power cap)", alpha=a, power_cap=lab,
                       median_saving_pct=100 * s.median(), pooled_saving_pct=100 * (s * df["mean"]).sum() / df["mean"].sum(),
                       loglog_slope_vs_cv=loglog_slope(df.cv, s),
                       median_ratio_defer_over_knob=float((d / s.where(s > 0)).median()))
            if np.isinf(cap):
                row["r2_powermean_vs_exact"] = r2(df[f"powermean_a{a}"], s)
                row["max_abs_diff_powermean_pp"] = float(100 * (df[f"powermean_a{a}"] - s).abs().max())
                row["r2_harberger_vs_exact"] = r2(df[f"harberger_a{a}"], s)
                row["median_harberger_over_exact"] = float((df[f"harberger_a{a}"] / s).median())
            out.append(row)
            if not np.isinf(cap):
                s2 = df[f"dvfs_speedcap_a{a}_{lab}"]
                out.append(dict(knob="speed scaling (same speed headroom)", alpha=a, power_cap=f"speed {lab}",
                                median_saving_pct=100 * s2.median(),
                                pooled_saving_pct=100 * (s2 * df["mean"]).sum() / df["mean"].sum(),
                                loglog_slope_vs_cv=loglog_slope(df.cv, s2),
                                median_ratio_defer_over_knob=float((d / s2.where(s2 > 0)).median())))
    return pd.DataFrame(out)


def f14_slopes() -> pd.DataFrame:
    """Second-order test on the independent F14 model (measured GPU batch curves, yearly budget)."""
    llm = pd.read_csv(RESULTS / "llm_batching_per_zone.csv")
    ci = load_carbon(2022)
    stats = {}
    for z in llm.zone.unique():
        c = ci[z].to_numpy(dtype=float)
        stats[z] = dict(cv_day=within_window_cv(c), cv_year=c.std() / c.mean())
    st = pd.DataFrame(stats).T
    rows = []
    for (m, b), g in llm.groupby(["model", "budget_ms"]):
        gain = 1 - g.aware.to_numpy() / g.static.to_numpy()
        z = g.zone.to_numpy()
        rows.append(dict(model=m, budget_ms=b, n=len(g),
                         slope_vs_cv_year=loglog_slope(st.loc[z, "cv_year"], gain),
                         slope_vs_cv_day=loglog_slope(st.loc[z, "cv_day"], gain),
                         median_gain_pct=100 * np.median(gain)))
    return pd.DataFrame(rows)


def figure(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    P.setup()
    d = df[(df.year == 2022) & (df.cv > 1e-3)]  # drop constant-signal zones (CV ~ 0)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.scatter(d.cv, 100 * d["defer_+20%"], s=8, label="deferral, +20% cap", color=P.BLUE)
    ax.scatter(d.cv, 100 * d["dvfs_a3.0_+20%"], s=8, label="speed scaling α=3, +20% cap", color=P.ORANGE)
    ax.scatter(d.cv, 100 * d["dvfs_a3.0_uncapped"], s=8, marker="x", label="speed scaling α=3, uncapped", color=P.AQUA)
    x = np.linspace(0.01, d.cv.max(), 100)
    ax.plot(x, 100 * 0.75 * x ** 2, "k--", lw=0.8, label="law (χ/2)·CV², χ = 1.5")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("within-day CV of carbon intensity")
    ax.set_ylabel("carbon saving (%)")
    ax.legend(fontsize=6, frameon=False)
    P.save(fig, "smoothknob")


def main() -> None:
    df = per_zone()
    df.to_csv(RESULTS / "smoothknob_per_zone.csv", index=False)
    s = summarise(df)
    s.to_csv(RESULTS / "smoothknob_summary.csv", index=False)
    f = f14_slopes()
    f.to_csv(RESULTS / "smoothknob_f14_slopes.csv", index=False)
    figure(df)
    pd.set_option("display.width", 220)
    print(s.round(3).to_string())
    print(f.round(3).to_string())


if __name__ == "__main__":
    main()
