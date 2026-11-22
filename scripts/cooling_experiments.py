"""The cooling penalty of carbon-aware scheduling (F26).

47 cloud-hosting zones (Electricity Maps 2022) with hourly weather at one
data-centre hub each (Open-Meteo / ERA5, UTC) and two cooling archetypes
(carbonshift.cooling.PLANTS). Facility: flat IT load, flexible share phi in
{0.25, 0.5}, slack 24 h, IT headroom h in {+20%, +50%, uncapped}.

Schedulers:
  carbon_only    optimise on c(t), what carbon-aware systems do today
  cooling_aware  optimise on c(t) * PUE(t), the carbon cost of an IT kWh

For each: the true saving (-cov(c*PUE, d)), the saving an operator would claim
(annual PUE x carbon-only saving) and the cooling penalty (claimed - true).
Diagnostic: correlation of the mean daily profiles of c and PUE ("are clean
hours hot?").

Outputs: results/cooling_*.csv, figures/cooling.png
Run: python scripts/cooling_experiments.py   (~1 min)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift import plotting as P
from carbonshift.cooling import HUBS, PLANTS, decompose, load_weather, penalty_law, plant_pue
from carbonshift.data import load_carbon
from carbonshift.facility import capped_shift

RESULTS = Path(__file__).resolve().parents[1] / "results"
PHIS, HEADS, SLACK = (0.25, 0.5), (0.2, 0.5, np.inf), 24


def _profile_corr(a: np.ndarray, b: np.ndarray) -> float:
    pa, pb = a.reshape(-1, 24).mean(0), b.reshape(-1, 24).mean(0)
    if pa.std() < 1e-9 or pb.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(pa, pb)[0, 1])


def _run(args):
    zone, ci, pues = args
    rows = []
    for plant, p in pues.items():
        diag = {"zone": zone, "hub": HUBS[zone][0], "plant": plant, "pue_mean": float(p.mean()),
                "pue_daily_swing": float(np.ptp(p.reshape(-1, 24).mean(0))),
                "corr_profile_c_pue": _profile_corr(ci, p), "ci_mean": float(ci.mean()),
                "penalty_law_pct": penalty_law(ci, p) * 100}
        for phi in PHIS:
            for h in HEADS:
                cap = 1 + h
                out = {}
                for name, sig in (("carbon_only", ci), ("cooling_aware", ci * p)):
                    r = capped_shift(sig, phi, cap, SLACK)
                    d = (1 - phi) + r.served
                    out[name] = decompose(ci, p, d)
                rows.append({**diag, "phi": phi, "head": h,
                             "claimed_g": out["carbon_only"]["claimed"],
                             "true_carbon_only_g": out["carbon_only"]["true"],
                             "true_cooling_aware_g": out["cooling_aware"]["true"],
                             "penalty_g": out["carbon_only"]["penalty"],
                             "baseline_true_g": float((ci * p).mean())})
    return rows


def main() -> None:
    ci = load_carbon(2022)
    w = load_weather(2022)
    assert len(w) == len(ci) == 8760
    jobs = [(z, ci[z].to_numpy(), {k: plant_pue(w, z, k) for k in PLANTS}) for z in HUBS]
    with ProcessPoolExecutor() as ex:
        df = pd.DataFrame([r for rows in ex.map(_run, jobs) for r in rows])
    df["penalty_pct_of_claim"] = np.where(df.claimed_g > 0.5, df.penalty_g / df.claimed_g * 100, np.nan)
    df["recovered_pct"] = np.where(df.true_carbon_only_g > 0.5,
                                   (df.true_cooling_aware_g / df.true_carbon_only_g - 1) * 100, np.nan)
    df.to_csv(RESULTS / "cooling_per_zone.csv", index=False)

    s = {"zones": df.zone.nunique()}
    for plant in PLANTS:
        g = df[df.plant == plant]
        z = g.drop_duplicates("zone")
        s[f"{plant}_pue_mean_median"] = z.pue_mean.median()
        s[f"{plant}_pue_daily_swing_median"] = z.pue_daily_swing.median()
        s[f"{plant}_share_zones_clean_hours_hot"] = float((z.corr_profile_c_pue < -0.3).mean())
        for phi in PHIS:
            for h in HEADS:
                x = g[(g.phi == phi) & (g["head"] == h)]
                key = f"{plant}_phi{phi}_h{'inf' if np.isinf(h) else h}"
                ok = x.claimed_g > 0.5
                s[f"{key}_penalty_pct_pooled"] = float(x.penalty_g[ok].sum() / x.claimed_g[ok].sum() * 100)
                s[f"{key}_penalty_pct_median"] = float(x.penalty_pct_of_claim.median())
                s[f"{key}_penalty_pct_max"] = float(x.penalty_pct_of_claim.max())
                s[f"{key}_zones_penalty_over_10pct"] = int((x.penalty_pct_of_claim > 10).sum())
                s[f"{key}_recovered_pct_pooled"] = float(
                    (x.true_cooling_aware_g[ok].sum() / x.true_carbon_only_g[ok].sum() - 1) * 100)
                y = x[ok & x.penalty_law_pct.notna()]
                s[f"{key}_law_r2"] = float(np.corrcoef(y.penalty_pct_of_claim, y.penalty_law_pct)[0, 1] ** 2)
                s[f"{key}_law_spearman"] = float(y[["penalty_pct_of_claim", "penalty_law_pct"]].corr(method="spearman").iloc[0, 1])
                s[f"{key}_law_median_ratio"] = float((y.penalty_law_pct / y.penalty_pct_of_claim).median())
                s[f"{key}_corr_penalty_vs_profilecorr"] = float(
                    x[ok][["penalty_pct_of_claim", "corr_profile_c_pue"]].corr(method="spearman").iloc[0, 1])
    flat = df[(df.claimed_g < 0.5) & (df.phi == 0.5) & (df["head"] == 0.2)]
    for plant in PLANTS:
        f = flat[flat.plant == plant]
        s[f"{plant}_flat_grid_zones"] = len(f)
        s[f"{plant}_flat_grid_cooling_aware_g_max"] = float(f.true_cooling_aware_g.max()) if len(f) else 0.0
    pd.Series(s).to_csv(RESULTS / "cooling_summary.csv", header=["value"])
    for k, v in s.items():
        print(f"{k:55s} {v:.4g}")
    figure(df)


def figure(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    P.setup()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for plant, col in (("legacy", P.ORANGE), ("hyperscale", P.BLUE)):
        x = df[(df.plant == plant) & (df.phi == 0.5) & (df["head"] == 0.2) & (df.claimed_g > 0.5)]
        a.scatter(x.penalty_law_pct, x.penalty_pct_of_claim, s=10, color=col, lw=0, alpha=0.8, label=plant)
        b.scatter(x.claimed_g, x.true_carbon_only_g, s=10, color=col, lw=0, alpha=0.8, label=plant)
    lim = float(np.nanmax(df[(df.phi == 0.5) & (df["head"] == 0.2)][["penalty_law_pct", "penalty_pct_of_claim"]].to_numpy()))
    a.plot([-5, lim], [-5, lim], color=P.INK_2, lw=0.8)
    a.set(xlabel="Closed-form law (% of claim)", ylabel="Cooling penalty, LP (% of claim)",
          title="Clean hours that are hot cost cooling")
    a.legend(fontsize=7)
    x = df[(df.phi == 0.5) & (df["head"] == 0.2)]
    m = float(max(x.claimed_g.max(), x.true_carbon_only_g.max()))
    b.plot([0, m], [0, m], color=P.INK_2, lw=0.8)
    b.set(xlabel="Claimed saving (g/kWh of IT)", ylabel="True saving (g/kWh of IT)",
          title="Claimed vs true, phi = 0.5, +20%")
    fig.tight_layout()
    P.save(fig, "cooling")


if __name__ == "__main__":
    main()
