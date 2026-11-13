"""Robustness of the cooling penalty (F26) across 2020, 2021 and 2022.

Same 47 zones, hubs and cooling plants as cooling_experiments.py, with each
year's carbon trace and ERA5 weather. phi = 0.5, slack 24 h, headroom +20%
and uncapped. Checks:
  * does the pooled penalty hold each year?
  * is the per-zone penalty stable (Spearman between years)?
  * out of sample: does the closed-form law computed on LAST year's carbon
    and weather predict THIS year's LP penalty? (an operator can evaluate the
    law before running anything)

Outputs: results/cooling_years.csv, results/cooling_years_summary.csv
Run: python scripts/cooling_years.py   (~1 min)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift.cooling import HUBS, PLANTS, decompose, load_weather, penalty_law, plant_pue
from carbonshift.data import load_carbon
from carbonshift.facility import capped_shift

RESULTS = Path(__file__).resolve().parents[1] / "results"
YEARS = (2020, 2021, 2022)
PHI, SLACK, HEADS = 0.5, 24, (0.2, np.inf)


def _run(args):
    year, zone, ci, pues = args
    rows = []
    for plant, p in pues.items():
        law = penalty_law(ci, p) * 100
        for h in HEADS:
            out = {}
            for name, sig in (("carbon_only", ci), ("cooling_aware", ci * p)):
                r = capped_shift(sig, PHI, 1 + h, SLACK)
                out[name] = decompose(ci, p, (1 - PHI) + r.served)
            rows.append({"year": year, "zone": zone, "plant": plant, "head": h, "pue_mean": float(p.mean()),
                         "claimed_g": out["carbon_only"]["claimed"], "true_carbon_only_g": out["carbon_only"]["true"],
                         "true_cooling_aware_g": out["cooling_aware"]["true"], "penalty_g": out["carbon_only"]["penalty"],
                         "law_pct": law})
    return rows


def main() -> None:
    jobs = []
    for y in YEARS:
        ci, w = load_carbon(y), load_weather(y)
        assert len(ci) == len(w), (y, len(ci), len(w))
        for z in HUBS:
            jobs.append((y, z, ci[z].to_numpy(), {k: plant_pue(w, z, k) for k in PLANTS}))
    with ProcessPoolExecutor() as ex:
        df = pd.DataFrame([r for rows in ex.map(_run, jobs) for r in rows])
    df["penalty_pct"] = np.where(df.claimed_g > 0.5, df.penalty_g / df.claimed_g * 100, np.nan)
    df.to_csv(RESULTS / "cooling_years.csv", index=False)

    s = {}
    for (plant, h), g in df.groupby(["plant", "head"]):
        key = f"{plant}_h{'inf' if np.isinf(h) else h}"
        for y in YEARS:
            x = g[(g.year == y) & (g.claimed_g > 0.5)]
            s[f"{key}_{y}_penalty_pct_pooled"] = float(x.penalty_g.sum() / x.claimed_g.sum() * 100)
            s[f"{key}_{y}_penalty_pct_median"] = float(x.penalty_pct.median())
            s[f"{key}_{y}_zones_over_10pct"] = int((x.penalty_pct > 10).sum())
            v = x.dropna(subset=["law_pct"])
            s[f"{key}_{y}_law_r2_same_year"] = float(np.corrcoef(v.penalty_pct, v.law_pct)[0, 1] ** 2)
            s[f"{key}_{y}_recovered_pct_pooled"] = float((x.true_cooling_aware_g.sum() / x.true_carbon_only_g.sum() - 1) * 100)
        wide = g.pivot(index="zone", columns="year", values="penalty_pct")
        law = g.pivot(index="zone", columns="year", values="law_pct")
        for a, b in zip(YEARS[:-1], YEARS[1:]):
            pair = wide[[a, b]].dropna()
            s[f"{key}_spearman_{a}_{b}"] = float(pair.corr(method="spearman").iloc[0, 1])
            oos = pd.DataFrame({"pred": law[a], "lp": wide[b]}).dropna()
            s[f"{key}_oos_law_{a}_to_{b}_r2"] = float(np.corrcoef(oos.pred, oos.lp)[0, 1] ** 2)
            s[f"{key}_oos_law_{a}_to_{b}_spearman"] = float(oos.corr(method="spearman").iloc[0, 1])
            # does last year's law flag this year's >10% zones?
            flag, truth = oos.pred > 10, oos.lp > 10
            s[f"{key}_oos_flag_{a}_to_{b}_hits"] = int((flag & truth).sum())
            s[f"{key}_oos_flag_{a}_to_{b}_misses"] = int((~flag & truth).sum())
            s[f"{key}_oos_flag_{a}_to_{b}_false_alarms"] = int((flag & ~truth).sum())
    pd.Series(s).to_csv(RESULTS / "cooling_years_summary.csv", header=["value"])
    for k, v in s.items():
        print(f"{k:55s} {v:.4g}")


if __name__ == "__main__":
    main()
