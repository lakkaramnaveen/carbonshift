"""Monte Carlo: does "build power, not servers" (F8) survive embodied-carbon uncertainty?

Per draw: embodied factors per build (``embodied.sample_per_kw``), lifetimes
(electrical 20-30 y, servers 4-6 y) and an annual decline in savings from grid
decarbonisation (0-5 %/y). Annual savings per kW are each zone's shadow price
from scripts/redundancy_experiments.py (phi = 0.5, 24 h slack). Output: the
distribution of the share of zones whose headroom repays its embodied carbon
within its life, and how often the ordering of builds flips.

Outputs: results/embodied_uncertainty_summary.csv
Run: python scripts/embodied_uncertainty.py   (seconds)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift.embodied import BUILDS, pays_back_within, sample_per_kw

RESULTS = Path(__file__).resolve().parents[1] / "results"
N = 5000
LIFE = {"grid_only": (20, 30), "backed": (20, 30), "cpu_servers": (4, 6), "gpu_servers": (4, 6)}


def main() -> None:
    marg = pd.read_csv(RESULTS / "redundancy_marginal_per_zone.csv")
    rng = np.random.default_rng(2026)
    rows = []
    for cap in (1.2, 1.5):
        g = marg[(marg.slack == 24) & np.isclose(marg.cap, cap)].g_per_kw_year.to_numpy()  # zones
        decline = rng.uniform(0.0, 0.05, N)
        shares = {}
        for b in BUILDS:
            emb = sample_per_kw(b, rng, N)
            life = rng.uniform(*LIFE[b], N)
            ok = pays_back_within(emb[:, None], g[None, :], life[:, None], decline[:, None])  # draws x zones
            shares[b] = ok.mean(axis=1) * 100
        order = (shares["cpu_servers"] < shares["gpu_servers"]) & (shares["gpu_servers"] <= shares["backed"]) & (shares["backed"] <= shares["grid_only"])
        for b, s in shares.items():
            rows.append({"cap": cap, "build": b, "median_pct": np.median(s), "p05_pct": np.percentile(s, 5), "p95_pct": np.percentile(s, 95),
                         "prob_majority_pay_back": float((s > 50).mean())})
        rows.append({"cap": cap, "build": "ordering cpu<gpu<=backed<=grid holds", "median_pct": np.nan, "p05_pct": np.nan, "p95_pct": np.nan,
                     "prob_majority_pay_back": float(order.mean())})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "embodied_uncertainty_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
