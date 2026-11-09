"""Do spatial and temporal flexibility add up under per-region power caps?

Two GCP region groups (Europe, US) with measured latency, 100 ms limit, phi = 0.5,
24 h slack, 2022, solved in monthly blocks. Scenarios: temporal only (no
migration), spatial only (no slack), joint. Caps: Google-real headroom
(1/0.70) and effectively unlimited (50x mean load).

Interaction = joint saving / (temporal saving + spatial saving):
1 means additive, < 1 means the two compete for the same headroom.

Outputs: results/joint_summary.csv
Run: python scripts/joint_experiments.py   (a few minutes on 14 cores)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift.data import load_carbon
from carbonshift.spatial_capacity import joint_shift

RESULTS = Path(__file__).resolve().parents[1] / "results"
PHI, SLACK, LIMIT = 0.5, 24, 100
GROUPS = {"Europe": ["PL", "FI", "ES", "BE", "GB", "DE", "NL", "CH", "IT-NO", "FR"],
          "US": ["US-CENT-SWPP", "US-CAR-SC", "US-MIDA-PJM", "US-TEX-ERCO", "US-NW-PACW", "US-CAL-CISO", "US-NW-PACE", "US-NW-NEVP"]}
CAPS = {"real (1/0.70)": 1 / 0.70, "unlimited": 50.0}


def _task(args):
    group, capname, month, ci_block, lat = args
    cap = CAPS[capname]
    T, R = ci_block.shape
    base = float(ci_block.mean())
    regs = [str(i) for i in range(R)]
    temporal = joint_shift(ci_block, regs, lat, PHI, cap, limit_ms=-1, slack=SLACK)
    spatial = joint_shift(ci_block, regs, lat, PHI, cap, limit_ms=LIMIT, slack=0)
    joint = joint_shift(ci_block, regs, lat, PHI, cap, limit_ms=LIMIT, slack=SLACK)
    return {"group": group, "cap": capname, "month": month, "hours": T, "base": base, "temporal": temporal, "spatial": spatial, "joint": joint}


def main() -> None:
    ci = load_carbon(2022)
    lat_df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "gcp_latency_matrix.csv", index_col="origin")
    lat_df = lat_df[~lat_df.index.duplicated()]
    lat_df = lat_df.loc[:, ~lat_df.columns.duplicated()]
    jobs = []
    for group, zones in GROUPS.items():
        m = lat_df.loc[zones, zones].to_numpy(float)
        lat = (m + m.T) / 2
        X = ci[zones]
        for capname in CAPS:
            for month in range(1, 13):
                block = X[X.index.month == month].to_numpy()
                jobs.append((group, capname, month, block, lat))
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(_task, jobs))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "joint_per_month.csv", index=False)
    w = df.assign(**{c: df[c] * df.hours for c in ("base", "temporal", "spatial", "joint")}).groupby(["group", "cap"])[["base", "temporal", "spatial", "joint", "hours"]].sum()
    for c in ("base", "temporal", "spatial", "joint"):
        w[c] /= w.hours
    out = pd.DataFrame({
        "temporal_pct": (w.base - w.temporal) / w.base * 100,
        "spatial_pct": (w.base - w.spatial) / w.base * 100,
        "joint_pct": (w.base - w.joint) / w.base * 100,
    })
    out["additive_sum_pct"] = out.temporal_pct + out.spatial_pct
    out["interaction_ratio"] = out.joint_pct / out.additive_sum_pct
    out["joint_extra_over_spatial_pct"] = out.joint_pct - out.spatial_pct
    out = out.reset_index()
    out.to_csv(RESULTS / "joint_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
