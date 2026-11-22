"""Flexible load under a power cap IS a lossless battery (F22).

Theorem: with backlog B_t = arrivals - served, the cumulative-flow constraints of
capped_shift become 0 <= B_t <= s*phi and -h <= dB_t <= phi (flat load, slack s,
headroom h), and the objective is phi*sum(ci) - sum(ci*dB). With e = s*phi - B
this is a lossless battery of energy s*phi, discharge power phi and charge power h
(only the boundary condition differs: empty backlog at the start vs cyclic).

Checks:
  1. Equivalence on real grids: capped_shift vs battery_shift(E = s*phi, P = phi,
     eta = 1, cap = 1 + h) for 20 zones, h <= phi.
  2. Effective horizon: the virtual battery refills in s*phi/h hours, so tight caps
     plan over longer than the slack. Water-filling over T = s*max(1, phi/h) vs fixed
     s-hour blocks, against all LP results (123 zones, 24 h and 168 h slack).
  3. Losses, not batteries, block hedging: F20 found a 90%-efficient battery cannot
     serve both average and marginal ledgers. A *lossless* battery should blend like
     load shifting (F19). 1 h battery, 48 US BAs, w = 0.7.

Outputs: results/virtual_battery_*.csv
Run: python scripts/virtual_battery_experiments.py   (~2 min)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift.data import load_carbon
from carbonshift.eia930 import signals
from carbonshift.facility import battery_shift, capped_shift
from carbonshift.hedge import saving
from carbonshift.waterfill import saving_general

RESULTS = Path(__file__).resolve().parents[1] / "results"


def r2(pred, obs) -> float:
    pred, obs = np.asarray(pred), np.asarray(obs)
    return float(1 - ((pred - obs) ** 2).sum() / ((obs - obs.mean()) ** 2).sum())


def _equiv(args):
    zone, ci = args
    rows = []
    for phi, cap in ((0.5, 1.05), (0.5, 1.2), (0.5, 1.5), (0.25, 1.2)):
        lp = capped_shift(ci, phi, cap, 24)
        b = battery_shift(ci, energy_h=24 * phi, power=phi, round_trip_eff=1.0, cap=cap)
        rows.append({"zone": zone, "phi": phi, "cap": cap, "shift_saving_g": lp.baseline_g - lp.optimised_g,
                     "virtual_battery_saving_g": b.baseline_g - b.optimised_g})
    return rows


def _blend(args):
    ba, avg, mar = args
    out = []
    for eta in (1.0, 0.9):
        for w in (0.0, 0.7, 1.0):
            net = battery_shift(w * avg + (1 - w) * mar, energy_h=1.0, round_trip_eff=eta).net
            out.append({"ba": ba, "eta": eta, "w": w, "avg_saving_g": saving(avg, net, 0.0), "marg_saving_g": saving(mar, net, 0.0)})
    return out


def main() -> None:
    ci = load_carbon(2022)
    summ = {}

    # 1. equivalence
    zones = list(ci.columns[:: max(1, len(ci.columns) // 20)])[:20]
    with ProcessPoolExecutor() as ex:
        eq = pd.DataFrame([r for rows in ex.map(_equiv, [(z, ci[z].to_numpy()) for z in zones]) for r in rows])
    eq.to_csv(RESULTS / "virtual_battery_equivalence.csv", index=False)
    m = eq[eq.shift_saving_g > 1e-6]
    rel = (m.virtual_battery_saving_g - m.shift_saving_g) / m.shift_saving_g
    summ.update(equiv_cases=len(m), equiv_median_abs_rel_diff=float(rel.abs().median()), equiv_max_abs_rel_diff=float(rel.abs().max()))

    # 2. effective horizon
    lp = pd.read_csv(RESULTS / "facility_capped_per_zone.csv").assign(save=lambda d: d.baseline_g - d.optimised_g)
    rows = []
    for r in lp.itertuples():
        x = ci[r.zone].to_numpy()
        h = r.cap - 1
        T = int(round(r.slack * max(1.0, r.phi / h))) if np.isfinite(r.cap) and h > 0 else int(r.slack)
        rows.append({"zone": r.zone, "slack": r.slack, "phi": r.phi, "cap": r.cap, "lp": r.save,
                     "block_law": saving_general(x, r.phi, r.cap, int(r.slack)), "horizon_law": saving_general(x, r.phi, r.cap, T)})
    hz = pd.DataFrame(rows)
    hz.to_csv(RESULTS / "virtual_battery_horizon_law.csv", index=False)
    for s, g in hz.groupby("slack"):
        g = g[g.lp > 1e-6]
        for c in ("block_law", "horizon_law"):
            summ[f"{c}_slack{s}_r2"] = r2(g[c], g.lp)
            summ[f"{c}_slack{s}_median_abs_rel_err"] = float((abs(g[c] - g.lp) / g.lp).median())
            summ[f"{c}_slack{s}_bias_at_cap1.05"] = float(((g[c] - g.lp) / g.lp)[np.isclose(g.cap, 1.05)].median())

    # 3. lossless vs lossy battery under blending
    s = signals()
    pa = s.pivot(index="utc", columns="ba", values="average")
    pm = s.pivot(index="utc", columns="ba", values="marginal")
    bas = [b for b in pa.columns if pm[b].notna().all() and np.ptp(pa[b]) > 1e-9 and pm[b].mean() > 0]
    with ProcessPoolExecutor() as ex:
        bl = pd.DataFrame([r for rows in ex.map(_blend, [(b, pa[b].to_numpy(), pm[b].to_numpy()) for b in bas]) for r in rows])
    bl.to_csv(RESULTS / "virtual_battery_blend_per_ba.csv", index=False)
    for eta in (1.0, 0.9):
        e = bl[bl.eta == eta]
        ref_a = e[e.w == 1.0].avg_saving_g.sum()
        ref_m = e[e.w == 0.0].marg_saving_g.sum()
        x = e[e.w == 0.7]
        tag = "lossless" if eta == 1.0 else "eta90"
        summ.update({f"{tag}_w0.7_raises_avg_pct": float((x.avg_saving_g < 0).mean() * 100),
                     f"{tag}_w0.7_raises_marg_pct": float((x.marg_saving_g < 0).mean() * 100),
                     f"{tag}_w0.7_pooled_avg_share": float(x.avg_saving_g.sum() / ref_a),
                     f"{tag}_w0.7_pooled_marg_share": float(x.marg_saving_g.sum() / ref_m),
                     f"{tag}_w1_raises_marg_pct": float((e[e.w == 1.0].marg_saving_g < 0).mean() * 100)})
    summ["bas"] = len(bas)
    pd.Series(summ).to_csv(RESULTS / "virtual_battery_summary.csv", header=["value"])
    print(pd.Series(summ).round(4).to_string())


if __name__ == "__main__":
    main()
