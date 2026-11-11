"""Are UPS batteries worth more under a marginal signal? US BAs, EIA-930 2022.

F5 found a 10-minute UPS battery cuts average-signal emissions by only 0.19%.
Battery arbitrage is pure incremental load, so the marginal signal is the right
accounting. For each BA and battery size, dispatch on the average signal, the
marginal signal, and the F19 blend (w = 0.7), and score each in both signals.
Round-trip efficiency 90%, no reserve, no grid cap.

Noise check: marginal factors are regression estimates, and an optimiser will
exploit their noise. Re-estimate them on even and on odd days of the year,
dispatch on one half and score on the other (out-of-sample), versus dispatch
and score on the same half (in-sample). Done for batteries and for capped load
shifting (phi = 0.5, 24 h slack, cap 1/0.70, as F17).

Outputs: results/battery_marginal_*.csv, figures/battery_marginal.png
Run: python scripts/battery_marginal_experiments.py   (~2 min)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift import plotting as P
from carbonshift.eia930 import signals, split_half_marginal
from carbonshift.facility import battery_shift, capped_shift
from carbonshift.hedge import saving

RESULTS = Path(__file__).resolve().parents[1] / "results"
SIZES_H = [1 / 6, 1.0, 4.0]
W_BLEND = 0.7
PHI, SLACK, CAP = 0.5, 24, 1 / 0.70


def _split_task(args):
    ba, me, mo = args
    rows = []
    plans = {f"battery_{e:.3g}h": (lambda sig, e=e: battery_shift(sig, energy_h=e).net, 0.0) for e in SIZES_H}
    plans["shift_capped"] = (lambda sig: capped_shift(sig, PHI, CAP, SLACK).served, PHI)
    for name, (plan, phi) in plans.items():
        xe, xo = plan(me), plan(mo)
        ins = (saving(mo, xo, phi) + saving(me, xe, phi)) / 2
        oos = (saving(mo, xe, phi) + saving(me, xo, phi)) / 2
        rows.append({"ba": ba, "plan": name, "in_sample_g": ins, "out_of_sample_g": oos})
    return rows


def _task(args):
    ba, avg, mar = args
    rows = []
    for e in SIZES_H:
        for name, sig in (("average", avg), ("marginal", mar), ("blend", W_BLEND * avg + (1 - W_BLEND) * mar)):
            net = battery_shift(sig, energy_h=e).net
            rows.append({"ba": ba, "energy_h": e, "dispatch_on": name,
                         "avg_red_pct": saving(avg, net, 0.0) / avg.mean() * 100,
                         "marg_red_pct": saving(mar, net, 0.0) / mar.mean() * 100})
    return rows


def main() -> None:
    P.setup()
    s = signals()
    pa = s.pivot(index="utc", columns="ba", values="average")
    pm = s.pivot(index="utc", columns="ba", values="marginal")
    bas = [b for b in pa.columns if pm[b].notna().all() and np.ptp(pa[b]) > 1e-9 and pm[b].mean() > 0]
    with ProcessPoolExecutor() as ex:
        df = pd.DataFrame([r for rows in ex.map(_task, [(b, pa[b].to_numpy(), pm[b].to_numpy()) for b in bas]) for r in rows])
    df.to_csv(RESULTS / "battery_marginal_per_ba.csv", index=False)

    g = df.groupby(["energy_h", "dispatch_on"])
    summ = g.agg(avg_red_median=("avg_red_pct", "median"), marg_red_median=("marg_red_pct", "median"),
                 avg_red_max=("avg_red_pct", "max"), marg_red_max=("marg_red_pct", "max"),
                 bas_raising_avg_pct=("avg_red_pct", lambda x: (x < 0).mean() * 100),
                 bas_raising_marg_pct=("marg_red_pct", lambda x: (x < 0).mean() * 100)).reset_index()
    summ.insert(0, "bas", len(bas))
    summ.to_csv(RESULTS / "battery_marginal_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))

    import matplotlib.pyplot as plt

    d = df[df.energy_h == 1 / 6].pivot(index="ba", columns="dispatch_on", values=["avg_red_pct", "marg_red_pct"])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    lim = [0, d.max().max() * 1.05]
    axes[0].scatter(d[("avg_red_pct", "average")], d[("marg_red_pct", "marginal")], s=14, color=P.BLUE)
    axes[0].plot(lim, lim, color=P.INK_2, lw=0.8, ls="--")
    axes[0].set(xlabel="Reduction, average signal (%)", ylabel="Reduction, marginal signal (%)",
                title="10-min UPS: each signal's own optimum")
    for name, c in (("average", P.ORANGE), ("blend", P.BLUE)):
        axes[1].scatter(d[("avg_red_pct", name)], d[("marg_red_pct", name)], s=14, color=c, label=f"dispatch on {name}")
    axes[1].axhline(0, color=P.INK_2, lw=0.6)
    axes[1].set(xlabel="Reduction, average signal (%)", ylabel="Reduction, marginal signal (%)",
                title="Average-dispatched vs blended (w = 0.7)")
    axes[1].legend(frameon=False, fontsize=8)
    P.save(fig, "battery_marginal")

    sp = split_half_marginal()
    piv_e = sp.pivot(index="utc", columns="ba", values="m_even")
    piv_o = sp.pivot(index="utc", columns="ba", values="m_odd")
    sb = [b for b in bas if b in piv_e.columns and piv_e[b].notna().all() and piv_o[b].notna().all() and np.ptp(piv_e[b]) > 1e-9]
    with ProcessPoolExecutor() as ex:
        oos = pd.DataFrame([r for rows in ex.map(_split_task, [(b, piv_e[b].to_numpy(), piv_o[b].to_numpy()) for b in sb]) for r in rows])
    oos.to_csv(RESULTS / "battery_marginal_split_per_ba.csv", index=False)
    os_ = oos.groupby("plan").agg(bas=("ba", "size"), in_sample_g_sum=("in_sample_g", "sum"), out_of_sample_g_sum=("out_of_sample_g", "sum"),
                                   bas_oos_negative_pct=("out_of_sample_g", lambda x: (x < 0).mean() * 100))
    os_["retained_pooled"] = os_.out_of_sample_g_sum / os_.in_sample_g_sum
    mm = sp.groupby("ba")[["m_even", "m_odd"]].mean().mean(axis=1)
    pct = oos.assign(ins_pct=oos.in_sample_g / oos.ba.map(mm) * 100, oos_pct=oos.out_of_sample_g / oos.ba.map(mm) * 100)
    os_ = os_.join(pct.groupby("plan")[["ins_pct", "oos_pct"]].median().add_suffix("_median"))
    os_.to_csv(RESULTS / "battery_marginal_split_summary.csv")
    print(os_.round(3).to_string())


if __name__ == "__main__":
    main()
