"""Who bears the cost of carbon-aware shifting? Incidence under location-based accounting (F28).

Part A (first order, 47 US balancing authorities, EIA-930 2022): a capped facility
(phi = 0.5, 24 h slack, cap 1/0.70, as F17/F19) schedules on the average signal, as
location-based Scope 2 accounting rewards. Its reported saving splits exactly into
the real drop in grid emissions (marginal signal) and a transfer onto other consumers'
location-based inventories. Real savings are scored on the full-year marginal factors
and, as a noise check, on split-half factors (the average-optimal plan does not use the
marginal signal, so the split only tests estimation noise). The marginal-optimal plan is
evaluated out of sample (plan on even-day MEFs, score on odd days, and vice versa).

Part B (exact, finite size, independent margin model, CAISO 2021-2024): a new fleet of
100 MW or 1 GW mean load is added to CAISO. Grid emissions are thermal x 490 +
imports x 428 g/kWh; the fleet's own load is served first by curtailed renewables and
otherwise by gas at 490 g/kWh (the F24 model). Attributions are computed exactly under
location-based and two-part accounting, for the average-optimal (herd) schedule and the
curtailment-aware (coordinated) schedule, and compared with the first-order formula.

Outputs: results/incidence_per_ba.csv, results/incidence_summary.csv,
         results/incidence_caiso.csv, figures/incidence.png
Run: python scripts/incidence_experiments.py   (~3 min)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift import plotting as P
from carbonshift.curtailment import caiso_hourly, coordinated_profile, herd_profile
from carbonshift.eia930 import signals, split_half_marginal
from carbonshift.facility import capped_shift
from carbonshift.incidence import exact_incidence, linear_incidence

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGS = ROOT / "results", ROOT / "figures"
PHI, SLACK, CAP = 0.5, 24, 1 / 0.70
M_GAS, IMP = 490.0, 428.0


def _ba(args):
    ba, avg, mar, me, mo = args
    xa = (1 - PHI) + capped_shift(avg, PHI, CAP, SLACK).served
    xe = (1 - PHI) + capped_shift(me, PHI, CAP, SLACK).served
    xo = (1 - PHI) + capped_shift(mo, PHI, CAP, SLACK).served
    a = linear_incidence(avg, mar, xa)
    a_split = (linear_incidence(avg, me, xa)["real"] + linear_incidence(avg, mo, xa)["real"]) / 2
    # marginal-optimal plan, out of sample: reported (location) saving and held-out real saving
    m_loc = (linear_incidence(avg, mo, xe)["claimed"] + linear_incidence(avg, me, xo)["claimed"]) / 2
    m_real = (linear_incidence(avg, mo, xe)["real"] + linear_incidence(avg, me, xo)["real"]) / 2
    r = avg - mar
    return {"ba": ba, "avg_mean": avg.mean(), "marg_mean": mar.mean(), "residual_mean": r.mean(),
            "avgopt_claimed_g": a["claimed"], "avgopt_real_g": a["real"], "avgopt_transfer_g": a["transfer"],
            "avgopt_real_split_g": a_split,
            "margopt_location_reported_g": m_loc, "margopt_real_oos_g": m_real}


def part_a() -> pd.DataFrame:
    s = signals()
    pa = s.pivot(index="utc", columns="ba", values="average")
    pm = s.pivot(index="utc", columns="ba", values="marginal")
    sp = split_half_marginal()
    pe = sp.pivot(index="utc", columns="ba", values="m_even")
    po = sp.pivot(index="utc", columns="ba", values="m_odd")
    bas = [b for b in pa.columns if pm[b].notna().all() and np.ptp(pa[b]) > 1e-9
           and b in pe.columns and pe[b].notna().all() and po[b].notna().all()]
    tasks = [(b, pa[b].to_numpy(), pm[b].to_numpy(), pe[b].to_numpy(), po[b].to_numpy()) for b in bas]
    with ProcessPoolExecutor() as ex:
        df = pd.DataFrame(list(ex.map(_ba, tasks)))
    df["transfer_share"] = df.avgopt_transfer_g / df.avgopt_claimed_g
    df.to_csv(RESULTS / "incidence_per_ba.csv", index=False)
    return df


def _caiso(args):
    year, P_mw, cap = args
    h = caiso_hourly(year)
    D0 = h.load.to_numpy()
    E0 = h.thermal.to_numpy() * M_GAS + h.imports.to_numpy() * IMP           # kg
    C = h.curtail.to_numpy()
    avg = E0 / D0
    emis = lambda L: M_GAS * np.maximum(0.0, L - C)                           # noqa: E731
    flat = np.full(len(D0), float(P_mw))
    m = np.where(C >= P_mw, 0.0, M_GAS)                                       # margin at the flat operating point
    rows = []
    sched = {"average (herd)": herd_profile(avg, PHI, cap, SLACK),
             "curtailment-aware (coordinated)": coordinated_profile(C / P_mw, np.full(len(D0), M_GAS), PHI, cap, SLACK)}
    for name, x in sched.items():
        e = exact_incidence(E0, D0, emis, flat, P_mw * x, m)
        lin = linear_incidence(avg, m, x)
        mwh = P_mw * len(D0)
        rows.append({"year": year, "fleet_mw": P_mw, "cap": cap, "schedule": name,
                     **{k + "_t": v / 1000 for k, v in e.items()},
                     "real_g": -e["real"] / mwh, "location_claimed_g": -e["location_self"] / mwh,
                     "two_part_claimed_g": -e["two_part_self"] / mwh,
                     "transfer_g": e["location_others"] / mwh, "transfer_linear_g": lin["transfer"],
                     "real_linear_g": lin["real"]})
    return rows


def part_b() -> pd.DataFrame:
    tasks = [(y, p, 1.2) for y in (2021, 2022, 2023, 2024) for p in (100, 1000)]
    with ProcessPoolExecutor() as ex:
        df = pd.DataFrame([r for rows in ex.map(_caiso, tasks) for r in rows])
    df.to_csv(RESULTS / "incidence_caiso.csv", index=False)
    return df


def summarise(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    pos = a.avgopt_claimed_g > 0
    rows = {
        "bas": len(a),
        "avgopt_claimed_g_median": a.avgopt_claimed_g.median(),
        "avgopt_real_g_median": a.avgopt_real_g.median(),
        "transfer_share_pooled": a.avgopt_transfer_g.sum() / a.avgopt_claimed_g.sum(),
        "transfer_share_median": a.transfer_share[pos].median(),
        "transfer_share_pooled_split": 1 - a.avgopt_real_split_g.sum() / a.avgopt_claimed_g.sum(),
        "bas_transfer_over_half": int((a.transfer_share[pos] > 0.5).sum()),
        "bas_transfer_over_100pct": int((a.avgopt_real_g < 0).sum()),
        "residual_mean_g_median": a.residual_mean.median(),
        "margopt_real_oos_g_median": a.margopt_real_oos_g.median(),
        "margopt_location_reported_g_median": a.margopt_location_reported_g.median(),
        "bas_margopt_reported_as_increase": int((a.margopt_location_reported_g < 0).sum()),
        "bas_margopt_real_positive": int((a.margopt_real_oos_g > 0).sum()),
        "bas_real_saving_reported_as_increase": int(((a.margopt_real_oos_g > 0) & (a.margopt_location_reported_g < 0)).sum()),
        "real_gain_margopt_over_avgopt_pooled": a.margopt_real_oos_g.sum() / a.avgopt_real_split_g.sum(),
    }
    h = b[b.schedule.str.startswith("average")]
    rows.update({
        "caiso_herd_transfer_share_min": float((h.transfer_g / h.location_claimed_g).min()),
        "caiso_herd_transfer_share_max": float((h.transfer_g / h.location_claimed_g).max()),
        "caiso_linear_vs_exact_transfer_maxrelerr_100mw": float(
            (abs(h.transfer_linear_g - h.transfer_g) / abs(h.transfer_g))[h.fleet_mw == 100].max()),
        "caiso_two_part_minus_real_g_maxabs": float((b.two_part_claimed_g - b.real_g).abs().max()),
    })
    out = pd.DataFrame([rows]).T.rename(columns={0: "value"})
    out.to_csv(RESULTS / "incidence_summary.csv")
    return out


def figure(a: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    P.setup()
    d = a.sort_values("avgopt_claimed_g", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    x = np.arange(len(d))
    ax.bar(x, d.avgopt_real_g, color="#2a7f62", label="real drop in grid emissions")
    ax.bar(x, d.avgopt_transfer_g, bottom=np.where(d.avgopt_real_g > 0, d.avgopt_real_g, 0),
           color="#d08c3a", label="moved onto other consumers")
    ax.plot(x, d.avgopt_claimed_g, "k_", ms=7, label="reported (location-based)")
    ax.axhline(0, color="0.3", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(d.ba, rotation=90, fontsize=5)
    ax.set_ylabel("g per kWh of facility load")
    ax.set_title("What an average-signal scheduler reports, split by who it lands on", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "incidence.png", dpi=200)


def main() -> None:
    a = part_a()
    b = part_b()
    print(summarise(a, b).to_string())
    print(b[["year", "fleet_mw", "schedule", "real_g", "location_claimed_g", "two_part_claimed_g",
             "transfer_g", "transfer_linear_g"]].round(2).to_string())
    figure(a)


if __name__ == "__main__":
    main()
