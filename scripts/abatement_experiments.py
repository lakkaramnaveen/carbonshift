"""F34: the price of a shifted tonne. What does carbon-aware shifting cost per tonne?

For every zone (Electricity Maps average signal, 123 zones x 2020-2022) and every US
balancing authority (EIA-930 2022, average and marginal signals), with phi = 0.5 and
24 h slack, the capped LP (facility.capped_shift) gives the carbon value of headroom
three ways:

* v(h) = S(h)/h, the average value of a build of h (primal, h = 0.01, 0.2, 0.42);
* the LP dual: grams per year saved by the *next* kW at h (F7's shadow price);
* the daily closed form D (F33) and the horizon-free bound c_max - cbar.

These are priced with abatement.COSTS (facility power, GPU capacity, 4 h battery),
annualised at 3 / 7 / 10 %, and compared with the US EPA social cost of carbon.

Outputs: results/abatement_{zones,bas,summary,sensitivity}.csv,
figures/abatement_mac.png, paper/sigmetrics/fig_abatement.pdf.
`--plot-only` recomputes the summary and figure from the CSVs.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift import abatement as ab
from carbonshift import idle
from carbonshift.data import load_carbon
from carbonshift.facility import capped_shift

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
PHI = 0.5
HS = (0.01, 0.2, 0.42)


def zone_row(ci, phi=PHI):
    ci = np.asarray(ci, dtype=float)
    row = dict(cbar=float(ci.mean()), cmax=float(ci.max()), D=idle.first_unit_value(ci),
               bound=ab.value_bound(ci), batt_t_per_kwh=ab.battery_tonnes_per_kwh_year(ci))
    for h in HS:
        r = capped_shift(ci, phi, 1.0 + h, 24)
        S = r.baseline_g - r.optimised_g
        row[f"v_h{h}"] = S / h                                   # g per kW-headroom per hour
        row[f"dual_t_h{h}"] = r.annual_marginal_saving / 1e6     # t per extra kW per year
        row[f"moved_mwh_h{h}"] = ab.moved_mwh_per_kw_year(r.served, phi, h)
    return row


def main():
    rows = []
    for year in (2020, 2021, 2022):
        df = load_carbon(year)
        for z in df.columns:
            rows.append(dict(year=year, zone=z, **zone_row(df[z].to_numpy())))
        print(year, "done", flush=True)
    Z = pd.DataFrame(rows)
    Z.to_csv(OUT / "abatement_zones.csv", index=False)

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
    B.to_csv(OUT / "abatement_bas.csv", index=False)
    summarise(Z, B)
    plot(Z, B)


def macs(g, level="central", rate=0.07):
    """Per-zone abatement costs ($/t) of the options."""
    kf, kg, kb = (ab.annual_cost(i, level, rate) for i in ("facility_power", "gpu_capacity", "battery_4h"))
    t1 = g["v_h0.01"] * ab.T_PER_GKWH_YEAR          # first unit of headroom (cheapest tonnes)
    t42 = g["v_h0.42"] * ab.T_PER_GKWH_YEAR         # average over a Google-real +42% build
    out = pd.DataFrame(dict(
        facility_first=kf / t1, facility_h042=kf / t42, facility_marginal_h02=kf / g["dual_t_h0.2"],
        gpu_first=kg / t1, gpu_h042=kg / t42,
        battery=kb / g.batt_t_per_kwh,
        facility_floor=kf / (g.bound.clip(lower=0) * ab.T_PER_GKWH_YEAR),
        gpu_floor=kg / (g.bound.clip(lower=0) * ab.T_PER_GKWH_YEAR),
    ))
    return out


def summarise(Z, B):
    s = {}
    groups = [(f"zones{y}", Z[Z.year == y]) for y in (2020, 2021, 2022)] + \
             [("ba_average", B[B.signal == "average"]), ("ba_marginal", B[B.signal == "marginal"])]
    kf = ab.annual_cost("facility_power")
    kg = ab.annual_cost("gpu_capacity")
    s["kappa_facility_central"] = kf
    s["kappa_gpu_capacity_central"] = kg
    s["kappa_battery_central_per_kwh"] = ab.annual_cost("battery_4h")
    s["swing_needed_facility_central_scc190"] = ab.swing_needed(kf, ab.SCC["central"])
    s["swing_needed_facility_low_r3_scc340"] = ab.swing_needed(ab.annual_cost("facility_power", "low", 0.03), 340)
    s["swing_needed_gpu_low_r3_scc340"] = ab.swing_needed(ab.annual_cost("gpu_capacity", "low", 0.03), 340)
    for name, g in groups:
        M = macs(g)
        s[f"{name}_n"] = len(g)
        for col in M.columns:
            s[f"{name}_mac_{col}_median"] = M[col].median()
            s[f"{name}_mac_{col}_min"] = M[col].min()
            s[f"{name}_share_{col}_below_scc190"] = float((M[col] < 190).mean())
        s[f"{name}_t_first_median"] = (g["v_h0.01"] * ab.T_PER_GKWH_YEAR).median()
        s[f"{name}_t_dual_h0.01_median"] = g["dual_t_h0.01"].median()
        s[f"{name}_t_D_median"] = (g.D * ab.T_PER_GKWH_YEAR).median()
        s[f"{name}_t_bound_median"] = (g.bound * ab.T_PER_GKWH_YEAR).median()
        s[f"{name}_max_v_over_bound"] = float((g["v_h0.01"] / g.bound).max())
        be = ab.breakeven_capex(g["v_h0.01"] * ab.T_PER_GKWH_YEAR, 190, 20, 0.07)
        s[f"{name}_breakeven_capex_per_kw_first_median"] = be.median()
        s[f"{name}_breakeven_capex_per_kw_first_max"] = be.max()
        s[f"{name}_breakeven_share_of_central_capex_median"] = be.median() / ab.COSTS["facility_power"]["capex"][1]
        be42 = ab.breakeven_capex(g["v_h0.42"] * ab.T_PER_GKWH_YEAR, 190, 20, 0.07)
        s[f"{name}_breakeven_capex_per_kw_h042_median"] = be42.median()
        s[f"{name}_breakeven_capex_per_kw_h042_max"] = be42.max()
        s[f"{name}_battery_cheaper_share"] = float((M.battery < M.facility_first).mean())
        s[f"{name}_facility_over_battery_median"] = (M.facility_first / M.battery).median()
        t = g["v_h0.01"] * ab.T_PER_GKWH_YEAR
        pg = [ab.price_gain_needed(kf, ti, mi, 190) for ti, mi in zip(t, g["moved_mwh_h0.01"])]
        s[f"{name}_moved_mwh_first_median"] = g["moved_mwh_h0.01"].median()
        s[f"{name}_price_gain_needed_first_median"] = float(np.median(pg))
        t42 = g["v_h0.42"] * ab.T_PER_GKWH_YEAR
        pg42 = [ab.price_gain_needed(kf, ti, mi, 190) for ti, mi in zip(t42, g["moved_mwh_h0.42"])]
        s[f"{name}_moved_mwh_h042_median"] = g["moved_mwh_h0.42"].median()
        s[f"{name}_price_gain_needed_h042_median"] = float(np.median(pg42))
    pd.Series(s).to_csv(OUT / "abatement_summary.csv", header=["value"])
    for k, v in s.items():
        print(f"{k:60s} {v:.4g}")

    # Sensitivity: most favourable to least favourable combinations
    sens = []
    for name, g in groups:
        for level in ("low", "central", "high"):
            for rate in (0.03, 0.07, 0.10):
                M = macs(g, level, rate)
                for scc_name, scc in ab.SCC.items():
                    sens.append(dict(group=name, cost=level, rate=rate, scc=scc_name,
                                     facility_first_median=M.facility_first.median(),
                                     share_facility_first_pass=float((M.facility_first < scc).mean()),
                                     share_facility_h042_pass=float((M.facility_h042 < scc).mean()),
                                     share_gpu_first_pass=float((M.gpu_first < scc).mean()),
                                     share_facility_floor_pass=float((M.facility_floor < scc).mean()),
                                     share_gpu_floor_pass=float((M.gpu_floor < scc).mean()),
                                     share_battery_pass=float((M.battery < scc).mean()),
                                     battery_median=M.battery.median()))
    S = pd.DataFrame(sens)
    S.to_csv(OUT / "abatement_sensitivity.csv", index=False)
    best = S[(S.cost == "low") & (S.rate == 0.03) & (S.scc == "high")]
    print(best.to_string(index=False))


def plot(Z, B):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = Z[Z.year == 2022]
    M = macs(g)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    series = [("GPU capacity, +42% build", M.gpu_h042, "#7b3294"),
              ("facility power, +42% build", M.facility_h042, "#2a6fbb"),
              ("facility power, first kW", M.facility_first, "#5aa0d8"),
              ("4 h battery", M.battery, "#1a9850")]
    for lab, x, col in series:
        x = np.sort(np.clip(x.to_numpy(), 10, 1e6))
        ax.plot(x, np.arange(1, len(x) + 1) / len(x) * 100, color=col, lw=1.5, label=lab)
    for v, lab in ((190, "SCC 190"), (340, "340")):
        ax.axvline(v, color="#c0392b", lw=0.9, ls="--")
        ax.text(v * (0.55 if v == 190 else 1.08), 101, lab, fontsize=5.5, color="#c0392b", va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(50, 3e5)
    ax.set_xlabel("abatement cost of carbon-aware shifting (USD per t CO2, log)", fontsize=6.5)
    ax.set_ylabel("zones (%), cumulative", fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.set_ylim(0, 108)
    ax.legend(fontsize=5.5, loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "abatement_mac.png", dpi=200)
    fig.savefig(ROOT / "paper" / "sigmetrics" / "fig_abatement.pdf")
    print("figure written")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        Z, B = pd.read_csv(OUT / "abatement_zones.csv"), pd.read_csv(OUT / "abatement_bas.csv")
        summarise(Z, B)
        plot(Z, B)
    else:
        main()
