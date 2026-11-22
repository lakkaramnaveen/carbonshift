"""When does carbon-aware computing saturate? A growing fleet against CAISO curtailment (F24).

A *new* data-centre fleet of mean load P (MW) is added to California's grid (2021-2024; 2022 is the main year)
(CAISO 5-minute production and curtailment data, aggregated hourly). Its emissions are
m * max(0, L_t - C_t): curtailed renewables are free, everything else comes from the
gas-fired margin (m = 490 g/kWh lifecycle; marginal supply in CAISO is gas or imports).
Half the load is flexible with 24 h slack and a facility cap (+20%, +50%, uncapped).

Strategies, all scored on the same convex cost:
  run-on-arrival      flat draw
  herd (average)      every facility optimises alone on CAISO average intensity
                      ((thermal x 490 + imports x 428) / load; 428 = CARB unspecified imports)
  herd (curtailment)  every facility optimises alone on a 'free when curtailed' signal
                      (0 if C_t > 0 else m, ties broken by the average signal)
  coordinated         the fleet-level LP that knows how much is curtailed
Plus the closed-form daily absorption bound, checked against the coordinated LP.
Sensitivity: m_t = EIA-930 CISO season x hour marginal factors instead of a constant.
Motivation check (supply_curve_check): across U.S. balancing authorities, does the
marginal emission rate rise with fossil output, which would let a herd raise the margin?
Needs the raw EIA-930 files (git-ignored); skipped if they are absent.

Outputs: results/curtailment_fleet.csv, results/curtailment_summary.csv, results/curtailment_years_summary.csv, figures/curtailment_fleet.png
Run: python scripts/curtailment_experiments.py   (~2 min)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift import plotting as P
from carbonshift.curtailment import absorbable_daily, caiso_hourly, coordinated_profile, fleet_emissions, herd_profile

RESULTS = Path(__file__).resolve().parents[1] / "results"
M_GAS = 490.0
IMPORT_T_PER_MWH = 0.428
FLEETS_MW = [100, 250, 500, 1000, 2000, 4000, 8000]
CAPS = [1.2, 1.5, np.inf]
PHI, SLACK = 0.5, 24
YEARS = [2021, 2022, 2023, 2024]


def inputs(time_varying: bool = False, year: int = 2022):
    h = caiso_hourly(year)
    avg = (h.thermal * M_GAS + h.imports * IMPORT_T_PER_MWH * 1000) / h.load
    m = np.full(len(h), M_GAS)
    if time_varying:
        from carbonshift.eia930 import signals
        s = signals()
        s = s[s.ba == "CISO"].set_index("utc").marginal
        local = (h.date + pd.to_timedelta(h.hour - 1, unit="h")).dt.tz_localize(
            "America/Los_Angeles", ambiguous=True, nonexistent="shift_forward").dt.tz_convert("UTC").dt.tz_localize(None)
        m = s.reindex(local).interpolate().bfill().ffill().clip(lower=0).to_numpy()
    return h.curtail.to_numpy(), avg.to_numpy(), m


def _run(args):
    P_mw, cap, tv, year = args
    C, avg, m = inputs(tv, year)
    Cpu = C / P_mw
    curt_sig = np.where(C > 0, 0.0, M_GAS) + 1e-3 * avg
    prof = {"run_on_arrival": np.ones(len(C)),
            "herd_average": herd_profile(avg, PHI, cap, SLACK),
            "herd_curtailment": herd_profile(curt_sig, PHI, cap, SLACK),
            "coordinated": coordinated_profile(Cpu, m, PHI, cap, SLACK)}
    rows = []
    flex_mwh = PHI * P_mw * len(C)
    base_t = fleet_emissions(P_mw * prof["run_on_arrival"], C, m)
    for k, x in prof.items():
        L = P_mw * x
        t = fleet_emissions(L, C, m)
        rows.append({"year": year, "fleet_mw": P_mw, "cap": cap, "marginal": "eia930_hourly" if tv else "gas_constant", "strategy": k,
                     "emissions_t": t, "saving_t": base_t - t, "saving_g_per_flex_kwh": (base_t - t) * 1e6 / (flex_mwh * 1000),
                     "curtailed_share_of_energy": float(np.minimum(L, C).sum() / L.sum()),
                     "absorbed_twh": float(np.minimum(L, C).sum() / 1e6)})
    rows.append({"year": year, "fleet_mw": P_mw, "cap": cap, "marginal": "eia930_hourly" if tv else "gas_constant", "strategy": "closed_form_bound",
                 "curtailed_share_of_energy": absorbable_daily(Cpu, PHI, cap)})
    return rows


def main() -> None:
    for y in YEARS:
        caiso_hourly(y)  # build the derived hourly files once, before forking
    jobs = [(p, c, False, y) for y in YEARS for c in CAPS for p in FLEETS_MW] + [(p, c, True, 2022) for c in CAPS for p in FLEETS_MW]
    with ProcessPoolExecutor() as ex:
        allyears = pd.DataFrame([r for rows in ex.map(_run, jobs) for r in rows])
    allyears.to_csv(RESULTS / "curtailment_fleet.csv", index=False)
    years_summary(allyears)
    supply_curve_check()
    df = allyears[allyears.year == 2022]
    C, avg, _ = inputs()
    s = {"curtailed_twh_2022": C.sum() / 1e6, "hours_with_curtailment": int((C > 0).sum()),
         "curtailment_mean_mw": C.mean(), "caiso_mean_load_mw": float(caiso_hourly(2022).load.mean())}
    g = df[df.marginal == "gas_constant"].pivot_table(index=["cap", "fleet_mw"], columns="strategy", values="saving_g_per_flex_kwh")
    sh = df[df.marginal == "gas_constant"].pivot_table(index=["cap", "fleet_mw"], columns="strategy", values="curtailed_share_of_energy")
    for (cap, p), r in g.iterrows():
        k = f"cap{cap}_P{p}"
        s[f"coord_g_per_flex_kwh_{k}"] = r.coordinated
        s[f"herd_avg_over_coord_{k}"] = r.herd_average / r.coordinated if r.coordinated > 0 else np.nan
        s[f"herd_curt_over_coord_{k}"] = r.herd_curtailment / r.coordinated if r.coordinated > 0 else np.nan
        s[f"bound_over_lp_share_{k}"] = sh.loc[(cap, p), "closed_form_bound"] / sh.loc[(cap, p), "coordinated"]
    tv = df[(df.marginal == "eia930_hourly") & (df.strategy != "closed_form_bound")]
    s["tv_cases_herd_raises_emissions"] = int((tv[tv.strategy.str.startswith("herd")].saving_t < 0).sum())
    s["tv_cases_herd"] = int(tv.strategy.str.startswith("herd").sum())
    pd.Series(s).to_csv(RESULTS / "curtailment_summary.csv", header=["value"])
    print(df.drop(columns=["emissions_t"]).round(3).to_string())
    for k, v in s.items():
        print(f"{k:50s} {v:.4g}")
    figure(df)


def years_summary(df: pd.DataFrame) -> None:
    d = df[(df.marginal == "gas_constant") & (df.strategy != "closed_form_bound")]
    s = {}
    for y in YEARS:
        C = caiso_hourly(y).curtail.to_numpy()
        s[f"curtailed_twh_{y}"] = C.sum() / 1e6
        for cap in CAPS:
            g = d[(d.year == y) & (d.cap == cap)].pivot_table(index="fleet_mw", columns="strategy", values="saving_t")
            s[f"max_avoided_kt_{y}_cap{cap}"] = float(g.coordinated.max() / 1000)
            s[f"herd_worst_kt_{y}_cap{cap}"] = float(min(g.herd_average.min(), g.herd_curtailment.min()) / 1000)
            s[f"herd_min_share_of_coord_{y}_cap{cap}"] = float((g.herd_curtailment / g.coordinated)[g.coordinated > 1].min())
            gpk = d[(d.year == y) & (d.cap == cap) & (d.strategy == "coordinated")].set_index("fleet_mw").saving_g_per_flex_kwh
            s[f"coord_g_per_flex_kwh_{y}_cap{cap}_P100"] = float(gpk[100])
            s[f"coord_g_per_flex_kwh_{y}_cap{cap}_P1000"] = float(gpk[1000])
        b = df[(df.year == y) & (df.marginal == "gas_constant")].pivot_table(index=["cap", "fleet_mw"], columns="strategy", values="curtailed_share_of_energy")
        r = b.closed_form_bound / b.coordinated
        s[f"closed_form_over_lp_min_{y}"], s[f"closed_form_over_lp_max_{y}"] = float(r.min()), float(r.max())
    pd.Series(s).to_csv(RESULTS / "curtailment_years_summary.csv", header=["value"])
    for k, v in s.items():
        print(f"{k:50s} {v:.4g}")


def supply_curve_check() -> None:
    """Fit E = a + b F + c F^2 per BA x season (F = fossil MW) and compare the slope at low and high output."""
    from carbonshift.eia930 import FUELS, RAW, SEASON
    files = sorted(RAW.glob("EIA930_BALANCE_2022_*.csv"))
    if not files:
        print("supply-curve check skipped: raw EIA-930 files not present")
        return
    fossil = ["Coal", "Natural Gas", "All Petroleum Products", "Other Fuel Sources", "Unknown Fuel Sources"]
    cols = ["Balancing Authority", "UTC Time at End of Hour"] + [f"Net Generation (MW) from {f} (Adjusted)" for f in fossil]
    d = pd.concat([pd.read_csv(f, usecols=cols, thousands=",", low_memory=False) for f in files])
    d["month"] = pd.to_datetime(d["UTC Time at End of Hour"], format="%m/%d/%Y %I:%M:%S %p").dt.month
    gen = d[cols[2:]].clip(lower=0).fillna(0.0)
    d["F"] = gen.sum(axis=1) / 1000
    d["E"] = (gen * np.array([FUELS[f] for f in fossil])).sum(axis=1) / 1000
    rows = []
    for ba, g in d[d.F > 0].groupby("Balancing Authority"):
        if len(g) < 8000 or g.F.mean() < 0.3:
            continue
        for season in range(4):
            x = g[g.month.map(SEASON) == season]
            A = np.c_[np.ones(len(x)), x.F, x.F ** 2]
            coef = np.linalg.lstsq(A, x.E, rcond=None)[0]
            lo, hi = np.percentile(x.F, [10, 90])
            rows.append({"ba": ba, "season": season, "slope_p10": coef[1] + 2 * coef[2] * lo, "slope_p90": coef[1] + 2 * coef[2] * hi})
    r = pd.DataFrame(rows)
    rel = (r.slope_p90 / r.slope_p10 - 1).abs()
    s = {"supply_ba_seasons": len(r), "supply_bas": r.ba.nunique(), "supply_median_abs_change_p10_p90": float(rel.median()),
         "supply_share_change_below_10pct": float((rel < 0.10).mean()), "supply_share_rising": float((r.slope_p90 > r.slope_p10).mean())}
    pd.Series(s).to_csv(RESULTS / "curtailment_supply_curve.csv", header=["value"])
    for k, v in s.items():
        print(f"{k:50s} {v:.4g}")


def figure(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    P.setup()
    d = df[df.marginal == "gas_constant"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    lines = [(np.inf, "coordinated", P.BLUE, "-", "uncapped, coordinated"), (np.inf, "herd_curtailment", P.ORANGE, "-", "uncapped, herd"),
             (1.2, "coordinated", P.BLUE, "--", "+20% cap, coordinated"), (1.2, "herd_curtailment", P.ORANGE, ":", "+20% cap, herd")]
    for cap, k, col, ls, lab in lines:
        x = d[(d.cap == cap) & (d.strategy == k)]
        a.plot(x.fleet_mw / 1000, x.saving_t / 1000, color=col, ls=ls, marker="o", ms=3, label=lab)
    a.axhline(0, color=P.INK_2, lw=0.6)
    a.set_ylabel("Avoided vs run-on-arrival (kt CO2/yr)")
    a.set_title("Value of carbon-awareness, CAISO 2022")
    a.legend(fontsize=6.8, loc="upper left")
    x = d[(d.cap == 1.2) & (d.strategy == "coordinated")]
    b.plot(x.fleet_mw / 1000, x.curtailed_share_of_energy * 100, color=P.BLUE, marker="o", ms=3, label="coordinated LP (+20% cap)")
    bd = d[(d.cap == 1.2) & (d.strategy == "closed_form_bound")]
    b.plot(bd.fleet_mw / 1000, bd.curtailed_share_of_energy * 100, color=P.INK, lw=0.8, ls="--", label="daily water-filling formula")
    r = d[(d.cap == 1.2) & (d.strategy == "run_on_arrival")]
    b.plot(r.fleet_mw / 1000, r.curtailed_share_of_energy * 100, color=P.INK_2, ls=":", marker="o", ms=2.5, label="run-on-arrival")
    b.set_ylabel("Fleet energy from curtailment (%)")
    b.set_title("Free energy runs out")
    b.legend(fontsize=6.8)
    for ax in (a, b):
        ax.set_xscale("log")
        ax.set_xlabel("New flexible fleet, mean load (GW)")
    fig.tight_layout()
    P.save(fig, "curtailment_fleet")


if __name__ == "__main__":
    main()
