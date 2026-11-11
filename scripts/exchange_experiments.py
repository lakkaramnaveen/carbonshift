"""Storage exchange rate of data-centre flexibility (F23).

For every zone (Electricity Maps 2022, 123 zones) compare the carbon saving of
capacity-capped flexible load (facility_capped_per_zone.csv, from the LP) with a
real grid-scale battery: 4 h, 90 % round trip, cyclic, behind a flat load (LP).
Battery dispatch is homogeneous in capacity while power <= load, so one LP per zone
gives the saving per kWh, and the exchange rate X = S_flex / b (kWh per kW of load).

Also checks the closed-form daily spread law for b against the battery LP, and
prices the embodied carbon a flexible facility avoids versus building that battery
(Li-ion LFP 54-69 kg CO2eq per kWh, the range in embodied.py) and versus its own cost,
grid-only headroom (transformer + switchgear, per kW).

Outputs: results/exchange_per_zone.csv, results/exchange_summary.csv, figures/exchange_rate.png
Run: python scripts/exchange_experiments.py   (~1 min)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift import plotting as P
from carbonshift.data import load_carbon
from carbonshift.embodied import per_kw
from carbonshift.exchange import exchange_rate, spread_saving_per_kwh
from carbonshift.facility import battery_shift
from carbonshift.waterfill import saving_law

RESULTS = Path(__file__).resolve().parents[1] / "results"
LFP_KG_PER_KWH = (54.0, 61.5, 69.0)
DURATIONS = (2.0, 4.0)


def r2(pred, obs) -> float:
    pred, obs = np.asarray(pred), np.asarray(obs)
    return float(1 - ((pred - obs) ** 2).sum() / ((obs - obs.mean()) ** 2).sum())


def _battery(args):
    zone, ci = args
    out = {"zone": zone}
    for d in DURATIONS:
        b = battery_shift(ci, energy_h=1.0, power=1.0 / d, round_trip_eff=0.90)
        out[f"b{int(d)}h_lp"] = b.baseline_g - b.optimised_g
        out[f"b{int(d)}h_law"] = spread_saving_per_kwh(ci, d, 0.90)
    for phi in (0.25, 0.5):
        for cap in (1.05, 1.1, 1.2, 1.5, np.inf):
            out[f"law_{phi}_{cap}"] = saving_law(ci, phi, cap, 24)
    return out


def main() -> None:
    ci = load_carbon(2022)
    with ProcessPoolExecutor() as ex:
        bat = pd.DataFrame(list(ex.map(_battery, [(z, ci[z].to_numpy()) for z in ci.columns])))

    lp = pd.read_csv(RESULTS / "facility_capped_per_zone.csv")
    lp = lp[(lp.slack == 24) & lp.cap.isin([1.05, 1.1, 1.2, 1.5, np.inf]) & lp.phi.isin([0.25, 0.5])].copy()
    lp["flex_saving_g"] = lp.baseline_g - lp.optimised_g
    df = lp.merge(bat, on="zone")
    df["X_kwh_per_kw"] = [exchange_rate(f, b) for f, b in zip(df.flex_saving_g, df.b4h_lp)]
    df["X_law_kwh_per_kw"] = [exchange_rate(r[f"law_{r.phi}_{r.cap}"], r.b4h_law) for _, r in df.iterrows()]
    df = df.drop(columns=[c for c in df.columns if c.startswith("law_")])
    df["X2h_kwh_per_kw"] = [exchange_rate(f, b) for f, b in zip(df.flex_saving_g, df.b2h_lp)]
    df["virtual_battery_kwh_per_kw"] = 24 * df.phi
    df["headroom_kw_per_kw"] = df.cap - 1
    grid_only = per_kw("grid_only")
    df["battery_embodied_kg_per_kw"] = df.X_kwh_per_kw * LFP_KG_PER_KWH[1]
    df["headroom_embodied_kg_per_kw"] = np.where(np.isfinite(df.cap), (df.cap - 1) * grid_only[1], np.nan)
    df.to_csv(RESULTS / "exchange_per_zone.csv", index=False)

    s = {"zones": int(df.zone.nunique()),
         "spread_law_r2_4h": r2(bat.b4h_law, bat.b4h_lp), "spread_law_r2_2h": r2(bat.b2h_law, bat.b2h_lp),
         "spread_law_median_ratio_4h": float((bat.b4h_law / bat.b4h_lp).replace([np.inf, -np.inf], np.nan).median()),
         "battery_4h_saving_median_g_per_kwh": float(bat.b4h_lp.median())}
    fin = df[np.isfinite(df.X_kwh_per_kw) & np.isfinite(df.X_law_kwh_per_kw) & (df.flex_saving_g > 1e-6)]
    s["closed_form_X_r2_log"] = r2(np.log(fin.X_law_kwh_per_kw), np.log(fin.X_kwh_per_kw))
    s["closed_form_X_median_ratio"] = float((fin.X_law_kwh_per_kw / fin.X_kwh_per_kw).median())
    kappa = df[np.isinf(df.cap) & np.isfinite(df.X_kwh_per_kw)].assign(k=lambda d: d.X_kwh_per_kw / d.virtual_battery_kwh_per_kw)
    s["kappa_uncapped_spread_max_minus_min_over_phi"] = float(kappa.groupby("zone").k.agg(lambda v: v.max() - v.min()).max())
    ok = df[np.isfinite(df.X_kwh_per_kw) & (df.flex_saving_g > 1e-6)]
    for (phi, cap), g in ok.groupby(["phi", "cap"]):
        key = f"phi{phi}_cap{cap}"
        s[f"X_median_{key}"] = float(g.X_kwh_per_kw.median())
        s[f"X_p10_{key}"] = float(g.X_kwh_per_kw.quantile(0.1))
        s[f"X_p90_{key}"] = float(g.X_kwh_per_kw.quantile(0.9))
        s[f"X_over_virtual_median_{key}"] = float((g.X_kwh_per_kw / g.virtual_battery_kwh_per_kw).median())
        if np.isfinite(cap):
            emb = g.X_kwh_per_kw * LFP_KG_PER_KWH[0] / ((cap - 1) * grid_only[2])  # conservative: low battery, high headroom
            s[f"embodied_ratio_battery_over_headroom_min_{key}"] = float(emb.min())
            s[f"embodied_ratio_battery_over_headroom_median_{key}"] = float(emb.median())
    pd.Series(s).to_csv(RESULTS / "exchange_summary.csv", header=["value"])
    for k, v in s.items():
        print(f"{k:60s} {v:.4g}")
    figure(df, bat)


def figure(df: pd.DataFrame, bat: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    P.setup()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    caps = [1.05, 1.1, 1.2, 1.5]
    for phi, col in ((0.25, P.ORANGE), (0.5, P.BLUE)):
        g = df[(df.phi == phi) & df.cap.isin(caps) & np.isfinite(df.X_kwh_per_kw)].groupby("cap").X_kwh_per_kw
        x = (np.array(caps) - 1) * 100
        a.fill_between(x, g.quantile(0.25), g.quantile(0.75), color=col, alpha=0.18, lw=0)
        a.plot(x, g.median(), color=col, marker="o", label=f"φ = {phi}")
    a.set_xlabel("Existing headroom (% of mean load)")
    a.set_ylabel("Battery kWh per kW of load")
    a.set_title("Equal-carbon 4 h battery (median, IQR)")
    a.legend()
    m = max(bat.b4h_lp.max(), bat.b4h_law.max()) * 1.05
    b.plot([0, m], [0, m], color=P.INK_2, lw=0.8)
    b.scatter(bat.b4h_lp, bat.b4h_law, s=10, color=P.BLUE, alpha=0.7, lw=0)
    b.set_xlabel("Battery LP saving (g per kWh, per hour)")
    b.set_ylabel("Daily spread law")
    b.set_title("Spread law vs LP, 123 zones")
    fig.tight_layout()
    P.save(fig, "exchange_rate")


if __name__ == "__main__":
    main()
