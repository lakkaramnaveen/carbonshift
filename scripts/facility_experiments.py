"""Extension: how facility power headroom and UPS batteries bound carbon-aware temporal shifting.

Sweeps all 123 zones (2022) over
  phi   - flexible share of the facility's energy      {0.25, 0.5, 0.75}
  slack - deadline slack of flexible work (hours)      {24, 168}
  cap   - max grid draw / mean load                    1.0 ... 4.0, inf
and battery sizes (hours of mean load) for an otherwise inflexible facility.

Outputs: results/facility_*.csv and figures/facility_*.png
Run: python scripts/facility_experiments.py   (a few minutes on 8+ cores)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift import plotting as P
from carbonshift.data import load_carbon
from carbonshift.facility import battery_shift, capped_shift

RESULTS = Path(__file__).resolve().parents[1] / "results"
PHIS = (0.25, 0.5, 0.75)
SLACKS = (24, 168)
CAPS = (1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, np.inf)
BATTERY_H = (1 / 6, 0.5, 1.0, 2.0, 4.0)  # 1/6 h = 10-minute UPS ride-through at full load
SHOWCASE = ("US-CAL-CISO", "DE", "AU-SA", "IN-WE")


def _zone(args):
    zone, ci = args
    rows = []
    for phi in PHIS:
        for slack in SLACKS:
            for cap in CAPS:
                r = capped_shift(ci, phi, cap, slack)
                rows.append({"zone": zone, "phi": phi, "slack": slack, "cap": cap,
                             "baseline_g": r.baseline_g, "optimised_g": r.optimised_g,
                             "saving_per_flex_kwh": r.saving_per_flex_kwh,
                             "facility_reduction_pct": r.facility_reduction_pct,
                             "peak_draw": float((1 - phi) + r.served.max())})
    batt = []
    for e in BATTERY_H:
        for reserve in (0.0, 0.5):
            b = battery_shift(ci, energy_h=e, reserve_frac=reserve)
            batt.append({"zone": zone, "energy_h": e, "reserve_frac": reserve,
                         "reduction_pct": b.reduction_pct, "cycles_per_year": b.throughput_kwh / (e * (1 - reserve))})
    return rows, batt


def headroom_needed(curve: pd.DataFrame, target: float) -> float:
    """Smallest headroom (cap-1) retaining `target` of the uncapped saving, by linear interpolation."""
    ideal = curve.loc[np.inf]
    if ideal <= 1e-9:
        return np.nan
    finite = curve.drop(index=np.inf) / ideal
    caps, frac = finite.index.to_numpy(), finite.to_numpy()
    if frac[-1] < target:
        return np.inf
    return float(np.interp(target, frac, caps) - 1.0)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    P.setup()
    ci = load_carbon(2022)
    with ProcessPoolExecutor() as ex:
        out = list(ex.map(_zone, [(z, ci[z].to_numpy()) for z in ci.columns]))
    cap_df = pd.DataFrame([r for rows, _ in out for r in rows])
    bat_df = pd.DataFrame([r for _, b in out for r in b])
    cap_df.to_csv(RESULTS / "facility_capped_per_zone.csv", index=False)
    bat_df.to_csv(RESULTS / "facility_battery_per_zone.csv", index=False)

    # Global summary: mean over zones; retained share = ratio of mean savings.
    g = cap_df.groupby(["phi", "slack", "cap"]).agg(
        reduction_pct=("facility_reduction_pct", "mean"),
        saving_g=("saving_per_flex_kwh", "mean"),
        peak_draw=("peak_draw", "median")).reset_index()
    ideal = g[np.isinf(g.cap)].set_index(["phi", "slack"]).saving_g
    g["retained_of_ideal"] = g.apply(lambda r: r.saving_g / ideal[(r.phi, r.slack)], axis=1)
    g.to_csv(RESULTS / "facility_summary.csv", index=False)

    # Headroom each zone needs to keep 50% / 80% of its ideal saving (phi=0.5).
    need = []
    for (zone, slack), d in cap_df[cap_df.phi == 0.5].groupby(["zone", "slack"]):
        curve = d.set_index("cap").saving_per_flex_kwh
        need.append({"zone": zone, "slack": slack, "ideal_g": curve.loc[np.inf],
                     "headroom_for_50pct": headroom_needed(curve, 0.5),
                     "headroom_for_80pct": headroom_needed(curve, 0.8)})
    need = pd.DataFrame(need)
    need.to_csv(RESULTS / "facility_headroom_needed.csv", index=False)

    bsum = bat_df.groupby(["energy_h", "reserve_frac"]).reduction_pct.agg(["mean", "max"]).reset_index()
    bsum.to_csv(RESULTS / "facility_battery_summary.csv", index=False)

    print(g[g.phi == 0.5].round(3).to_string(index=False))
    print(need.groupby("slack")[["headroom_for_50pct", "headroom_for_80pct"]].median())
    print(bsum.round(3).to_string(index=False))
    figures(g, cap_df, bat_df)


def figures(g: pd.DataFrame, cap_df: pd.DataFrame, bat_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    finite = g[np.isfinite(g.cap)]
    # 1) Retained share of the ideal vs headroom, global mean.
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for ax, slack in zip(axes, SLACKS):
        for k, phi in enumerate(PHIS):
            d = finite[(finite.slack == slack) & (finite.phi == phi)]
            ax.plot((d.cap - 1) * 100, d.retained_of_ideal * 100, "o-", ms=3.5,
                    color=(P.BLUE, P.ORANGE, P.AQUA)[k], label=f"{int(phi * 100)}% flexible")
        ax.axhline(100, color=P.INK_2, lw=0.8, ls="--")
        ax.set(xlabel="Power headroom above mean load (%)", title=f"Slack {slack} h", xlim=(0, 300), ylim=(0, 105))
    axes[0].set_ylabel("Share of ideal (uncapped)\ncarbon saving retained (%)")
    axes[0].legend(loc="lower right")
    P.save(fig, "facility_retained")

    # 2) Showcase zones: facility-level reduction vs headroom (phi=0.5, 24 h).
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    sub = cap_df[(cap_df.phi == 0.5) & (cap_df.slack == 24)]
    for k, z in enumerate(SHOWCASE):
        d = sub[sub.zone == z]
        fin = d[np.isfinite(d.cap)]
        col = (P.BLUE, P.ORANGE, P.AQUA, P.MAGENTA)[k]
        ax.plot((fin.cap - 1) * 100, fin.facility_reduction_pct, "o-", ms=3.5, color=col, label=z)
        ax.axhline(float(d[np.isinf(d.cap)].facility_reduction_pct.iloc[0]), color=col, lw=0.8, ls=":")
    ax.set(xlabel="Power headroom above mean load (%)", ylabel="Facility CO$_2$ reduction (%)",
           title="50% flexible, 24 h slack (dotted = uncapped)", xlim=(0, 300))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=8)
    P.save(fig, "facility_zones")

    # 3) Battery: reduction vs size.
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for k, reserve in enumerate((0.0, 0.5)):
        d = bat_df[bat_df.reserve_frac == reserve].groupby("energy_h").reduction_pct
        ax.plot(d.mean().index, d.mean(), "o-", color=(P.BLUE, P.ORANGE)[k],
                label=f"mean, {int(reserve * 100)}% held as ride-through reserve")
    d = bat_df[bat_df.reserve_frac == 0].groupby("energy_h").reduction_pct.max()
    ax.plot(d.index, d, "o--", color=P.AQUA, label="best zone, no reserve")
    ax.set_xscale("log")
    ax.set_xticks(BATTERY_H, ["10 min\n(UPS)", "30 min", "1 h", "2 h", "4 h"])
    ax.minorticks_off()
    ax.set(xlabel="Battery energy (hours of mean load)", ylabel="Facility CO$_2$ reduction (%)",
           title="Battery shifting, inflexible load, 90% round trip")
    ax.legend(fontsize=7.5, loc="upper left")
    P.save(fig, "facility_battery")


if __name__ == "__main__":
    main()
