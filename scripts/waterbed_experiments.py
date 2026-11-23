"""The waterbed: what carbon-aware shifting cuts under an emissions cap (F31).

Part A (EU ETS bank). Mechanical simulation of the TNAC / Market Stability Reserve from the
published 2022-2025 TNACs (2025: 1,023.5 Mt) to 2039, for a grid of stylised net-supply
paths d_y = d2026 - decline * (y - 2026) and three rule sets (Directive 2023/959; the
provisional October 2026 deal with a 700 Mt invalidation threshold; the Commission's April
2026 proposal without invalidation). For each path and abatement year 2025-2033: the share of
an avoided tonne destroyed, locked in the MSR at the end, or leaked back to the market
(waterbed), by two independent derivations (re-simulation and forward linearisation).
Results are also binned by the lag between the abatement year and the year the TNAC first
falls to 833 Mt (the bottom of the absorbing band).

Part B (who is under a cap). Each of the 123 Electricity Maps zones is mapped to its cap
regime. With the repository's temporal-shifting saving (slack 24 h, 1 h jobs, 2022;
results/repro_temporal_per_zone.csv) and the cloud-region counts in
data/provider_regions.csv, we report the share of the headline saving (zone-equal and
cloud-region weighted) that sits under a cap, and the share of US marginal savings (F17,
results/marginal_per_ba.csv) under WCI / RGGI / Washington.

Outputs: results/waterbed_fate.csv, results/waterbed_fate_by_lag.csv,
         results/waterbed_zones.csv, results/waterbed_summary.csv, figures/waterbed.png
Run: python scripts/waterbed_experiments.py   (~20 s)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift import plotting as P
from carbonshift.waterbed import (
    REGIME_BRACKET, RULES, TNAC_PUBLISHED, cap_regime, fate_bruteforce, fate_linear,
    implied_net_supply, net_supply_path, simulate,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EU_RULES = ["law_2023 (400 Mt)", "deal_2026 (700 Mt)", "proposal_2026 (none)"]
D0 = range(0, 201, 25)
DECLINE = range(0, 61, 10)
YEARS = range(2025, 2034)


def part_a() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for name in EU_RULES:
        r = RULES[name]
        for d0 in D0:
            for dl in DECLINE:
                d = net_supply_path(d0, dl)
                b = simulate(d, r)
                cross = next((y for y in sorted(b.tnac) if b.tnac[y] <= 833.0), None)
                for y in YEARS:
                    fl = fate_linear(d, r, y)
                    fb = fate_bruteforce(d, r, y, size=1e-3)
                    rows.append({"rules": name, "d2026": d0, "decline": dl, "cross_833": cross,
                                 "tnac_2026": b.tnac[2026], "tnac_2027": b.tnac[2027],
                                 "tnac_min": min(b.tnac.values()), "year": y,
                                 "lag": (y - cross) if cross else np.nan,
                                 **{f"lin_{k}": v for k, v in fl.items()},
                                 **{f"bf_{k}": v for k, v in fb.items()}})
    df = pd.DataFrame(rows)
    df["agree"] = (df.lin_real_upper - df.bf_real_upper).abs() < 1e-6
    ok = df[df.cross_833.notna() & (df.cross_833 <= 2035)]
    lag = (ok.groupby(["rules", "lag"])
             .agg(n=("lin_real_upper", "size"),
                  destroyed_median=("lin_destroyed", "median"),
                  real_upper_median=("lin_real_upper", "median"),
                  real_upper_p10=("lin_real_upper", lambda s: s.quantile(0.1)),
                  real_upper_p90=("lin_real_upper", lambda s: s.quantile(0.9)),
                  share_real_gt_half=("lin_real_upper", lambda s: (s > 0.5).mean()))
             .reset_index())
    return df, lag


def part_b() -> pd.DataFrame:
    tz = pd.read_csv(RESULTS / "repro_temporal_per_zone.csv")
    tz = tz[(tz.slack == 24) & (tz.job == 1)][["zone", "defer"]]
    prov = pd.read_csv(ROOT / "data" / "provider_regions.csv", encoding="utf-8-sig")
    prov = prov[prov.zone_code.notna()][["zone_code", "SUM"]].rename(columns={"zone_code": "zone", "SUM": "cloud_regions"})
    z = tz.merge(prov, on="zone", how="left").fillna({"cloud_regions": 0})
    z["regime"] = z.zone.map(cap_regime)
    return z


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    print("implied net supply d_2024, d_2025 (Mt):", {k: round(v, 1) for k, v in implied_net_supply().items()})
    df, lag = part_a()
    df.to_csv(RESULTS / "waterbed_fate.csv", index=False)
    lag.to_csv(RESULTS / "waterbed_fate_by_lag.csv", index=False)
    print(f"derivations agree (|lin - bf| < 1e-6) in {df.agree.mean():.1%} of {len(df)} cases")
    print(lag[(lag.lag >= -4) & (lag.lag <= 4)].to_string(index=False))

    z = part_b()
    z.to_csv(RESULTS / "waterbed_zones.csv", index=False)
    rows = []
    for w in ("zone", "cloud"):
        wt = np.ones(len(z)) if w == "zone" else z.cloud_regions.to_numpy()
        tot = (wt * z.defer).sum()
        for reg, g in z.groupby("regime"):
            ww = wt[g.index]
            lo, hi = REGIME_BRACKET.get(reg, (np.nan, np.nan))
            rows.append({"weight": w, "regime": reg, "zones": len(g), "cloud_regions": int(g.cloud_regions.sum()),
                         "saving_share": (ww * g.defer).sum() / tot, "bracket_low": lo, "bracket_high": hi})
    summ = pd.DataFrame(rows)

    # US marginal savings (F17) under caps
    mb = pd.read_csv(RESULTS / "marginal_per_ba.csv")
    capped = {"CISO", "BANC", "LDWP", "TIDC", "ISNE", "NYIS", "SCL", "TPWR", "PSEI", "CHPD", "DOPD", "GCPD"}
    in_cap = mb.ba.isin(capped)
    us_share = mb.loc[in_cap, "marg_saving_best_g"].sum() / mb.marg_saving_best_g.sum()
    summ = pd.concat([summ, pd.DataFrame([{"weight": "us_marginal_F17", "regime": "WCI+RGGI+WA",
                                           "zones": int(in_cap.sum()), "saving_share": us_share}])])

    # headline: EU-cloud-region saving times fate, for shifting done in 2026 vs 2028
    main_r = df[(df.rules == "deal_2026 (700 Mt)")]
    for y in (2025, 2026, 2027, 2028, 2030):
        s = main_r[main_r.year == y].lin_real_upper
        summ = pd.concat([summ, pd.DataFrame([{"weight": "eu_fate", "regime": f"abated {y}", "zones": len(s),
                                               "saving_share": s.median(), "bracket_low": s.quantile(0.1),
                                               "bracket_high": s.quantile(0.9)}])])
    summ.to_csv(RESULTS / "waterbed_summary.csv", index=False)
    print(summ.to_string(index=False))

    # figure
    P.setup()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), gridspec_kw={"wspace": 0.55})
    ax = axes[0]
    styles = {EU_RULES[0]: dict(color=P.BLUE, ls="--", lw=1.2, marker=None),
              EU_RULES[1]: dict(color=P.ORANGE, ls="-", lw=2, marker="o"),
              EU_RULES[2]: dict(color=P.AQUA, ls=":", lw=1.5, marker="s", ms=3)}
    for name in EU_RULES:
        g = lag[(lag.rules == name) & (lag.lag.between(-6, 6))]
        y = g.real_upper_median if name != EU_RULES[2] else g.destroyed_median
        ax.plot(g.lag, y, label=name if name != EU_RULES[2] else "proposal_2026: destroyed only", zorder=3 if name == EU_RULES[0] else 2,
                **styles[name])
        if name == EU_RULES[1]:
            ax.fill_between(g.lag, g.real_upper_p10, g.real_upper_p90, color=P.ORANGE, alpha=0.15, lw=0)
    ax.axvline(-0.5, color=P.INK_2, lw=0.8, ls=":")
    ax.set_xlabel("abatement year minus year TNAC first reaches 833 Mt")
    ax.set_ylabel("real cut per tonne avoided")
    ax.set_title("EU ETS: fate of an avoided tonne")
    ax.legend(fontsize=7, loc="upper right")
    ax = axes[1]
    s = summ[summ.weight == "cloud"].sort_values("saving_share")
    ax.barh(s.regime, s.saving_share * 100, color=[P.BLUE if r == "no cap" else P.ORANGE for r in s.regime])
    ax.set_xlabel("share of cloud-weighted shifting saving (%)")
    ax.set_title("Where the headline saving sits")
    hist = ", ".join(f"{y}: {v:,.0f}" for y, v in TNAC_PUBLISHED.items())
    fig.suptitle(f"Published TNAC (Mt) {hist}", fontsize=8, y=1.02)
    P.save(fig, "waterbed")


if __name__ == "__main__":
    main()
