"""Hourly matching (24/7 CFE) rewards load shifting that the grid does not see (F30).

Setting as F17/F28: 48 US balancing authorities (EIA-930 2022), a facility of mean load 1,
phi = 0.5 flexible, 24 h slack, power cap 1/0.70. The facility holds a solar portfolio
(clear-sky shape at the BA's centroid) sized to 100% of its annual load (50% and 150% as
sensitivity), already built. It schedules to minimise hourly-matched market-based emissions
sum_t c_t max(L_t - P_t, 0), with c_t the hourly grid average as residual factor.

For each BA and schedule we report, in g per kWh of mean load, against running flat:
  reported = drop in hourly-matched emissions; real = drop in grid emissions (marginal signal,
  portfolio fixed); transfer = reported - real (others' residual-mix inventories).
Schedules: hourly-matching optimal with three tie-breaks (average signal = main; none;
marginal signal = best case for the grid), the location-optimal plan (F28) and the
marginal-optimal plan (out of sample, split-half MEFs). Real savings of the hourly-matching
plans are also scored on split-half MEFs as a noise check.

CAISO check: the same experiment for CISO with the real 2022 CAISO solar profile and a
solar + wind (50/50 energy) profile instead of the clear-sky shape.

Outputs: results/hourly_matching_per_ba.csv, results/hourly_matching_summary.csv,
         results/hourly_matching_caiso.csv, figures/hourly_matching.png
Run: python scripts/hourly_matching_experiments.py   (~5 min)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift import plotting as P
from carbonshift.curtailment import caiso_hourly
from carbonshift.eia930 import signals, split_half_marginal
from carbonshift.facility import capped_shift
from carbonshift.hourly_matching import BA_LATLON, clear_sky_solar, hm_shift, ledger, portfolio

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGS = ROOT / "results", ROOT / "figures"
PHI, SLACK, CAP = 0.5, 24, 1 / 0.70
KAPPAS = (0.5, 1.0, 1.5)


def _runs(ba, utc, avg, mar, me, mo, shape, kappas=KAPPAS):
    rows = []
    loc = (1 - PHI) + capped_shift(avg, PHI, CAP, SLACK).served
    xe = (1 - PHI) + capped_shift(me, PHI, CAP, SLACK).served
    xo = (1 - PHI) + capped_shift(mo, PHI, CAP, SLACK).served
    for k in kappas:
        Pk = portfolio(shape, k)
        plans = {"hm_avg_tb": hm_shift(avg, Pk, PHI, CAP, SLACK, tiebreak=avg),
                 "hm_no_tb": hm_shift(avg, Pk, PHI, CAP, SLACK),
                 "hm_marg_tb": hm_shift(avg, Pk, PHI, CAP, SLACK, tiebreak=mar),
                 "location_opt": loc}
        for name, d in plans.items():
            L = ledger(avg, mar, Pk, d)
            split = (ledger(avg, me, Pk, d)["real"] + ledger(avg, mo, Pk, d)["real"]) / 2
            rows.append({"ba": ba, "kappa": k, "plan": name, **L, "real_split": split})
        # marginal-optimal, out of sample: plan on even-day MEFs, score on odd days and vice versa
        le, lo_ = ledger(avg, mo, Pk, xe), ledger(avg, me, Pk, xo)
        rows.append({"ba": ba, "kappa": k, "plan": "marginal_opt_oos",
                     "reported": (le["reported"] + lo_["reported"]) / 2, "real": (le["real"] + lo_["real"]) / 2,
                     "transfer": np.nan, "cfe_flat": le["cfe_flat"],
                     "cfe_shifted": (le["cfe_shifted"] + lo_["cfe_shifted"]) / 2, "real_split": np.nan})
    return rows


def _ba(args):
    ba, utc, avg, mar, me, mo = args
    lat, lon = BA_LATLON[ba]
    return _runs(ba, utc, avg, mar, me, mo, clear_sky_solar(utc, lat, lon))


def _panels():
    s = signals()
    pa = s.pivot(index="utc", columns="ba", values="average")
    pm = s.pivot(index="utc", columns="ba", values="marginal")
    sp = split_half_marginal()
    pe = sp.pivot(index="utc", columns="ba", values="m_even")
    po = sp.pivot(index="utc", columns="ba", values="m_odd")
    bas = [b for b in pa.columns if pm[b].notna().all() and np.ptp(pa[b]) > 1e-9
           and b in pe.columns and pe[b].notna().all() and po[b].notna().all()]
    return pd.DatetimeIndex(pa.index), pa, pm, pe, po, bas


def per_ba() -> pd.DataFrame:
    utc, pa, pm, pe, po, bas = _panels()
    tasks = [(b, utc, pa[b].to_numpy(), pm[b].to_numpy(), pe[b].to_numpy(), po[b].to_numpy()) for b in bas]
    with ProcessPoolExecutor() as ex:
        rows = [r for rs in ex.map(_ba, tasks) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "hourly_matching_per_ba.csv", index=False)
    return df


def caiso_profiles(utc: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Real CAISO 2022 solar and wind, local prevailing hour-ending -> UTC hour-beginning."""
    h = caiso_hourly(2022)
    local = pd.to_datetime(h["date"]) + pd.to_timedelta(h["hour"] - 1, unit="h")
    idx = local.dt.tz_localize("America/Los_Angeles", ambiguous="NaT", nonexistent="NaT")
    ok = idx.notna().to_numpy()
    z = pd.DataFrame({"solar": h.solar.to_numpy()[ok], "wind": h.wind.to_numpy()[ok]},
                     index=idx[ok].dt.tz_convert("UTC").dt.tz_localize(None))
    z = z.groupby(level=0).mean().reindex(utc).interpolate(limit_direction="both")
    sol, wind = z.solar.clip(lower=0).to_numpy(), z.wind.clip(lower=0).to_numpy()
    return {"real_solar": sol, "real_solar_wind": sol / sol.mean() + wind / wind.mean()}


def caiso() -> pd.DataFrame:
    utc, pa, pm, pe, po, _ = _panels()
    a, m, e, o = (x["CISO"].to_numpy() for x in (pa, pm, pe, po))
    prof = caiso_profiles(utc)
    cs = clear_sky_solar(utc, *BA_LATLON["CISO"])
    rows = []
    for name, shape in {"clear_sky": cs, **prof}.items():
        for r in _runs("CISO", utc, a, m, e, o, shape, kappas=(1.0,)):
            rows.append({"profile": name, **r})
    df = pd.DataFrame(rows)
    df.attrs["r_clear_vs_real_solar"] = float(np.corrcoef(cs, prof["real_solar"])[0, 1])
    df.to_csv(RESULTS / "hourly_matching_caiso.csv", index=False)
    return df


def summarise(df: pd.DataFrame, cz: pd.DataFrame) -> pd.DataFrame:
    out = {}
    k1 = df[df.kappa == 1.0]
    for plan, g in k1.groupby("plan"):
        out[f"{plan}_reported_g_median"] = g.reported.median()
        out[f"{plan}_real_g_median"] = g.real.median()
        out[f"{plan}_cfe_gain_pp_median"] = 100 * (g.cfe_shifted - g.cfe_flat).median()
        out[f"{plan}_bas_real_negative"] = int((g.real < 0).sum())
        if plan != "marginal_opt_oos":
            out[f"{plan}_transfer_share_pooled"] = g.transfer.sum() / g.reported.sum()
            out[f"{plan}_transfer_share_median"] = (g.transfer / g.reported).median()
            out[f"{plan}_real_share_pooled_split"] = g.real_split.sum() / g.reported.sum()
            out[f"{plan}_bas_transfer_over_half"] = int((g.transfer / g.reported > 0.5).sum())
    out["bas"] = k1.ba.nunique()
    out["cfe_flat_median"] = k1[k1.plan == "hm_avg_tb"].cfe_flat.median()
    hm = k1[k1.plan == "hm_avg_tb"].set_index("ba")
    mo = k1[k1.plan == "marginal_opt_oos"].set_index("ba")
    out["real_gain_margopt_over_hm_pooled"] = mo.real.sum() / hm.real.sum()
    out["margopt_hm_reported_over_hm_opt_pooled"] = mo.reported.sum() / hm.reported.sum()
    for k in KAPPAS:
        g = df[(df.kappa == k) & (df.plan == "hm_avg_tb")]
        out[f"kappa{k}_hm_transfer_share_pooled"] = g.transfer.sum() / g.reported.sum()
        out[f"kappa{k}_hm_reported_g_median"] = g.reported.median()
    for prof, g in cz[cz.plan == "hm_avg_tb"].groupby("profile"):
        r = g.iloc[0]
        out[f"caiso_{prof}_reported_g"] = r.reported
        out[f"caiso_{prof}_real_g"] = r.real
    out["caiso_r_clear_vs_real_solar"] = cz.attrs.get("r_clear_vs_real_solar", np.nan)
    s = pd.Series(out, name="value")
    s.to_csv(RESULTS / "hourly_matching_summary.csv")
    return s


def figure(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    P.setup()
    g = df[(df.kappa == 1.0) & (df.plan == "hm_avg_tb")].sort_values("reported")
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    x = np.arange(len(g))
    ax.bar(x, g.reported, color="#9db4d8", label="reported (hourly matching)")
    ax.bar(x, g.real, color="#1f4e8c", label="real (marginal, portfolio fixed)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, g.ba, rotation=90, fontsize=6)
    ax.set_ylabel("g CO2 per kWh of mean load")
    ax.set_title("Shifting to maximise the 24/7 CFE score: reported vs real saving", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "hourly_matching.png", dpi=200)
    plt.close(fig)


def main() -> None:
    df = per_ba()
    cz = caiso()
    print(summarise(df, cz).to_string())
    figure(df)


if __name__ == "__main__":
    main()
