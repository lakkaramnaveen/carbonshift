"""The power meter as a side channel of carbon-aware scheduling (F25).

Simulated facilities on real carbon traces (Electricity Maps 2022, 41 zones) with
inflexible load that is either flat or a real Google power-domain shape (hour-of-week
profile x day-to-day variability bootstrapped in 24 h blocks from the real May 2019
trace), plus 1% meter noise. Parameters: phi in {0.1, 0.25, 0.5}, slack in {12, 24} h,
headroom h in {0.2, 0.5}.

Schedulers:
  realtime     optimise on the actual hourly carbon signal (what carbon-aware systems do)
  time_of_day  optimise on each month's average carbon by hour of day (a 'quiet' schedule)
  dither       optimise on the signal plus Gaussian noise (0.5 x its std)
  unaware      run on arrival (the control group)

Observer: sees the meter and the public carbon signal for 7, 14, 30 or 90 days and
computes the correlation of hour-of-day residuals (sidechannel.detection_stat).
Reports detection AUC (aware vs unaware), the saving each scheduler keeps, the
identity saving = -cov(c, d), and meter-only estimates of phi, h and the virtual
battery energy s*phi.

Outputs: results/sidechannel_*.csv, figures/sidechannel.png
Run: python scripts/sidechannel_experiments.py   (~3 min)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift import plotting as P
from carbonshift.data import load_carbon
from carbonshift.facility import capped_shift
from carbonshift.google_power import load_all, tile, week_profile
from carbonshift.sidechannel import auc, detection_stat, infer_battery, saving_from_cov, time_of_day_signal

RESULTS = Path(__file__).resolve().parents[1] / "results"
INDEX = pd.date_range("2022-01-01", periods=8760, freq="h")
HOURS, MONTHS = INDEX.hour.to_numpy(), INDEX.month.to_numpy()
WINDOWS = [7, 14, 30, 90]
STARTS = [10, 100, 190, 260]  # day of year where each observation window starts
PHIS, SLACKS, HEADS = (0.1, 0.25, 0.5), (12, 24), (0.2, 0.5)
METER_NOISE = 0.01


def google_residual_days() -> tuple[list[str], dict]:
    """Real day-to-day variability: per domain, 24-hour blocks of trace / hour-of-week mean."""
    d = load_all().set_index("ts").groupby("domain").measured_power_util.resample("1h").mean().dropna().reset_index()
    d["how"] = d.ts.dt.day_of_week * 24 + d.ts.dt.hour
    d["ratio"] = d.measured_power_util / d.groupby(["domain", "how"]).measured_power_util.transform("mean")
    d["day"] = d.ts.dt.normalize()
    blocks = {}
    for dom, g in d.groupby("domain"):
        days = [x.ratio.to_numpy() for _, x in g.groupby("day") if len(x) == 24]
        if len(days) >= 20:
            blocks[dom] = np.array(days)
    return sorted(blocks), blocks


def load_shape(domain: str, blocks: np.ndarray, rng) -> np.ndarray:
    base = tile(week_profile(domain), INDEX)
    noise = blocks[rng.integers(0, len(blocks), 365)].ravel()
    x = base * noise
    return x / x.mean()


def _run(args):
    zone, ci, kind, domain, blocks, seed = args
    rng = np.random.default_rng(seed)
    load = np.ones(8760) if kind == "flat" else load_shape(domain, blocks, rng)
    sig = {"realtime": ci, "time_of_day": time_of_day_signal(ci, HOURS, MONTHS),
           "dither": ci + rng.normal(0, 0.5 * ci.std(), len(ci))}
    rows, est = [], []
    for phi in PHIS:
        for s in SLACKS:
            for h in HEADS:
                cap = load.max() * (1 + h) if kind != "flat" else 1 + h
                draws = {"unaware": load}
                for k, x in sig.items():
                    r = capped_shift(x, phi, cap, s, load=load)
                    draws[k] = (1 - phi) * load + r.served
                base_g = float((ci * load).mean())
                for k, d in draws.items():
                    meter = d * (1 + rng.normal(0, METER_NOISE, len(d)))
                    row = {"zone": zone, "load": kind, "phi": phi, "slack": s, "head": h, "scheduler": k,
                           "saving_g": base_g - float((ci * d).mean()), "saving_cov_g": saving_from_cov(ci, d),
                           "saving_pct": (base_g - float((ci * d).mean())) / base_g * 100}
                    for w in WINDOWS:
                        for i, st in enumerate(STARTS):
                            sl = slice(st * 24, (st + w) * 24)
                            row[f"T{w}_{i}"] = detection_stat(ci[sl], meter[sl], HOURS[sl])
                    rows.append(row)
                    if k == "realtime":
                        for w in (30, 365):
                            sl = slice(STARTS[0] * 24, (STARTS[0] + w) * 24) if w < 365 else slice(None)
                            e = infer_battery(meter[sl])
                            est.append({"zone": zone, "load": kind, "phi": phi, "slack": s, "head": h, "days": w, **e,
                                        "energy_true": s * phi})
    return rows, est


def main() -> None:
    ci = load_carbon(2022)
    zones = [z for z in ci.columns[::3] if ci[z].std() > 1.0]
    doms, blocks = google_residual_days()
    rng = np.random.default_rng(0)
    jobs = []
    for i, z in enumerate(zones):
        dom = doms[rng.integers(len(doms))]
        for kind in ("flat", "google"):
            jobs.append((z, ci[z].to_numpy(), kind, dom, blocks[dom], 1000 + i))
    with ProcessPoolExecutor() as ex:
        out = list(ex.map(_run, jobs))
    df = pd.DataFrame([r for rows, _ in out for r in rows])
    est = pd.DataFrame([r for _, e in out for r in e])
    df.to_csv(RESULTS / "sidechannel_runs.csv", index=False)
    est.to_csv(RESULTS / "sidechannel_estimates.csv", index=False)

    s = {"zones": len(zones), "facilities": int(len(df) / 4)}
    flat = df[(df.load == "flat") & (df.scheduler != "unaware")]
    s["identity_max_abs_diff_g"] = float((flat.saving_g - flat.saving_cov_g).abs().max())
    aucs = []
    for kind in ("flat", "google"):
        g = df[df.load == kind]
        for w in WINDOWS:
            cols = [f"T{w}_{i}" for i in range(len(STARTS))]
            neg = g[g.scheduler == "unaware"][cols].to_numpy().ravel()
            for k in ("realtime", "time_of_day", "dither"):
                pos = g[g.scheduler == k][cols].to_numpy().ravel()
                a = auc(pos, neg)
                aucs.append({"load": kind, "days": w, "scheduler": k, "auc": a})
                s[f"auc_{kind}_{k}_{w}d"] = a
    pd.DataFrame(aucs).to_csv(RESULTS / "sidechannel_auc.csv", index=False)
    for kind in ("flat", "google"):
        g = df[df.load == kind].pivot_table(index=["zone", "phi", "slack", "head"], columns="scheduler", values="saving_g")
        ok = g.realtime > 0.5
        for k in ("time_of_day", "dither"):
            s[f"saving_kept_{kind}_{k}_pooled"] = float(g[k][ok].sum() / g.realtime[ok].sum())
            s[f"saving_kept_{kind}_{k}_median"] = float((g[k][ok] / g.realtime[ok]).median())
    for (kind, days), e in est.groupby(["load", "days"]):
        key = f"{kind}_{days}d"
        s[f"phi_mae_{key}"] = float((e.phi_hat - e.phi).abs().mean())
        s[f"head_mae_{key}"] = float((e.h_hat - e["head"]).abs().mean())
        s[f"energy_median_ratio_{key}"] = float((e.energy_hat / e.energy_true).median())
        s[f"energy_rank_corr_{key}"] = float(e[["energy_hat", "energy_true"]].corr(method="spearman").iloc[0, 1])
        s[f"phi_rank_corr_{key}"] = float(e[["phi_hat", "phi"]].corr(method="spearman").iloc[0, 1])
    pd.Series(s).to_csv(RESULTS / "sidechannel_summary.csv", header=["value"])
    for k, v in s.items():
        print(f"{k:50s} {v:.4g}")
    figure(pd.DataFrame(aucs), df)


def figure(aucs: pd.DataFrame, df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    P.setup()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    style = {"realtime": (P.BLUE, "real-time signal"), "dither": (P.AQUA, "signal + noise"), "time_of_day": (P.ORANGE, "monthly time-of-day")}
    for k, (col, lab) in style.items():
        for kind, ls in (("flat", "--"), ("google", "-")):
            x = aucs[(aucs.scheduler == k) & (aucs.load == kind)]
            a.plot(x.days, x.auc, color=col, ls=ls, marker="o", ms=3, label=f"{lab}" if kind == "google" else None)
    a.axhline(0.5, color=P.INK_2, lw=0.6)
    a.set(xscale="log", xlabel="Days of meter data observed", ylabel="Detection AUC", ylim=(0.4, 1.02),
          title="Is this facility carbon-aware?")
    a.set_xticks(WINDOWS, [str(w) for w in WINDOWS])
    a.legend(fontsize=6.8, loc="lower right", title="solid: Google load; dashed: flat", title_fontsize=6.5)
    g = df[df.load == "google"].pivot_table(index=["zone", "phi", "slack", "head"], columns="scheduler", values="saving_g")
    g = g[g.realtime > 0.5]
    for k, (col, lab) in style.items():
        if k == "realtime":
            continue
        b.scatter(g.realtime, g[k], s=6, color=col, alpha=0.5, lw=0, label=lab)
    m = g.realtime.max()
    b.plot([0, m], [0, m], color=P.INK_2, lw=0.8)
    b.set(xlabel="Saving, real-time signal (g/kWh)", ylabel="Saving, quieter scheduler (g/kWh)", title="What hiding costs")
    b.legend(fontsize=6.8)
    fig.tight_layout()
    P.save(fig, "sidechannel")


if __name__ == "__main__":
    main()
