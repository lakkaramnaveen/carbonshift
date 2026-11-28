"""F36: the carbon signal is a methodology, not a measurement.

For the 49 US balancing authorities present in both the Electricity Maps trace
(consumption-based, flow-traced; EuroSys'24 data) and the EIA-930 production-based
average signal (2022), with phi = 0.5 and 24 h slack:

* optimise a schedule on one signal and evaluate it under the other (daily
  water-filling, provenance.cross_evaluate) at h = 0.2, 0.42 and uncapped;
* compare with the regression prediction (kept ~ rho, confirmed ~ beta);
* re-derive the h = 0.2 numbers with the sliding-window LP (facility.capped_shift),
  an independent route that does not use calendar days;
* compare how the two methodologies rank BAs by level (siting) and by swing
  (value of shifting, F15).

Outputs: results/provenance_{bas,lp,summary}.csv, figures/provenance.png and
paper/sigmetrics/fig_provenance.pdf. `--plot-only` redraws from the CSVs.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from carbonshift import provenance as pv
from carbonshift.facility import capped_shift

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
PHI = 0.5
CAPS = {"h=0.2": 1.2, "h=0.42": 1.42, "uncapped": np.inf}
MIN_SD = 1.0  # g/kWh: below this a signal has no daily shape (hydro BAs with a constant factor)


def run():
    em, eia = pv.paired_signals(2022)
    rows = []
    for ba in em.columns:
        a, b = em[ba].to_numpy(), eia[ba].to_numpy()
        for name, cap in CAPS.items():
            r = pv.cross_evaluate(a, b, PHI, cap)
            rows.append(dict(
                ba=ba, cap=name, mean_em=a.mean(), mean_eia=b.mean(), sd_em=r.sd_a, sd_eia=r.sd_b, rho=r.rho,
                own_em=r.own_a, own_eia=r.own_b, em_on_eia=r.a_on_b, eia_on_em=r.b_on_a,
            ))
    df = pd.DataFrame(rows)
    shaped = (df.sd_em > MIN_SD) & (df.sd_eia > MIN_SD)
    df["shaped"] = shaped
    df["pct_em"] = 100 * df.own_em / (PHI * df.mean_em)
    df["pct_eia"] = 100 * df.own_eia / (PHI * df.mean_eia)
    df["confirmed_em"] = df.em_on_eia / df.own_em          # EIA confirms this share of EM's claim
    df["kept_eia"] = df.em_on_eia / df.own_eia              # share of EIA's optimum got by scheduling on EM
    df["confirmed_eia"] = df.eia_on_em / df.own_eia
    df["kept_em"] = df.eia_on_em / df.own_em
    df["beta_em"] = df.rho * df.sd_eia / df.sd_em
    df["claim_ratio"] = df.own_em / df.own_eia              # how much bigger EM's reported saving is
    df.to_csv(OUT / "provenance_bas.csv", index=False)

    # Independent route: sliding-window LP, h = 0.2.
    lp = []
    for ba in em.columns:
        a, b = em[ba].to_numpy(), eia[ba].to_numpy()
        if a.std() < MIN_SD or b.std() < MIN_SD:
            continue
        ra = capped_shift(a, PHI, 1.2, 24)
        rb = capped_shift(b, PHI, 1.2, 24)
        base = 1 - PHI
        def sav(c, served):
            return float((c * (base + PHI)).mean() - (c * (base + served)).mean())
        lp.append(dict(ba=ba, own_em=sav(a, ra.served), own_eia=sav(b, rb.served),
                       em_on_eia=sav(b, ra.served), eia_on_em=sav(a, rb.served)))
    lp = pd.DataFrame(lp)
    lp["kept_eia"] = lp.em_on_eia / lp.own_eia
    lp["confirmed_em"] = lp.em_on_eia / lp.own_em
    lp["kept_em"] = lp.eia_on_em / lp.own_em
    lp.to_csv(OUT / "provenance_lp.csv", index=False)
    pairs_table(em, eia)
    return df, lp


def pairs_table(em, eia):
    """Every pair of the three signals (EM average, EIA average, EIA marginal) at h = 0.2, both directions."""
    from carbonshift.eia930 import signals
    s = signals().assign(utc=lambda d: pd.to_datetime(d["utc"]))
    mef = s.pivot(index="utc", columns="ba", values="marginal").loc[eia.index, eia.columns]
    sig = {"em_avg": em, "eia_avg": eia, "eia_mef": mef}
    rows = []
    for ka, kb in [("em_avg", "eia_avg"), ("em_avg", "eia_mef"), ("eia_avg", "eia_mef")]:
        for ba in eia.columns:
            a, b = sig[ka][ba].to_numpy(), sig[kb][ba].to_numpy()
            if a.std() < MIN_SD or b.std() < MIN_SD:
                continue
            r = pv.cross_evaluate(a, b, PHI, 1.2)
            rows.append(dict(pair=f"{ka}|{kb}", ba=ba, rho=r.rho, kept_b=r.kept_b, kept_a=r.b_on_a / r.own_a))
    pd.DataFrame(rows).to_csv(OUT / "provenance_pairs.csv", index=False)


def summarise(df, lp):
    s = []
    for name in CAPS:
        d = df[(df.cap == name) & df.shaped]
        s.append(dict(
            cap=name, n=len(d),
            median_rho=d.rho.median(),
            median_kept_eia=d.kept_eia.median(), median_kept_em=d.kept_em.median(),
            median_confirmed_em=d.confirmed_em.median(), median_confirmed_eia=d.confirmed_eia.median(),
            p10_kept_eia=d.kept_eia.quantile(0.1), min_kept_eia=d.kept_eia.min(),
            n_kept_below_half=int(((d.kept_eia < 0.5) | (d.kept_em < 0.5)).sum()),
            median_claim_ratio=d.claim_ratio.median(),
            p10_claim_ratio=d.claim_ratio.quantile(0.1), p90_claim_ratio=d.claim_ratio.quantile(0.9),
            n_claim_off_by_1_5x=int(((d.claim_ratio > 1.5) | (d.claim_ratio < 1 / 1.5)).sum()),
            corr_kept_rho=np.corrcoef(d.kept_eia, d.rho)[0, 1],
            mae_kept_vs_rho=float((d.kept_eia - d.rho).abs().median()),
            corr_confirmed_beta=np.corrcoef(d.confirmed_em, d.beta_em)[0, 1],
            mae_confirmed_vs_beta=float((d.confirmed_em - d.beta_em).abs().median()),
            spearman_level=d.mean_em.corr(d.mean_eia, method="spearman"),
            spearman_swing=d.sd_em.corr(d.sd_eia, method="spearman"),
            spearman_saving=d.own_em.corr(d.own_eia, method="spearman"),
        ))
    s = pd.DataFrame(s)
    d = df[(df.cap == "h=0.2") & df.shaped].set_index("ba")
    j = lp.set_index("ba").join(d[["kept_eia", "confirmed_em", "kept_em"]].add_suffix("_wf"))
    s["lp_median_kept_eia"] = [lp.kept_eia.median() if c == "h=0.2" else np.nan for c in s.cap]
    s["lp_median_confirmed_em"] = [lp.confirmed_em.median() if c == "h=0.2" else np.nan for c in s.cap]
    s["lp_vs_wf_max_abs_kept"] = [float((j.kept_eia - j.kept_eia_wf).abs().max()) if c == "h=0.2" else np.nan for c in s.cap]
    s.to_csv(OUT / "provenance_summary.csv", index=False)
    p = pd.read_csv(OUT / "provenance_pairs.csv")
    k = pd.concat([p.kept_b, p.kept_a]).to_numpy()
    r = np.concatenate([p.rho, p.rho])
    slope, icpt = np.polyfit(r, k, 1)
    pp = p.groupby("pair").agg(n=("rho", "size"), median_rho=("rho", "median"), median_kept_b=("kept_b", "median"),
                               median_kept_a=("kept_a", "median"),
                               mae=("rho", lambda x: float(np.median(np.abs(np.r_[p.loc[x.index, "kept_b"] - x, p.loc[x.index, "kept_a"] - x])))))
    pp.loc["pooled"] = [len(k), np.median(r), np.median(k), np.nan, float(np.median(np.abs(k - r)))]
    pp["pooled_slope"], pp["pooled_intercept"], pp["pooled_r2"] = slope, icpt, np.corrcoef(r, k)[0, 1] ** 2
    pp.to_csv(OUT / "provenance_pairs_summary.csv")
    print(pp.round(3).to_string())
    return s


def plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[(df.cap == "h=0.2") & df.shaped]
    p = pd.read_csv(OUT / "provenance_pairs.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    style = {"em_avg|eia_avg": ("#1f6f8b", "o", "EM avg vs EIA avg"),
             "em_avg|eia_mef": ("#c4572e", "^", "EM avg vs marginal"),
             "eia_avg|eia_mef": ("#6a994e", "s", "EIA avg vs marginal")}
    for pair, (col, mk, lab) in style.items():
        q = p[p.pair == pair]
        ax[0].scatter(np.r_[q.rho, q.rho], np.r_[q.kept_b, q.kept_a], s=8, color=col, marker=mk, label=lab, alpha=0.8)
    g = np.linspace(p.rho.min(), 1, 2)
    ax[0].plot(g, g, color="0.3", lw=0.8, ls="--", label="kept = ρ")
    ax[0].set_xlabel("ρ, correlation of within-day shapes")
    ax[0].set_ylabel("share of other's optimum kept")
    ax[0].legend(fontsize=6, frameon=False)
    r = np.sort(d.claim_ratio.to_numpy())
    ax[1].plot(r, np.arange(1, len(r) + 1) / len(r), color="#1f6f8b", drawstyle="steps-post")
    ax[1].axvline(1, color="0.4", lw=0.8, ls="--")
    ax[1].set_xscale("log")
    ax[1].set_xticks([0.5, 1, 2, 3])
    ax[1].set_xticklabels(["0.5", "1", "2", "3"])
    ax[1].minorticks_off()
    ax[1].set_xlabel("reported saving, EM ÷ EIA (own schedules)")
    ax[1].set_ylabel("share of BAs")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "provenance.png", dpi=200)
    fig.savefig(ROOT / "paper" / "sigmetrics" / "fig_provenance.pdf")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        df, lp = pd.read_csv(OUT / "provenance_bas.csv"), pd.read_csv(OUT / "provenance_lp.csv")
    else:
        df, lp = run()
    s = summarise(df, lp)
    pd.set_option("display.width", 250)
    print(s.T.to_string())
    plot(df)
