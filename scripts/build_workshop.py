"""Two-column workshop paper (HotCarbon-style, ~5 pages) on carbon signals for power-capped data centres.

Every number is read from results/*.csv; the two figures are drawn here at column width.

    python scripts/build_workshop.py      -> carbonshift_workshop.pdf (project root)

Inputs come from scripts/marginal_experiments.py, hedge_experiments.py,
battery_marginal_experiments.py and diurnal_load_experiments.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, FrameBreak, Image, KeepTogether, NextPageTemplate, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

from carbonshift import plotting as P

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "carbonshift_workshop.pdf"

PAGE_W, PAGE_H = letter
MARGIN, GUTTER = 0.75 * inch, 0.33 * inch
COL_W = (PAGE_W - 2 * MARGIN - GUTTER) / 2
TITLE_H = 0.95 * inch

REFS = [
    "T. Sukprasert, A. Souza, N. Bashir, D. Irwin, P. Shenoy. On the Limitations of Carbon-Aware Temporal and Spatial Workload Shifting in the Cloud. EuroSys, 2024.",
    "T. Sukprasert, N. Bashir, A. Souza, D. Irwin, P. Shenoy. On the Implications of Choosing Average versus Marginal Carbon Intensity Signals on Carbon-aware Optimizations. ACM e-Energy, 2024.",
    "K. Siler-Evans, I. L. Azevedo, M. G. Morgan. Marginal Emissions Factors for the U.S. Electricity System. Environ. Sci. Technol. 46(9), 2012.",
    "U.S. Energy Information Administration. Hourly Electric Grid Monitor (Form EIA-930), 2022.",
    "A. Radovanovic, R. Koningstein, et al. Carbon-Aware Computing for Datacenters. arXiv:2106.11750, 2021.",
    "E. Hittinger, I. L. Azevedo. Bulk Energy Storage Increases United States Electricity System Emissions. Environ. Sci. Technol. 49(5), 2015.",
    "V. Sakalkar et al. Data Center Power Oversubscription with a Medium Voltage Power Plane and Priority-Aware Capping. ASPLOS, 2020 (Google PowerData2019 traces).",
    "W. A. Hanafy, L. Wu, D. Irwin, P. Shenoy. CarbonFlex: Enabling Carbon-aware Provisioning and Scheduling for Cloud Clusters. arXiv:2505.18357, 2025.",
    "A. Souza, N. Bashir, J. Murillo, W. Hanafy, Q. Liang, D. Irwin, P. Shenoy. Ecovisor: A Virtual Energy System for Carbon-Efficient Applications. ASPLOS, 2023.",
    "P. Wiesner, I. Behnke, D. Scheinert, K. Gontarska, L. Thamsen. Let's Wait Awhile: How Temporal Workload Shifting Can Reduce Carbon Emissions in the Cloud. Middleware, 2021.",
    "D. Maji, P. Shenoy, R. K. Sitaraman. CarbonCast: Multi-Day Forecasting of Grid Carbon Intensity. BuildSys, 2022.",
    "A. Hall, F. Micheli, G. Belgioioso, A. Radovanovic, F. Dörfler. Carbon-Aware Computing for Data Centers with Probabilistic Performance Guarantees. arXiv:2410.21510, 2024.",
]


def load() -> dict:
    n = {"ms": pd.read_csv(RES / "marginal_summary.csv", index_col=0)["value"],
         "hs": pd.read_csv(RES / "hedge_summary.csv", index_col=0)["value"],
         "ho": pd.read_csv(RES / "hedge_oos_pooled.csv", index_col=0),
         "hf": pd.read_csv(RES / "hedge_frontier_per_ba.csv"),
         "hob": pd.read_csv(RES / "hedge_oos_per_ba.csv"),
         "bm": pd.read_csv(RES / "battery_marginal_summary.csv"),
         "bo": pd.read_csv(RES / "battery_marginal_split_summary.csv", index_col=0),
         "ds": pd.read_csv(RES / "diurnal_summary.csv", index_col=0)["value"]}
    ok = n["hf"].groupby("ba").filter(lambda d: d.avg_saving_g.max() > 0.5 and d.marg_saving_g.max() > 0.5)
    best = ok.groupby("ba")[["avg_saving_g", "marg_saving_g"]].max().sum()
    pool = ok.groupby("w_avg")[["avg_saving_g", "marg_saving_g"]].sum()
    n["ok"], n["pool"] = ok, pd.DataFrame({"avg": pool.avg_saving_g / best.avg_saving_g, "marg": pool.marg_saving_g / best.marg_saving_g})
    o = n["hob"]
    w0 = o[o.w_avg == 0].set_index("ba").marg_out_of_sample_g
    best = o.loc[o.groupby("ba").marg_out_of_sample_g.idxmax()].set_index("ba")
    gain = (best.marg_out_of_sample_g / w0 - 1)[best.w_avg > 0] * 100
    n["shrink_bas"], n["shrink_gain_median"] = int((best.w_avg > 0).sum()), float(gain.median())
    return n


def bmv(n, e, on, col):
    b = n["bm"]
    return float(b[(b.energy_h.round(3) == round(e, 3)) & (b.dispatch_on == on)][col].iloc[0])


# ----------------------------------------------------------------------------- figures
def figures(n) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt
    P.setup()
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8})

    ho, ok, pool = n["ho"], n["ok"], n["pool"]
    fig, ax = plt.subplots(figsize=(3.3, 2.45))
    for _, d in ok.groupby("ba"):
        ax.plot(d.avg_share * 100, d.marg_share * 100, color=P.INK_2, lw=0.4, alpha=0.3)
    ax.plot(pool.avg * 100, pool.marg * 100, color=P.BLUE, lw=1.8,
            marker="o", ms=2.5, label="pooled, in-sample")
    ax.plot(ho.avg_share * 100, ho.marg_oos_vs_marginal_only_oos * 100, color=P.ORANGE, lw=1.4, ls="--", marker="s", ms=2.2,
            label="pooled, held-out MEFs")
    ax.annotate("w = 0.7", (pool.loc[0.7, "avg"] * 100, pool.loc[0.7, "marg"] * 100), xytext=(-42, -20),
                textcoords="offset points", fontsize=7, arrowprops={"arrowstyle": "-", "lw": 0.6, "color": P.INK_2})
    ax.axhline(0, color=P.INK_2, lw=0.5)
    ax.axvline(0, color=P.INK_2, lw=0.5)
    ax.set(xlim=(-60, 105), ylim=(-60, 105), xlabel="Average-ledger saving kept (%)", ylabel="Marginal saving kept (%)")
    ax.legend(frameon=False, loc="lower left")
    f1 = P.save(fig, "ws_frontier")

    bo = n["bo"]
    sizes = [(1 / 6, "battery_0.167h", "10 min"), (1.0, "battery_1h", "1 h"), (4.0, "battery_4h", "4 h")]
    x = np.arange(len(sizes))
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    wd = 0.26
    ax.bar(x - wd, [bmv(n, e, "average", "avg_red_median") for e, _, _ in sizes], wd, color=P.INK_2, label="average ledger, average dispatch")
    ax.bar(x, [bo.loc[k, "oos_pct_median"] for _, k, _ in sizes], wd, color=P.BLUE, label="marginal ledger, marginal dispatch (held-out)")
    ax.bar(x + wd, [bmv(n, e, "average", "marg_red_median") for e, _, _ in sizes], wd, color=P.ORANGE, label="marginal ledger, average dispatch")
    ax.axhline(0, color=P.INK_2, lw=0.6)
    ax.set_xticks(x, [s[2] for s in sizes])
    top = max(bo.loc[k, "oos_pct_median"] for _, k, _ in sizes)
    ax.set(ylabel="Median emission cut (%)", xlabel="Battery energy", ylim=(None, top * 1.75))
    ax.legend(frameon=False, loc="upper left")
    f2 = P.save(fig, "ws_battery")
    plt.rcdefaults()
    return f1, f2


# ----------------------------------------------------------------------------- shared layout
def styles() -> dict:
    """Paragraph styles and builders shared by both workshop papers."""
    body = ParagraphStyle("b", fontName="Times-Roman", fontSize=9.5, leading=11.4, alignment=TA_JUSTIFY, spaceAfter=4.5)
    h1 = ParagraphStyle("h1", parent=body, fontName="Times-Bold", fontSize=10.5, leading=12.5, spaceBefore=6, spaceAfter=3, alignment=0, keepWithNext=1)
    cap = ParagraphStyle("c", parent=body, fontSize=8.3, leading=9.8, spaceAfter=7)
    bul = ParagraphStyle("bl", parent=body, leftIndent=10, bulletIndent=1, spaceAfter=2.5)
    S = {"body": body, "cap": cap,
         "title": ParagraphStyle("t", parent=body, fontName="Times-Bold", fontSize=15.5, leading=18.5, alignment=TA_CENTER, spaceAfter=5),
         "auth": ParagraphStyle("a", parent=body, fontName="Times-Roman", fontSize=10.5, alignment=TA_CENTER, spaceAfter=1),
         "note": ParagraphStyle("n", parent=body, fontName="Times-Italic", fontSize=8.6, alignment=TA_CENTER),
         "ref": ParagraphStyle("r", parent=body, fontSize=8.0, leading=9.4, spaceAfter=2, leftIndent=12, firstLineIndent=-12, alignment=0)}
    S["p"] = lambda t: Paragraph(t, body)
    S["h"] = lambda t: Paragraph(t, h1)
    S["b"] = lambda t: Paragraph(t, bul, bulletText="•")

    def fig(path, caption, num):
        img = Image(str(path), width=COL_W, height=COL_W * _aspect(path))
        return KeepTogether([img, Paragraph(f"<b>Figure {num}.</b> {caption}", cap)])

    def table(rows, caption, num, widths):
        t = Table(rows, colWidths=widths, hAlign="CENTER")
        t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Times-Roman", 8.2), ("FONT", (0, 0), (-1, 0), "Times-Bold", 8.2),
                               ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                               ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black), ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                               ("TOPPADDING", (0, 0), (-1, -1), 1.4), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.4)]))
        return KeepTogether([Paragraph(f"<b>Table {num}.</b> {caption}", cap), t, Spacer(1, 6)])

    S["fig"], S["table"] = fig, table
    return S


def frames() -> tuple[list, list]:
    """First page: full-width title band over two columns; later pages: two columns."""
    body_h = PAGE_H - 2 * MARGIN
    col = lambda i, top, hgt: Frame(MARGIN + i * (COL_W + GUTTER), top, COL_W, hgt, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)  # noqa: E731
    first = [Frame(MARGIN, PAGE_H - MARGIN - TITLE_H, PAGE_W - 2 * MARGIN, TITLE_H, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
             col(0, MARGIN, body_h - TITLE_H - 8), col(1, MARGIN, body_h - TITLE_H - 8)]
    return first, [col(0, MARGIN, body_h), col(1, MARGIN, body_h)]


def page_no(c, d):
    c.saveState()
    c.setFont("Times-Roman", 8.5)
    c.drawCentredString(PAGE_W / 2, 0.45 * inch, str(d.page))
    c.restoreState()


# ----------------------------------------------------------------------------- document
def build() -> None:
    n = load()
    ms, hs, ho, bo, ds = n["ms"], n["hs"], n["ho"], n["bo"], n["ds"]
    f1, f2 = figures(n)
    pc = lambda v, d=0: f"{v * 100:.{d}f}%"  # noqa: E731
    lo_hi = lambda a, b: f"{min(a, b) * 100:.0f}–{max(a, b) * 100:.0f}%"  # noqa: E731

    S = styles()
    body, title, auth, note, ref = S["body"], S["title"], S["auth"], S["note"], S["ref"]
    p, h, b, fig, table = S["p"], S["h"], S["b"], S["fig"], S["table"]

    sh = lambda wv, col: float(ho.loc[wv, col])  # noqa: E731
    pool_a, pool_m = n["pool"].avg.to_dict(), n["pool"].marg.to_dict()
    bas = int(ms["bas"])
    nb = int(hs["bas"])
    ub10, ua10 = bo.loc["battery_0.167h", "oos_pct_median"], bmv(n, 1 / 6, "average", "avg_red_median")

    story = [
        Paragraph("Blend, Don’t Choose: Average and Marginal Carbon Signals for Power-Capped Data Centres", title),
        Paragraph("Naveen Kumar Lakkaram", auth),
        Paragraph("Workshop draft, November 2026 · code and data: github.com/lakkaramnaveen/carbonshift", note),
        FrameBreak(),
        NextPageTemplate("later"),
        Paragraph(
            "<b>Abstract.</b> Carbon-aware schedulers must pick a signal: the <i>average</i> grid intensity that emissions are reported in, or the "
            "<i>marginal</i> intensity that a change in load actually causes. Using EIA-930 data for "
            f"{bas} U.S. balancing authorities (BAs) and a power-capped facility model, we show that a schedule optimised on either signal "
            f"captures only {lo_hi(pool_m[1.0], pool_a[0.0])} of the other's saving and makes it worse in {hs['bas_avg_schedule_raises_marginal_pct']:.0f}–"
            f"{hs['bas_marginal_schedule_raises_avg_pct']:.0f}% of BAs. Yet the Pareto frontier between them is strongly concave: scheduling on "
            f"0.7·average + 0.3·marginal keeps {pc(pool_a[0.7])} of the average and {pc(pool_m[0.7])} of the marginal optimum, and this holds on "
            "held-out marginal estimates. Batteries are the exception. On the marginal ledger a UPS is worth about "
            f"{ub10 / ua10:.0f}× more than on the average ledger, but round-trip losses make storage unable to serve both ledgers, and blending does not help. "
            f"A split-half test shows that {100 - bo.loc['shift_capped', 'retained_pooled'] * 100:.0f}–{100 - bo.loc['battery_0.167h', 'retained_pooled'] * 100:.0f}% "
            "of in-sample marginal savings do not survive on held-out marginal factors.", body),

        h("1 Introduction"),
        p("Carbon-aware computing moves flexible work to hours and places where electricity is cleaner [1, 5, 10]. Every such system needs a signal. "
          "Corporate reporting under the GHG Protocol uses <i>average</i> intensity: the emissions of all generation divided by its output. But a "
          "scheduler's decision changes emissions at the <i>margin</i>: an extra kWh is served by whichever plant is dispatched next, often gas or coal even "
          "where the average is low [3]. The two signals are weakly correlated and can lead optimisers in opposite directions [2]."),
        p("Most studies pick one. We ask whether a facility operator, who must report on the average ledger but wants to cut real emissions, has to choose. "
          "We study this in the setting that matters for a data centre: shifting is limited by the facility's electrical capacity, so shifted work competes for "
          f"the same headroom. Google's released power traces put the median facility at {ds['mean_util_median'] * 100:.0f}% of its power rating "
          f"(+{ds['headroom_median_pct']:.0f}% headroom) [7], and we use that cap throughout."),
        p("<b>Contributions.</b>"),
        b(f"Under a power cap, schedules optimised on one signal capture {lo_hi(pool_m[1.0], pool_a[0.0])} of the other's saving and raise its emissions in "
          f"{hs['bas_avg_schedule_raises_marginal_pct']:.0f}–{hs['bas_marginal_schedule_raises_avg_pct']:.0f}% of BAs (§3)."),
        b(f"A blended objective recovers most of both: the frontier is strongly concave, and w = 0.7 keeps about four-fifths of each, in and out of sample (§4)."),
        b("Storage is different: a UPS battery is worth several times more on the marginal ledger, but no blend reconciles the ledgers (§5)."),
        b("A split-half estimate of marginal factors that separates real structure from noise, showing that in-sample marginal savings are overstated by a quarter to a half (§6)."),

        h("2 Setting and Method"),
        p("<b>Facility model.</b> A facility draws a mean load of 1. A share φ = 0.5 is flexible and may run up to 24 h after it arrives; the rest runs on "
          "arrival. Hourly grid draw is capped at 1/0.70 ≈ 1.43× the mean. We solve the carbon-optimal schedule as a linear program in cumulative-flow "
          "form over the whole year (8,760 hours), which is exact for fluid work and tested against brute force. Savings are reported per kWh of mean load."),
        p("<b>Signals.</b> From EIA-930 hourly net generation by fuel [4] we build both signals for each U.S. BA. The <i>average</i> signal weights "
          "lifecycle factors by generation; it matches Electricity Maps at r = "
          f"{ms['validation_mean_r']:.2f} (annual means) and a median {ms['validation_hourly_r_median']:.2f} (hourly). The <i>marginal</i> signal follows "
          "Siler-Evans et al. [3]: the slope of hour-to-hour emission changes on generation changes, estimated per season × hour-of-day bin."),
        p("<b>Blending.</b> Let a<sub>t</sub> and m<sub>t</sub> be the two signals. We schedule on w·a<sub>t</sub> + (1−w)·m<sub>t</sub> for w ∈ {0, 0.1, …, 1} "
          "and score each schedule on both. Because the program is linear, weighted sums trace every supported point of the Pareto frontier. Shares are "
          "relative to each signal's own capped optimum."),
        p("<b>Held-out marginal factors.</b> Regression bins are noisy: in 14 of 53 BAs some bins are negative, and one reaches 1,400 g/kWh. An optimiser "
          "exploits such noise. We therefore re-estimate the factors on even and on odd days separately, assigning each hour-to-hour difference to the day "
          "of its later hour, and also report <i>out-of-sample</i> results: plan on one half and score on the other, averaged over both directions."),

        h("3 The Signals Disagree"),
        p(f"A capped schedule optimised on the average signal captures {pc(ms['share_of_marg_optimum_pooled'])} of the marginal-optimal saving "
          f"(median BA {pc(ms['share_of_marg_optimum_median'], 1)}) and <i>raises</i> marginal emissions in {ms['bas_where_avg_schedule_raises_marginal_pct']:.0f}% "
          f"of BAs. The reverse is as bad: the marginal-optimal schedule captures {pc(pool_a[0.0])} of the average-ledger saving and raises average-ledger "
          f"emissions in {hs['bas_marginal_schedule_raises_avg_pct']:.0f}% of BAs. The structure of the problem survives, though. The value of an extra kW of "
          f"headroom tracks the signal's absolute daily swing on both signals (Spearman ρ = {ms['rho_shadow_marg_vs_std_marg']:.2f} for marginal), "
          f"not its mean (ρ = {ms['rho_shadow_marg_vs_mean_marg']:.2f}). This follows from an affine-invariance property of the program: adding a "
          "constant to the signal leaves the schedule unchanged. The two signals agree on <i>how much</i> flexibility is worth, but not on <i>when</i> to use it."),

        h("4 Blend, Don’t Choose"),
        p(f"Figure 1 and Table 1 show the frontier. It is strongly concave. Moving from w = 1 to w = 0.9 gives up {100 - pool_a[0.9] * 100:.0f}% "
          f"of the average saving but raises the marginal saving from {pc(pool_m[1.0])} to {pc(pool_m[0.9])}. "
          f"Moving from w = 0 to 0.3 gives up only {100 - pool_m[0.3] * 100:.0f}% of the marginal saving and raises the average one six-fold. At w = 0.7 "
          f"both ledgers keep about four-fifths. Per BA, the best single weight keeps a median {pc(hs['balanced_worst_share_median'])} of <i>both</i> optima; "
          f"it keeps at least half of both in {hs['bas_with_some_w_both_ge_50pct_pct']:.0f}% of BAs."),
        fig(f1, "Pareto frontier between average-ledger and marginal savings as the weight w on the average signal goes from 0 (top left) to 1 "
                f"(bottom right). Grey: {nb} BAs. Blue: pooled. Orange: pooled with marginal factors estimated on different days than they are scored on.", 1),
        table([["w (weight on average)", "0", "0.3", "0.5", "0.7", "1"],
               ["Average saving kept"] + [pc(pool_a[x]) for x in (0.0, 0.3, 0.5, 0.7, 1.0)],
               ["Marginal saving kept"] + [pc(pool_m[x]) for x in (0.0, 0.3, 0.5, 0.7, 1.0)],
               ["Marginal, held-out*"] + [pc(sh(x, "marg_oos_vs_marginal_only_oos")) for x in (0.0, 0.3, 0.5, 0.7, 1.0)]],
              f"Pooled share of each optimum kept ({nb} BAs). *Relative to the marginal-only plan under the same held-out test.", 1,
              [1.25 * inch] + [0.36 * inch] * 5),
        p("<b>Why it works.</b> With a cap and a slack window, each signal leaves many hours nearly tied: shifting a unit between them changes that signal's "
          "emissions very little. The other signal breaks those ties almost for free. Only near the ends of the frontier must one signal's real preferences be "
          "overridden, and that is where the cost appears."),
        p(f"<b>Out of sample.</b> With marginal factors estimated on different days than they are scored on, w = 0.7 keeps {pc(sh(0.7, 'avg_share'))} "
          f"of the average optimum and {pc(sh(0.7, 'marg_oos_vs_marginal_only_oos'))} of what the marginal-only plan achieves; w = 0.3 keeps "
          f"{pc(sh(0.3, 'marg_oos_vs_marginal_only_oos'))}. We also tested whether mixing in the smooth average signal <i>regularises</i> the noisy marginal one. "
          f"It does not, in aggregate: pooled, the held-out marginal saving peaks at w = 0. In {n['shrink_bas']} of {nb} BAs a weight above zero does best, "
          f"but by a median of only {n['shrink_gain_median']:.1f}%."),

        h("5 Batteries Cannot Blend"),
        p("A UPS battery behind the load can also shift carbon, by charging in clean hours and discharging in dirty ones. Its arbitrage is pure incremental "
          "load, so the marginal ledger is the natural one. We solve a cyclic dispatch program (90% round trip, no reserve) for each BA."),
        fig(f2, "Median emission cut from carbon-optimal battery dispatch across BAs. A battery dispatched on the average signal raises marginal emissions "
                "(orange, below zero).", 2),
        p(f"On the average ledger a 10-minute UPS cuts emissions by a median {ua10:.2f}%: not worth optimising. On the marginal ledger, scored on held-out "
          f"factors, the same UPS cuts {ub10:.2f}%, and a 4-hour battery {bo.loc['battery_4h', 'oos_pct_median']:.1f}% against "
          f"{bmv(n, 4, 'average', 'avg_red_median'):.1f}% (Figure 2). But the ledgers conflict more sharply than for load shifting. Average-signal dispatch raises "
          f"marginal emissions in {bmv(n, 1 / 6, 'average', 'bas_raising_marg_pct'):.0f}% of BAs; marginal-signal dispatch raises average-ledger emissions in "
          f"{bmv(n, 1 / 6, 'marginal', 'bas_raising_avg_pct'):.0f}%; and the w = 0.7 blend still raises the average ledger in "
          f"{bmv(n, 4, 'blend', 'bas_raising_avg_pct'):.0f}–{bmv(n, 1 / 6, 'blend', 'bas_raising_avg_pct'):.0f}% of BAs."),
        p("The reason is round-trip loss. Load shifting moves energy without creating any, so a tie-breaking move is nearly free. A battery cycle always "
          "draws more energy than it returns, so a cycle chosen by one signal is a net cost on the other ledger unless that ledger also sees a large enough "
          "spread. There are no free ties to break. This mirrors the finding that bulk storage raises U.S. system emissions when it arbitrages prices [6]."),

        h("6 How Much of a Marginal Saving Is Real?"),
        p("Split-half marginal factors agree closely at the level of BA means (r = 0.999) but only moderately hour by hour (median r = 0.80). Table 2 shows "
          "what this does to marginal-optimal plans."),
        table([["Plan", "In-sample", "Held-out", "Kept"],
               ["Capped load shifting", f"{bo.loc['shift_capped', 'ins_pct_median']:.1f}%", f"{bo.loc['shift_capped', 'oos_pct_median']:.1f}%",
                pc(bo.loc["shift_capped", "retained_pooled"])],
               ["10-min UPS", f"{bo.loc['battery_0.167h', 'ins_pct_median']:.2f}%", f"{ub10:.2f}%", pc(bo.loc["battery_0.167h", "retained_pooled"])],
               ["1 h battery", f"{bo.loc['battery_1h', 'ins_pct_median']:.1f}%", f"{bo.loc['battery_1h', 'oos_pct_median']:.1f}%",
                pc(bo.loc["battery_1h", "retained_pooled"])],
               ["4 h battery", f"{bo.loc['battery_4h', 'ins_pct_median']:.1f}%", f"{bo.loc['battery_4h', 'oos_pct_median']:.1f}%",
                pc(bo.loc["battery_4h", "retained_pooled"])]],
              "Marginal emission cut (median BA) when the plan is scored on the factors it was optimised on (in-sample) versus factors estimated on other "
              "days (held-out); \"Kept\" is the pooled ratio.", 2, [1.2 * inch, 0.62 * inch, 0.62 * inch, 0.5 * inch]),
        p("Batteries, which act on every hour-to-hour difference, lose about half of their apparent saving; load shifting, which acts on broader patterns, "
          "loses a quarter. Any study that optimises and evaluates on the same regression-based marginal factors overstates the saving by a similar amount. "
          "Day-parity splitting is a cheap guard: it needs no extra data and keeps every season and hour bin populated."),

        h("7 Related Work"),
        p("Sukprasert et al. [2] compare average and marginal signals across 65 regions and show that mixing signals for scheduling and reporting backfires; "
          "we add the capped-facility setting, the blended frontier, storage, and the noise check. Upper bounds on temporal and spatial shifting [1, 10] and "
          "forecasting [11] use average intensity. Google's carbon-intelligent computing caps flexible capacity hour by hour [5, 12], and CarbonFlex [8] and "
          "Ecovisor [9] expose energy and carbon to applications; none studies the two-ledger trade-off. Grid-scale storage studies use marginal factors [6], "
          "but not for data-centre UPS batteries or both ledgers."),

        h("8 Discussion and Limitations"),
        p("<b>Implications.</b> A carbon-aware scheduler does not need to resolve the average-versus-marginal debate before acting. A blended objective with "
          "a modest weight on the reported signal protects the reported number while delivering most of the real reduction. The weight is a policy knob, and "
          "the frontier tells an operator its price. For storage, by contrast, the ledger must be chosen, and claims of UPS-based carbon savings should state which one."),
        p("<b>Limitations.</b> Marginal factors are season × hour regressions on 2022 data for U.S. BAs only; imports are ignored; weights are applied to raw "
          "g/kWh values, so their meaning depends on the signals' scales; flexible work is fluid and forecasts are perfect, so all savings are upper bounds."),

        h("References"),
    ] + [Paragraph(f"[{i}] {r}", ref) for i, r in enumerate(REFS, 1)]

    doc = BaseDocTemplate(str(OUT), pagesize=letter, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN,
                          title="Blend, Don't Choose: Average and Marginal Carbon Signals for Power-Capped Data Centres",
                          author="Naveen Kumar Lakkaram", creator="carbonshift build_workshop.py", subject="Carbon-aware computing")
    first, later = frames()

    doc.addPageTemplates([PageTemplate("first", first, onPage=page_no), PageTemplate("later", later, onPage=page_no)])
    doc.build(story)
    print(f"wrote {OUT.name}  ({doc.page} pages)")


def _aspect(path: Path) -> float:
    from PIL import Image as PImage
    with PImage.open(path) as im:
        return im.height / im.width


if __name__ == "__main__":
    build()
