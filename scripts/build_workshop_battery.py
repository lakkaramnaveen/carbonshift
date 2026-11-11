"""Second workshop paper: the virtual-battery theory of power-capped carbon-aware shifting (F21-F23).

Every number is read from results/*.csv; the three figures are drawn here at column width.

    python scripts/build_workshop_battery.py   -> carbonshift_battery_workshop.pdf (project root)

Inputs come from scripts/waterfill_experiments.py, virtual_battery_experiments.py,
exchange_experiments.py and diurnal_load_experiments.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import BaseDocTemplate, FrameBreak, NextPageTemplate, PageTemplate, Paragraph

from build_workshop import COL_W, MARGIN, PAGE_H, PAGE_W, ROOT, frames, page_no, styles
from carbonshift import plotting as P
from carbonshift.waterfill import retained_sinc

RES = ROOT / "results"
OUT = ROOT / "carbonshift_battery_workshop.pdf"
TITLE = "Your Data Centre Is Already a Battery: A Closed-Form Theory of Carbon-Aware Shifting under Power Caps"

REFS = [
    "T. Sukprasert, A. Souza, N. Bashir, D. Irwin, P. Shenoy. On the Limitations of Carbon-Aware Temporal and Spatial Workload Shifting in the Cloud. EuroSys, 2024.",
    "A. Radovanovic, R. Koningstein, et al. Carbon-Aware Computing for Datacenters. arXiv:2106.11750, 2021.",
    "A. Souza, N. Bashir, J. Murillo, W. Hanafy, Q. Liang, D. Irwin, P. Shenoy. Ecovisor: A Virtual Energy System for Carbon-Efficient Applications. ASPLOS, 2023.",
    "W. A. Hanafy, Q. Liang, N. Bashir, D. Irwin, P. Shenoy. CarbonScaler: Leveraging Cloud Workload Elasticity for Optimizing Carbon-Efficiency. SIGMETRICS, 2024.",
    "P. Wiesner, I. Behnke, D. Scheinert, K. Gontarska, L. Thamsen. Let's Wait Awhile: How Temporal Workload Shifting Can Reduce Carbon Emissions in the Cloud. Middleware, 2021.",
    "V. Sakalkar et al. Data Center Power Oversubscription with a Medium Voltage Power Plane and Priority-Aware Capping. ASPLOS, 2020 (Google PowerData2019 traces).",
    "T. M. Cover, J. A. Thomas. Elements of Information Theory, 2nd ed., Ch. 9 (water-filling). Wiley, 2006.",
    "D. S. Callaway, I. A. Hiskens. Achieving Controllability of Electric Loads. Proc. IEEE 99(1), 2011.",
    "H. Hao, B. M. Sanandaji, K. Poolla, T. L. Vincent. Aggregate Flexibility of Thermostatically Controlled Loads. IEEE Trans. Power Syst. 30(1), 2015.",
    "E. Hittinger, I. L. Azevedo. Bulk Energy Storage Increases United States Electricity System Emissions. Environ. Sci. Technol. 49(5), 2015.",
    "Electricity Maps. Hourly carbon-intensity data, 2020-2022, as released with [1].",
]


def load() -> dict:
    rd = lambda f: pd.read_csv(RES / f, index_col=0)["value"]  # noqa: E731
    return {"wf": rd("waterfill_summary.csv"), "vb": rd("virtual_battery_summary.csv"), "ex": rd("exchange_summary.csv"),
            "ds": rd("diurnal_summary.csv"), "ws": pd.read_csv(RES / "waterfill_saving_per_zone.csv"),
            "hz": pd.read_csv(RES / "virtual_battery_horizon_law.csv"), "xz": pd.read_csv(RES / "exchange_per_zone.csv")}


def figures(n) -> list[Path]:
    import matplotlib.pyplot as plt
    P.setup()
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8})
    out = []

    # 1. universal retention curve
    ws = n["ws"]
    ws = ws[(ws.slack == 24) & np.isfinite(ws.cap) & (ws.cap > 1) & (ws.ideal > 1)]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for phi, col in ((0.25, P.ORANGE), (0.5, P.BLUE), (0.75, P.AQUA)):
        g = ws[ws.phi == phi].groupby("h_over_phi").retained.median()
        ax.plot(g.index, g.values * 100, color=col, marker="o", ms=2.5, lw=1.2, label=f"LP median, φ = {phi}")
    x = np.linspace(0.01, 6, 200)
    ax.plot(x, retained_sinc(1.0, 1 + x) * 100, color=P.INK, lw=1.0, ls="--", label="sinc(πq), sinusoidal day")
    ax.set(xscale="log", xlabel="Headroom per unit of flexible load, h/φ", ylabel="Share of uncapped saving kept (%)", ylim=(0, 100))
    ax.legend(frameon=False, loc="lower right")
    out.append(P.save(fig, "wsb_universal"))

    # 2. effective horizon fixes the tight-cap bias
    hz = n["hz"]
    hz = hz[np.isfinite(hz.cap) & (hz.cap > 1) & (hz.lp > 0.5)]
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    for slack, ls in ((24, "-"), (168, ":")):
        g = hz[hz.slack == slack].assign(b=lambda d: (d.block_law / d.lp - 1) * 100, h=lambda d: (d.horizon_law / d.lp - 1) * 100)
        m = g.groupby("cap")[["b", "h"]].median()
        hx = (m.index - 1) * 100
        ax.plot(hx, m.b, color=P.INK_2, ls=ls, marker="o", ms=2.3, lw=1.1, label=f"fixed {slack} h blocks")
        ax.plot(hx, m.h, color=P.BLUE, ls=ls, marker="s", ms=2.3, lw=1.3, label=f"effective horizon, slack {slack} h")
    ax.axhline(0, color=P.INK_2, lw=0.5)
    ax.set(xscale="log", xlabel="Headroom h (% of mean load)", ylabel="Law vs LP, median error (%)")
    ax.legend(frameon=False, loc="lower right")
    out.append(P.save(fig, "wsb_horizon"))

    # 3. exchange rate
    xz = n["xz"]
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    caps = [1.05, 1.1, 1.2, 1.5]
    for phi, col in ((0.25, P.ORANGE), (0.5, P.BLUE)):
        g = xz[(xz.phi == phi) & xz.cap.isin(caps) & np.isfinite(xz.X_kwh_per_kw)].groupby("cap").X_kwh_per_kw
        hx = (np.array(caps) - 1) * 100
        ax.fill_between(hx, g.quantile(0.25), g.quantile(0.75), color=col, alpha=0.18, lw=0)
        ax.plot(hx, g.median(), color=col, marker="o", ms=2.5, lw=1.4, label=f"φ = {phi}")
    ax.set(xlabel="Headroom h (% of mean load)", ylabel="Equal-carbon battery\n(kWh per kW of load)")
    ax.legend(frameon=False, loc="upper left")
    out.append(P.save(fig, "wsb_exchange"))
    plt.rcdefaults()
    return out


def build() -> None:
    n = load()
    wf, vb, ex, ds = n["wf"], n["vb"], n["ex"], n["ds"]
    f1, f2, f3 = figures(n)
    S = styles()
    p, h, b, fig, table = S["p"], S["h"], S["b"], S["fig"], S["table"]
    pc = lambda v, d=0: f"{v * 100:.{d}f}%"  # noqa: E731
    X = lambda phi, cap: ex[f"X_median_phi{phi}_cap{cap}"]  # noqa: E731
    E = lambda kind, cap: ex[f"embodied_ratio_battery_over_headroom_{kind}_phi0.5_cap{cap}"]  # noqa: E731

    story = [
        Paragraph(TITLE, S["title"]),
        Paragraph("Naveen Kumar Lakkaram", S["auth"]),
        Paragraph("Workshop draft, November 2026 · code and data: github.com/lakkaramnaveen/carbonshift", S["note"]),
        FrameBreak(),
        NextPageTemplate("later"),
        Paragraph(
            "<b>Abstract.</b> Studies of carbon-aware workload shifting solve an optimisation per grid and report what comes out. We show that, once the "
            "facility's power capacity is modelled, the answers follow from a short theory. With flexible share φ, slack s and headroom h, the backlog of "
            "deferred work obeys exactly the constraints of a <i>lossless battery</i> of energy s·φ, discharge power φ and charge power h. Three laws follow, "
            f"each checked against linear programs on 123 grids: the share of the ideal saving kept depends only on h/φ (exactly); the saving is "
            f"φ·A·sinc(πφ/(φ+h)), with A the daily swing (R² {wf['saving_r2_finite_caps']:.2f} over {wf['saving_cases']:,.0f} solves); and the carbon value of "
            f"a kilowatt is a water-level area (R² {wf['value_r2']:.2f}). The battery view explains why a UPS barely helps, why tight caps plan beyond the "
            f"slack, and why signals blend for load but not for storage. Priced against a real 4 h battery, a half-flexible facility with +20% headroom is "
            f"worth {X(0.5, 1.2):.1f} kWh of storage per kW of load, at a median {E('median', 1.2):.0f}× less embodied carbon.", S["body"]),

        h("1 Introduction"),
        p("Carbon-aware schedulers defer flexible work to hours when the grid is cleaner [1, 2, 5]. Upper-bound studies assume the facility can absorb any "
          "amount of shifted work [1]. Real facilities cannot: their transformers, switchgear and UPS modules are sized for a peak, and Google's released "
          f"power traces put the median facility at {ds['mean_util_median'] * 100:.0f}% of its rating, i.e. about +{ds['headroom_median_pct']:.0f}% headroom [6]. "
          "With that cap in the model, the saving drops sharply, and the remaining saving, the value of extra capacity and the right planning horizon are "
          "usually obtained by solving a year-long linear program for every grid and parameter."),
        p("This paper replaces those solves with a theory. Its core is an exact equivalence: <b>a power-capped flexible workload is a lossless battery</b>. "
          "Electrical engineers already treat aggregations of flexible loads as virtual storage [8, 9]; we make the mapping exact for deferrable compute "
          "under a facility cap, solve the resulting problem in closed form, and test every law against the optimiser on real grid data."),
        p("<b>Contributions.</b>"),
        b("The virtual-battery theorem, with energy s·φ, discharge φ and charge h (§3)."),
        b("Closed-form laws for the saving, the share kept, the value of a kW and the peak draw, all derived from water-filling (§4)."),
        b("Three explanations of earlier empirical puzzles, one confirmed as a prediction (§5)."),
        b("A storage exchange rate pricing flexibility in kWh and embodied carbon of real batteries (§6)."),

        h("2 Model"),
        p("A facility draws a mean load of 1. A share φ is flexible and must be served within s hours of arriving; the rest runs on arrival. Grid draw is "
          "capped at 1 + h each hour. Work is fluid and forecasts are perfect. We minimise the carbon of the year's draw under hourly carbon intensity c<sub>t</sub> "
          "with a linear program in cumulative-flow form (8,760 hours), using Electricity Maps traces for 123 zones from [1] [11]. Savings are g CO2 per kWh of mean load."),

        h("3 The Virtual-Battery Theorem"),
        p("Let B<sub>t</sub> be the backlog: flexible work that has arrived but not run. The cumulative-flow constraints say exactly that "
          "0 ≤ B<sub>t</sub> ≤ s·φ (nothing waits longer than s hours) and −h ≤ ΔB<sub>t</sub> ≤ φ (one cannot defer more than arrives, nor catch up faster "
          "than the headroom). The objective is φ·Σc<sub>t</sub> − Σc<sub>t</sub>·ΔB<sub>t</sub>. With stored energy e = s·φ − B, this is precisely a battery "
          "of energy <b>s·φ</b>, discharge power <b>φ</b>, charge power <b>h</b> and <b>100% round-trip efficiency</b>: deferring work discharges it; "
          "catching up recharges it. Only the boundary condition differs (empty backlog at the start versus a cyclic state of charge), an O(1/days) term. On "
          f"real grids the two programs agree to a median {vb['equiv_median_abs_rel_diff'] * 100:.2f}% ({vb['equiv_cases']:.0f} cases). A half-flexible facility "
          "with 24 h slack holds a 12-hour battery."),

        h("4 Closed-Form Laws"),
        p("For s = 24 h, the battery is filled and emptied within a day, so the optimal policy is <b>water-filling</b> [7]: each day's 24φ units of work run at "
          "the maximum rate φ + h in the day's cleanest hours, occupying a fraction q = φ/(φ+h) of the day."),
        p("<b>(i) Homogeneity.</b> Scaling φ and h together scales the solution, so the share of the uncapped saving kept depends only on h/φ. Over "
          f"{wf['homogeneity_pairs']:,.0f} zone pairs the largest difference is {wf['homogeneity_max_abs_diff_retained']:.4f} (Figure 1)."),
        fig(f1, "Share of the uncapped saving kept, against headroom per unit of flexible load. The three flexible shares coincide (homogeneity); "
                "the dashed curve is the sinusoidal closed form.", 1),
        p("<b>(ii) Saving.</b> For a sinusoidal day of amplitude A the cleanest fraction q averages M − A·sinc(πq), so the saving is "
          f"<b>φ·A·sinc(πq)</b>. Taking A as each zone's mean daily (mean − min), φ·A predicts the uncapped saving at R² {wf['uncapped_r2_phiA_vs_lp']:.3f} and "
          f"the capped law at R² {wf['saving_r2_finite_caps']:.3f} over {wf['saving_cases']:,.0f} LP solves (median signed error {wf['saving_median_rel_err'] * 100:.1f}%)."),
        p("<b>(iii) Value of a kilowatt.</b> Differentiating the order statistics gives dS/dh as the area between the water level (the price of the marginal "
          "filled hour) and the cheaper hours. From a sort of each day, it matches the LP shadow prices at "
          f"R² {wf['value_r2']:.3f}, median ratio {wf['value_median_ratio']:.3f} ({wf['value_cases']:.0f} zone-years). For a sinusoid it is π·A·q²·|sinc′(πq)|: "
          "linear in the swing A and independent of the mean level, which is why the value of capacity tracks the absolute daily swing of the grid, not its average."),
        p(f"<b>(iv) Peak.</b> Without a cap an hour can absorb the flexible arrivals of all s + 1 hours whose window contains it, so the peak draw is 1 + s·φ: "
          f"{wf['peak_law_phi0.5']:.0f}× mean load at φ = 0.5, exactly the LP's value. The uncapped bounds of [1] are therefore also power bounds."),

        h("5 What the Battery Explains"),
        p("<b>Planning horizon.</b> The virtual battery refills at rate h, taking s·φ/h hours. At tight caps this exceeds the slack, so the optimiser plans "
          "over days: fixed s-hour water-filling underestimates the saving by "
          f"{-vb['block_law_slack24_bias_at_cap1.05'] * 100:.0f}% at +5% headroom. Water-filling over T = s·max(1, φ/h) cuts the error to "
          f"{vb['horizon_law_slack24_bias_at_cap1.05'] * 100:.1f}% (24 h slack) and {vb['horizon_law_slack168_bias_at_cap1.05'] * 100:.1f}% (168 h; Figure 2)."),
        fig(f2, "Median error of the water-filling law against the LP. Fixed blocks of the slack length (grey) underestimate at tight caps; planning over the "
                "virtual battery's refill time (blue) removes most of the bias.", 2),
        p("<b>UPS batteries.</b> A 10-minute UPS is a battery with 1/72 of the virtual energy of half-flexible load, and with losses. That is why carbon-optimal "
          "UPS dispatch cuts emissions by fractions of a percent while shifting the load itself cuts several percent."),
        p("<b>Signals.</b> In earlier work we found that a schedule optimised on 0.7·average + 0.3·marginal carbon keeps most of both optima for load "
          "shifting, but no blend stops a 90%-efficient battery from raising one ledger. The theorem says load shifting <i>is</i> a battery, only lossless, so "
          "losses must be the obstacle. We predicted, then tested: a lossless 1 h battery on the blend raises neither ledger in any of "
          f"{vb['bas']:.0f} U.S. balancing authorities, while at 90% round trip it raises the average ledger in {vb['eta90_w0.7_raises_avg_pct']:.0f}%."),

        h("6 The Storage Exchange Rate"),
        p("How much real storage is a flexible facility worth? We compare the capped saving with a grid-scale 4 h, 90%-efficient battery dispatched optimally "
          "on the same signal. Its dispatch program is linear in capacity while power stays below load, so one solve per zone gives the saving per kWh, b. "
          "b obeys a <b>daily spread law</b>: charge in the 4 cleanest hours, discharge in the 4 dirtiest, skip days whose spread does not cover losses, "
          f"b = (1/24)·mean max(0, √η·H<sub>4</sub> − L<sub>4</sub>/√η) (R² {ex['spread_law_r2_4h']:.3f} against the LP). The exchange rate is X = S/b."),
        fig(f3, "kWh of 4 h, 90%-efficient battery that saves as much carbon as the capped flexible load (median and interquartile range, 123 zones, 24 h slack).", 3),
        table([["Headroom h", "+5%", "+10%", "+20%", "+50%"],
               ["Battery kWh per kW (φ = 0.5)"] + [f"{X(0.5, c):.1f}" for c in (1.05, 1.1, 1.2, 1.5)],
               ["Embodied ratio, median"] + [f"{E('median', c):.0f}×" for c in (1.05, 1.1, 1.2, 1.5)],
               ["Embodied ratio, minimum"] + [f"{E('min', c):.0f}×" for c in (1.05, 1.1, 1.2, 1.5)]],
              "Equal-carbon battery and the ratio of its embodied carbon (LFP, 54-69 kg/kWh) to that of the grid-only headroom (transformer and switchgear) "
              "unlocking the same saving.", 1, [1.45 * 72] + [0.4 * 72] * 4),
        p(f"Uncapped, one virtual kWh is worth κ = {ex['X_over_virtual_median_phi0.5_capinf']:.2f} real kWh (median), and by homogeneity X/(24φ) = κ·ρ(h/φ) "
          f"for a retention curve ρ. Chaining the two closed forms predicts X with log-R² {ex['closed_form_X_r2_log']:.2f}, without an optimiser. In 7 of 123 "
          "zones the daily spread never covers 10% losses: a battery saves nothing there, while flexible load still does."),

        h("7 Related Work"),
        p("Upper bounds on temporal and spatial shifting [1, 5] and Google's capacity-curve scheduling [2] are empirical or optimisation-based. Ecovisor [3] "
          "and CarbonScaler [4] expose energy and elasticity to applications. The virtual-battery view of flexible demand is established in power systems "
          "[8, 9], and water-filling is classical [7]; we are not aware of a closed-form treatment of capped carbon-aware compute. Storage can raise "
          "emissions when it arbitrages prices [10]; our spread law gives the carbon analogue per grid."),

        h("8 Discussion and Limitations"),
        p("<b>Implications.</b> An operator can estimate the saving, the value of new capacity and the right planning horizon from a sort of each day's carbon "
          "intensities. Because capacity value grows with the grid's daily swing, it will rise as solar grows. Electrical headroom for deferrable work does not "
          "need UPS or generator backup, and buys the equivalent of several kWh of batteries at a small fraction of their embodied carbon."),
        p("<b>Limitations.</b> Work is fluid and pre-emptible, forecasts are perfect and inflexible load is flat (real Google load shapes change savings by a "
          "few percent); signals are average intensities; embodied factors come from a few EPDs and a secondary battery source; batteries' grid-service "
          "revenue is ignored. The sinusoidal law is an approximation; the order-statistic law it approximates is exact for one-day windows."),

        h("References"),
    ] + [Paragraph(f"[{i}] {r}", S["ref"]) for i, r in enumerate(REFS, 1)]

    doc = BaseDocTemplate(str(OUT), pagesize=letter, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN,
                          title=TITLE, author="Naveen Kumar Lakkaram", creator="carbonshift build_workshop_battery.py", subject="Carbon-aware computing")
    first, later = frames()
    doc.addPageTemplates([PageTemplate("first", first, onPage=page_no), PageTemplate("later", later, onPage=page_no)])
    doc.build(story)
    print(f"wrote {OUT.name}  ({doc.page} pages)")


if __name__ == "__main__":
    build()
