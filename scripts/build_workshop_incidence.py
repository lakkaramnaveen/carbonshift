"""Fourth workshop paper: who bears the cost of carbon-aware computing (F28, with F17/F19).

Double-blind draft (no author names). Every number is read from results/*.csv; the two
figures are drawn here at column width.

    python scripts/build_workshop_incidence.py   -> carbonshift_incidence_workshop.pdf (project root)

Inputs come from scripts/incidence_experiments.py and hedge_experiments.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import BaseDocTemplate, FrameBreak, NextPageTemplate, PageTemplate, Paragraph

from build_workshop import MARGIN, ROOT, frames, page_no, styles
from carbonshift import plotting as P

RES = ROOT / "results"
OUT = ROOT / "carbonshift_incidence_workshop.pdf"
TITLE = "Moved, Not Cut: Who Bears the Cost of Carbon-Aware Computing?"

REFS = [
    "T. Sukprasert, A. Souza, N. Bashir, D. Irwin, P. Shenoy. On the Limitations of Carbon-Aware Temporal and Spatial Workload Shifting in the Cloud. EuroSys, 2024.",
    "T. Sukprasert, N. Bashir, A. Souza, D. Irwin, P. Shenoy. On the Implications of Choosing Average versus Marginal Carbon Intensity Signals on Carbon-aware Optimizations. ACM e-Energy, 2024.",
    "J. Gorka, N. Rhodes, L. Roald. ElectricityEmissions.jl: A Framework for the Comparison of Carbon Intensity Signals. arXiv:2411.06560, 2024.",
    "K. Siler-Evans, I. L. Azevedo, M. G. Morgan. Marginal Emissions Factors for the U.S. Electricity System. Environ. Sci. Technol. 46(9), 2012.",
    "U.S. Energy Information Administration. Hourly Electric Grid Monitor (Form EIA-930), 2022.",
    "World Resources Institute and WBCSD. GHG Protocol Scope 2 Guidance, 2015; and the 2025-26 public consultation on consequential electricity-sector impacts.",
    "WattTime. Accounting for Impact: Reconsidering Carbon Accounting in Pursuit of Emissions Reductions. White paper, 2022.",
    "R. H. Coase. The Marginal Cost Controversy. Economica 13(51), 1946.",
    "California ISO. Production and Curtailments Data, 2021-2024.",
    "A. Radovanovic, R. Koningstein, et al. Carbon-Aware Computing for Datacenters. arXiv:2106.11750, 2021.",
    "P. Wiesner, I. Behnke, D. Scheinert, K. Gontarska, L. Thamsen. Let's Wait Awhile: How Temporal Workload Shifting Can Reduce Carbon Emissions in the Cloud. Middleware, 2021.",
    "I. Riepin, T. Brown, V. M. Zavala. Spatio-temporal Load Shifting for Truly Clean Computing. Advances in Applied Energy 17, 100202, 2025 (arXiv:2405.00036).",
    "Q. Xu, W. Ricks, A. Manocha, N. Patankar, J. D. Jenkins. System-level Impacts of Voluntary Carbon-free Electricity Procurement Strategies. Joule 8(2), 2024.",
    "GHG Protocol. Scope 2 Market-Based Method Update: status of the 2025-26 consultation on hourly matching and deliverability, October 2026.",
    "G. Perino. New EU ETS Phase 4 Rules Temporarily Puncture Waterbed. Nature Climate Change 8(4), 262-264, 2018.",
    "K. E. Rosendahl. EU ETS and the Waterbed Effect. Nature Climate Change 9, 734-735, 2019.",
    "European Commission. Publication of the Total Number of Allowances in Circulation in 2025 for the Purposes of the Market Stability Reserve. Communication C/2026/2957, 2026.",
]


def load() -> dict:
    return {"s": pd.read_csv(RES / "incidence_summary.csv", index_col=0)["value"],
            "a": pd.read_csv(RES / "incidence_per_ba.csv"),
            "c": pd.read_csv(RES / "incidence_caiso.csv"),
            "hs": pd.read_csv(RES / "hedge_summary.csv", index_col=0)["value"],
            "hf": pd.read_csv(RES / "hedge_frontier_per_ba.csv"),
            "hm": pd.read_csv(RES / "hourly_matching_summary.csv", index_col=0)["value"],
            "wb": pd.read_csv(RES / "waterbed_summary.csv"),
            "wl": pd.read_csv(RES / "waterbed_fate_by_lag.csv")}


def figures(n) -> list[Path]:
    import matplotlib.pyplot as plt
    P.setup()
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8})
    a, out = n["a"], []

    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    lim = (-20, 75)
    ax.plot(lim, lim, color=P.INK_2, lw=0.6, ls=":")
    ax.axhline(0, color=P.INK_2, lw=0.5)
    ax.axvline(0, color=P.INK_2, lw=0.5)
    ax.scatter(a.avgopt_claimed_g, a.avgopt_real_g, s=9, color=P.ORANGE, label="average-optimal plan", zorder=3)
    ax.scatter(a.margopt_location_reported_g, a.margopt_real_oos_g, s=9, color=P.BLUE, marker="s",
               label="marginal-optimal plan (held-out)", zorder=3)
    ax.text(48, 52, "reported = real", fontsize=6.5, color=P.INK_2, rotation=38)
    ax.set(xlim=lim, ylim=lim, xlabel="Reported saving, location-based (g/kWh)", ylabel="Real cut in grid emissions (g/kWh)")
    ax.legend(frameon=False, loc="lower right")
    out.append(P.save(fig, "ws_incidence_scatter"))

    d = a.sort_values("avgopt_claimed_g", ascending=False).reset_index(drop=True)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    real_pos = d.avgopt_real_g.clip(lower=0)
    ax.bar(x, d.avgopt_real_g, color=P.BLUE, width=0.8, label="real cut")
    ax.bar(x, d.avgopt_transfer_g, bottom=real_pos, color=P.ORANGE, width=0.8, label="moved onto other consumers")
    ax.plot(x, d.avgopt_claimed_g, "_", color="k", ms=4, label="reported")
    ax.axhline(0, color=P.INK_2, lw=0.5)
    ax.set_xticks([])
    ax.set(xlabel=f"{len(d)} US balancing authorities, by reported saving", ylabel="g per kWh of load")
    ax.set_ylim(None, d.avgopt_claimed_g.max() * 1.45 + 10)
    ax.legend(frameon=False, loc="upper center", ncol=3, handlelength=1.2, columnspacing=0.8)
    out.append(P.save(fig, "ws_incidence_bars"))
    plt.rcdefaults()
    return out


def build() -> None:
    n = load()
    s, a, c, hs, hf, hm = n["s"], n["a"], n["c"], n["hs"], n["hf"], n["hm"]
    f1, f2 = figures(n)
    S = styles()
    p, h, b, fig, table = S["p"], S["h"], S["b"], S["fig"], S["table"]
    pc = lambda v, d=0: f"{v * 100:.{d}f}%"  # noqa: E731
    wb, wlag = n["wb"], n["wl"]
    wb_cloud = wb[wb.weight == "cloud"]
    wb_cap = 1 - float(wb_cloud[wb_cloud.regime == "no cap"].saving_share.iloc[0])
    wb_eu = float(wb_cloud[wb_cloud.regime == "EU ETS (MSR)"].saving_share.iloc[0])
    wb_us = float(wb[wb.weight == "us_marginal_F17"].saving_share.iloc[0])
    wd = wlag[wlag.rules == "deal_2026 (700 Mt)"]
    wl_before = float(wd[wd.lag == -1].real_upper_median.iloc[0])
    wl_after = float(wd[wd.lag == 0].real_upper_median.iloc[0])
    wl_n = int(wd[wd.lag == -1].n.iloc[0])
    nb = int(s["bas"])
    ok = hf.groupby("ba").filter(lambda d: d.avg_saving_g.max() > 0.5 and d.marg_saving_g.max() > 0.5)
    best = ok.groupby("ba")[["avg_saving_g", "marg_saving_g"]].max().sum()
    pool = ok.groupby("w_avg")[["avg_saving_g", "marg_saving_g"]].sum()
    w7a, w7m = pool.loc[0.7, "avg_saving_g"] / best.avg_saving_g, pool.loc[0.7, "marg_saving_g"] / best.marg_saving_g
    herd = c[c.schedule.str.startswith("average")]
    h100, h1g = herd[herd.fleet_mw == 100], herd[herd.fleet_mw == 1000]
    sh1g = h1g.transfer_g / h1g.location_claimed_g
    recog = a.margopt_location_reported_g.sum() / a.margopt_real_oos_g.sum()
    rng = lambda v, d=0: (f"{v.min():.{d}f} to {v.max():.{d}f}" if v.min() < 0 else f"{v.min():.{d}f}–{v.max():.{d}f}")  # noqa: E731

    story = [
        Paragraph(TITLE, S["title"]),
        Paragraph("Anonymous submission", S["auth"]),
        Paragraph("Workshop draft, November 2026 · double-blind · code and data available on request", S["note"]),
        FrameBreak(),
        NextPageTemplate("later"),
        Paragraph(
            "<b>Abstract.</b> Carbon-aware computing is evaluated with the grid's <i>average</i> carbon intensity, the signal of location-based Scope 2 "
            "accounting. Because that accounting is budget balanced, a shifting load's reported saving equals its real effect on grid emissions plus "
            "emissions moved onto every other consumer's inventory. We measure the split on "
            f"{nb} U.S. balancing authorities. For a power-capped data centre that schedules on the average signal, {pc(s['transfer_share_pooled'])} of the "
            f"reported saving (median authority {pc(s['transfer_share_median'])}) is moved, not cut; in {s['bas_transfer_over_100pct']:.0f} of {nb} the grid "
            f"emits more. The schedule that does cut emissions, by {s['real_gain_margopt_over_avgopt_pooled']:.0f}× more on held-out marginal factors, is "
            f"recognised at {pc(recog, 1)} "
            f"and booked as an increase in {s['bas_margopt_reported_as_increase']:.0f}. Borrowing the two-part tariff from utility "
            "economics, we propose an accounting rule that keeps budget balance but reports exactly the real effect. Hourly matching (24/7 CFE), the "
            "proposed stricter standard, does not fix this: shifting into one's own solar hours reports a median "
            f"{hm['hm_avg_tb_reported_g_median']:.0f} g/kWh saved, of which {pc(hm['hm_avg_tb_transfer_share_pooled'])} is moved onto other consumers.", S["body"]),

        h("1 Introduction"),
        p("Carbon-aware schedulers move flexible work to hours when the grid is cleaner [1, 10, 11]. Whether they help is usually asked as a question about "
          "the signal: should the scheduler use the <i>average</i> intensity that corporate inventories are reported in, or the <i>marginal</i> intensity of "
          "the plant that actually responds [2, 4]? We ask a different question: <i>whose</i> emissions change. Location-based accounting [6] divides each "
          "hour's grid emissions among all consumers in proportion to their use. That makes it a closed ledger. Any saving one consumer books that the grid "
          "did not deliver must appear on someone else's books."),
        p("<b>Contributions.</b>"),
        b("An exact incidence identity, and its first-order form: the average rate is a marginal charge plus an <i>hourly residual</i>, and the residual is what moves (§2)."),
        b(f"Measurement on {nb} U.S. grids: most of a location-based saving is a transfer, and the real reduction is booked as an increase in about half of them (§3)."),
        b("A two-part accounting rule, budget balanced and incentive aligned, checked on a structural model of California with curtailment (§4, §5)."),
        b("The same incidence under hourly-matched (24/7 CFE) accounting: absorbing one's own clean surplus is a claim on energy the neighbours would have used (§6)."),

        h("2 The Incidence Identity"),
        p("Let the grid emit E<sub>t</sub> to serve demand D<sub>t</sub>, so the average intensity is c<sub>t</sub> = E<sub>t</sub>/D<sub>t</sub>, and a "
          "consumer drawing L<sub>t</sub> is attributed Σ<sub>t</sub> c<sub>t</sub>L<sub>t</sub>. Attributions sum to Σ E<sub>t</sub>, so for any change "
          "of schedule, dA<sub>self</sub> + dA<sub>others</sub> = dE exactly. If the grid answers a small change of load at the marginal rate "
          "m<sub>t</sub>, then dE = Σ m<sub>t</sub>dL<sub>t</sub> and, to first order in the consumer's size,"),
        p("<para alignment='center'>transfer = dA<sub>others</sub> = Σ<sub>t</sub> (m<sub>t</sub> − c<sub>t</sub>) dL<sub>t</sub> = reported − real.</para>"),
        p("Write c<sub>t</sub> = m<sub>t</sub> + r<sub>t</sub>. Location-based accounting is marginal accounting plus an hourly residual r<sub>t</sub>, "
          "allocated hour by hour. In a clean hour with gas at the margin, r<sub>t</sub> is large and negative: a kWh the grid serves with gas is "
          "billed at the solar-rich average. A scheduler that seeks low c<sub>t</sub> harvests this residual, and the residual is paid by everyone else."),

        h("3 Measurement on U.S. Grids"),
        p("<b>Setting.</b> A data centre with mean load 1, half of it flexible within 24 h, capped at 1/0.70 of mean (median Google utilisation), "
          "scheduled by an exact year-long LP. Average and marginal signals for each balancing authority (BA) come from EIA-930 2022 [5]; marginal factors "
          "are Siler-Evans season × hour regressions [4]. Marginal-optimal plans are scored on factors from held-out days (plan on even days, score on "
          "odd, and vice versa), so regression noise cannot inflate them."),
        fig(f1, "Reported (location-based) against real saving per BA. Average-optimal plans (orange) sit on the x-axis: reported, not delivered. "
                "Marginal-optimal plans (blue) sit on the y-axis: delivered, not reported.", 1),
        p(f"<b>Most of the reported saving is moved.</b> The average-optimal plan reports a median {s['avgopt_claimed_g_median']:.1f} g per kWh of load, but "
          f"cuts grid emissions by only {s['avgopt_real_g_median']:.1f} g. Pooled, {pc(s['transfer_share_pooled'])} of the reported saving is transfer; in "
          f"{s['bas_transfer_over_half']:.0f} of {nb} BAs it is more than half, and in {s['bas_transfer_over_100pct']:.0f} it exceeds the claim, so the grid emits "
          f"more while the operator books a saving (Figure 2). Scoring on split-half factors changes the pooled share to {pc(s['transfer_share_pooled_split'])}."),
        fig(f2, "Average-optimal plan per BA: the reported saving (black tick) split into a real cut (blue) and emissions moved onto other consumers (orange).", 2),
        p(f"<b>The real reduction is invisible.</b> The marginal-optimal plan cuts a median {s['margopt_real_oos_g_median']:.1f} g/kWh on held-out factors, "
          f"{s['real_gain_margopt_over_avgopt_pooled']:.0f}× the average plan's real cut pooled, and it cuts emissions in {s['bas_margopt_real_positive']:.0f} of {nb} "
          f"BAs. Location-based books recognise {pc(recog, 1)} of it and record an <i>increase</i> in "
          f"{s['bas_real_saving_reported_as_increase']:.0f} BAs where emissions fall. An operator held to its location-based number is paid to move emissions "
          "and penalised for cutting them."),

        h("4 A Two-Part Rule"),
        p("Utility economics met the same problem with prices: marginal-cost prices give the right incentive but do not recover total cost. Coase's answer was "
          "a two-part tariff [8]: charge the margin, recover the rest with a charge the customer cannot change by its behaviour at the margin. Applied to carbon,"),
        p("<para alignment='center'>a<sub>i</sub> = Σ<sub>t</sub> m<sub>t</sub>L<sub>it</sub> + ρ Σ<sub>t</sub> L<sub>it</sub>,  "
          "ρ = (Σ E<sub>t</sub> − Σ m<sub>t</sub>D<sub>t</sub>) / Σ D<sub>t</sub>.</para>"),
        p("The residual is allocated by <i>annual</i> energy instead of hour by hour. Attributions still sum to grid emissions, which answers the objection "
          "that marginal accounting over-counts. But an energy-preserving shift now changes a<sub>i</sub> by Σ m<sub>t</sub>dL<sub>t</sub>, its real effect, "
          "and moves nothing onto others. Under it, the best plan to report is the best plan for the grid, and blending signals [2] is no longer needed: "
          f"for comparison, the best blend under location-based reporting (0.7 · average + 0.3 · marginal) keeps {pc(w7a)} of one and {pc(w7m)} of the other."),

        h("5 Check on a Structural Margin"),
        p("Regression factors are one model of the margin. As an independent check we add a new data-centre fleet to California's grid (CAISO 2021-2024 [9]), "
          "serve its load first from curtailed solar and wind and otherwise from gas at 490 g/kWh, and compute both rules <i>exactly</i> at finite size."),
        table([["CAISO, average-signal herd", "100 MW", "1 GW"],
               ["Real cut (g/kWh)", rng(h100.real_g), rng(h1g.real_g)],
               ["Reported, location-based", rng(h100.location_claimed_g), rng(h1g.location_claimed_g)],
               ["Reported, two-part", rng(h100.two_part_claimed_g), rng(h1g.two_part_claimed_g)],
               ["Moved onto others", rng(h100.transfer_g), rng(h1g.transfer_g)]],
              "Ranges over 2021-2024; +20% power cap, half the load flexible, 24 h slack.", 1, [1.55 * inch, 0.75 * inch, 0.75 * inch]),
        p(f"The identity holds to rounding, and two-part claims are within {s['caiso_two_part_minus_real_g_maxabs']:.1f} g/kWh of the real effect in every case. "
          "The incidence is <i>signed</i>. At 100 MW the herd uses hours with curtailment, where m &lt; c, so it under-claims and other consumers gain. At 1 GW "
          f"the curtailment is used up and {rng(sh1g * 100)}% of the reported saving is moved onto others. California is the favourable exception that the U.S. "
          "regression data also show (CISO: −5%); the typical grid has gas or coal at the margin in its clean hours."),

        h("6 Hourly Matching Does Not Fix It"),
        p("The GHG Protocol's Scope 2 revision proposes hourly-matched market-based accounting [14]: a consumer with a clean portfolio P<sub>t</sub> is charged "
          "only for unmatched load, Σ c<sub>t</sub> max(L<sub>t</sub> − P<sub>t</sub>, 0), and surplus earns nothing. A kWh moved from a deficit hour into a "
          "surplus hour of one's <i>own</i> portfolio is booked as a full residual-mix kWh saved. But the portfolio is already built; the surplus it no longer "
          "exports is replaced by the rest of the grid at m<sub>t</sub>, so the real change is Σ m<sub>t</sub>dL<sub>t</sub> exactly as without a portfolio, and "
          "hourly market-based accounting, being budget balanced, moves the difference onto others' residual mix. We repeat §3 with a solar portfolio "
          "(clear-sky shape at each BA, sized to 100% of annual load) and an exact LP that maximises the hourly match."),
        p(f"Shifting lifts the CFE score by a median {hm['hm_avg_tb_cfe_gain_pp_median']:.0f} points (from {pc(hm['cfe_flat_median'])}) and reports a median "
          f"{hm['hm_avg_tb_reported_g_median']:.0f} g/kWh saved, five times the location-based claim of §3, while the grid's emissions fall by "
          f"{hm['hm_avg_tb_real_g_median']:.1f} g. Pooled, {pc(hm['hm_avg_tb_transfer_share_pooled'])} is moved (split-half factors: "
          f"{pc(1 - hm['hm_avg_tb_real_share_pooled_split'])}); the grid emits more in {hm['hm_avg_tb_bas_real_negative']:.0f} of {nb} BAs. Portfolios of 50% "
          f"and 150% give {pc(hm['kappa0.5_hm_transfer_share_pooled'])} and {pc(hm['kappa1.5_hm_transfer_share_pooled'])}; breaking LP ties on the marginal "
          f"signal changes nothing ({pc(hm['hm_marg_tb_transfer_share_pooled'])}). The marginal-optimal plan cuts {hm['real_gain_margopt_over_hm_pooled']:.1f}× "
          "more and barely moves the CFE score. A closed form, real share ≈ (mean m in deficit hours − mean m in surplus hours) / mean c in deficit hours, tracks the LP "
          "(Pearson 0.94 across BAs): the margin at night and at noon is usually the same fossil fleet. California is again the exception "
          f"(real CAISO solar: {hm['caiso_real_solar_real_g']:.0f} of {hm['caiso_real_solar_reported_g']:.0f} g/kWh real), because its surplus hours are curtailed. "
          "This is not an argument against 24/7 procurement, whose <i>building</i> of clean capacity can be real [12, 13]; it is that operating against a built "
          "portfolio is scored on the wrong signal. The two-part rule extends directly: credit the portfolio at m<sub>t</sub>, a<sub>i</sub> = Σ m<sub>t</sub>"
          "(L<sub>it</sub> − P<sub>it</sub>) + ρ Σ L<sub>it</sub>."),

        h("7 Under a Cap, Moved Again"),
        p("Every number above is a change in power-plant emissions. But much of the cloud sits in grids under an emissions cap, and under a binding cap a "
          "tonne one consumer avoids is a tonne an installation elsewhere may emit later: it is moved through the allowance bank, not cut (the waterbed "
          f"[15, 16]). Mapping the 123 zones of [1] to their cap regimes, {pc(wb_cap)} of the cloud-region-weighted temporal-shifting saving "
          f"({pc(wb_eu)} in the EU ETS alone) and {pc(wb_us)} of the U.S. marginal saving of §3 sit under a cap. The EU's Market Stability Reserve punctures "
          "the waterbed: an avoided tonne raises the surplus TNAC by one, and while the TNAC is between 833 and 1,096 Mt the reserve takes in exactly the "
          "excess over 833 Mt, so the tonne is absorbed and later invalidated. The published 2025 TNAC is 1,023 Mt [17]. We simulate the bank mechanically "
          f"for {wl_n} net-supply paths and three rule sets (the 2023 law, the October 2026 deal raising the invalidation threshold to 700 Mt, the "
          "Commission's April 2026 proposal to stop invalidation), with two independent derivations that agree in every case. The real cut is a step "
          f"function of time: a median {wl_before:.2f} t per tonne for abatement the year before the TNAC first reaches 833 Mt (transients of 0.6-1.7, from "
          f"overlapping intake years), and {wl_after:.2f} from that year on, when the reserve stops absorbing. That crossing comes within about a year of the "
          "market turning short, which analysts expect around 2026-27. The 700 Mt deal changes nothing at the margin as long as the reserve holds more than "
          "700 Mt. From then on a European carbon-aware scheduler cuts the allowance price, not emissions. By symmetry, new demand there adds none, so the "
          "transfer of §3 becomes a transfer to other capped sectors."),

        h("8 Related Work"),
        p("Gorka, Rhodes and Roald [3] show on test networks that shifting on average intensity can lower a consumer's allocation while raising system "
          "emissions, and examine non-shifting consumers. Sukprasert et al. [2] compare signals across 65 regions. WattTime's impact accounting [7] and the "
          "GHG Protocol's consultation [6] propose marginal charges. Riepin, Brown and Zavala [12] show that load shifting cuts the cost of 24/7 "
          "procurement and argue that grid signals then matter little; Xu et al. [13] model its system-level impacts through new build. We add the exact identity, measurement on real grids under a facility cap, and a "
          "budget-balanced rule that aligns incentives."),

        h("9 Threats to Validity"),
        p("Marginal factors are season × hour regressions on one year and exclude imports; held-out scoring guards against noise, not model error. The "
          "identity is first order in the consumer's size (exact at finite size in §5). Flexible work is fluid and forecasts are perfect. The two-part rule "
          "needs an agreed marginal signal, §6 uses clear-sky solar portfolios (r = 0.95 with real CAISO solar) and the grid average as residual mix, and how to allocate the residual (by energy, by peak, by customer) is a policy choice we do not settle."),

        h("References"),
    ] + [Paragraph(f"[{i}] {r}", S["ref"]) for i, r in enumerate(REFS, 1)]

    doc = BaseDocTemplate(str(OUT), pagesize=letter, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN,
                          title=TITLE, author="Anonymous", creator="carbonshift build_workshop_incidence.py", subject="Carbon-aware computing")
    first, later = frames()
    doc.addPageTemplates([PageTemplate("first", first, onPage=page_no), PageTemplate("later", later, onPage=page_no)])
    doc.build(story)
    print(f"wrote {OUT.name}  ({doc.page} pages)")


if __name__ == "__main__":
    build()
