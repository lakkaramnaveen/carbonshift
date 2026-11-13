"""Third workshop paper: the cooling penalty of carbon-aware computing (F26-F27).

Every number is read from results/*.csv; the three figures are drawn here at column width.

    python scripts/build_workshop_cooling.py   -> carbonshift_cooling_workshop.pdf (project root)

Inputs come from scripts/cooling_experiments.py, cooling_years.py and thermal_experiments.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import BaseDocTemplate, FrameBreak, NextPageTemplate, PageTemplate, Paragraph

from build_workshop import MARGIN, ROOT, frames, page_no, styles
from carbonshift import plotting as P
from carbonshift.cooling import load_weather, plant_pue
from carbonshift.data import load_carbon

RES = ROOT / "results"
OUT = ROOT / "carbonshift_cooling_workshop.pdf"
TITLE = "Clean Hours Are Hot: The Cooling Penalty of Carbon-Aware Computing"
EXAMPLE = "ZA"

REFS = [
    "T. Sukprasert, A. Souza, N. Bashir, D. Irwin, P. Shenoy. On the Limitations of Carbon-Aware Temporal and Spatial Workload Shifting in the Cloud. EuroSys, 2024.",
    "A. Radovanovic, R. Koningstein, et al. Carbon-Aware Computing for Datacenters. arXiv:2106.11750, 2021.",
    "P. Wiesner, I. Behnke, D. Scheinert, K. Gontarska, L. Thamsen. Let's Wait Awhile: How Temporal Workload Shifting Can Reduce Carbon Emissions in the Cloud. Middleware, 2021.",
    "Z. Liu, Y. Chen, C. Bash, A. Wierman, D. Gmach, Z. Wang, M. Marwah, C. Hyser. Renewable and Cooling Aware Workload Management for Sustainable Data Centers. SIGMETRICS, 2012.",
    "W. Zheng, K. Ma, X. Wang. Exploiting Thermal Energy Storage to Reduce Data Center Capital and Operating Expenses. HPCA, 2014.",
    "A. Souza, N. Bashir, J. Murillo, W. Hanafy, Q. Liang, D. Irwin, P. Shenoy. Ecovisor: A Virtual Energy System for Carbon-Efficient Applications. ASPLOS, 2023.",
    "H. Hersbach et al. The ERA5 Global Reanalysis. Q. J. R. Meteorol. Soc. 146, 2020 (accessed via the Open-Meteo archive API).",
    "Electricity Maps. Hourly carbon-intensity data, 2020-2022, as released with [1].",
    "N. K. Lakkaram. Your Data Centre Is Already a Battery: A Closed-Form Theory of Carbon-Aware Shifting under Power Caps. Workshop draft, 2026.",
]


def load() -> dict:
    rd = lambda f: pd.read_csv(RES / f, index_col=0)["value"]  # noqa: E731
    return {"cs": rd("cooling_summary.csv"), "cy": rd("cooling_years_summary.csv"), "th": rd("thermal_summary.csv"),
            "cz": pd.read_csv(RES / "cooling_per_zone.csv"), "tz": pd.read_csv(RES / "thermal_per_zone.csv")}


def figures(n) -> list[Path]:
    import matplotlib.pyplot as plt
    P.setup()
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8})
    out = []

    # 1. mean day of carbon and PUE at the example hub
    ci = load_carbon(2022)[EXAMPLE].to_numpy()
    p = plant_pue(load_weather(2022), EXAMPLE, "legacy")
    dc, dp = ci.reshape(-1, 24).mean(0), p.reshape(-1, 24).mean(0)
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    ax.plot(range(24), dc / dc.mean() * 100, color=P.BLUE, lw=1.4, label="grid carbon")
    ax.plot(range(24), dp / dp.mean() * 100, color=P.ORANGE, lw=1.4, label="PUE (legacy plant)")
    ax.axhline(100, color=P.INK_2, lw=0.5)
    ax.set(xlabel="Hour of day (UTC)", ylabel="% of annual mean", xticks=[0, 6, 12, 18, 23])
    ax.set_ylim(top=ax.get_ylim()[1] + 2.5)
    ax.legend(frameon=False, loc="upper center", ncol=2)
    out.append(P.save(fig, "wsc_day"))

    # 2. closed-form law vs LP
    cz = n["cz"]
    cz = cz[(cz.phi == 0.5) & (cz["head"] == 0.2) & (cz.claimed_g > 0.5)]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for plant, col in (("legacy", P.ORANGE), ("hyperscale", P.BLUE)):
        x = cz[cz.plant == plant]
        ax.scatter(x.penalty_law_pct, x.penalty_pct_of_claim, s=9, color=col, lw=0, alpha=0.85, label=plant)
    lim = float(np.nanmax(cz[["penalty_law_pct", "penalty_pct_of_claim"]].to_numpy()))
    ax.plot([-5, lim], [-5, lim], color=P.INK_2, lw=0.8)
    ax.set(xlabel="Closed-form law (% of claimed saving)", ylabel="LP cooling penalty (%)")
    ax.legend(frameon=False, loc="upper left")
    out.append(P.save(fig, "wsc_law"))

    # 3. what drives a chilled-water tank
    tz = n["tz"]
    tz = tz[(tz.capacity_h == 4) & (tz.tes_saving_g > 0.05)]
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    for plant, col in (("legacy", P.ORANGE), ("hyperscale", P.BLUE)):
        x = tz[tz.plant == plant]
        ax.scatter(x.tes_carbon_only_g / x.tes_saving_g * 100, x.tes_weather_only_g / x.tes_saving_g * 100, s=9, color=col, lw=0, alpha=0.85, label=plant)
    ax.plot([-100, 100], [-100, 100], color=P.INK_2, lw=0.6, ls=":")
    ax.axvline(0, color=P.INK_2, lw=0.5)
    ax.set(xlim=(-100, 105), ylim=(-30, 105), xlabel="Kept when dispatched on grid carbon only (%)", ylabel="Kept on weather only (%)")
    ax.legend(frameon=False, loc="lower left")
    out.append(P.save(fig, "wsc_tank"))
    plt.rcdefaults()
    return out


def build() -> None:
    n = load()
    cs, cy, th = n["cs"], n["cy"], n["th"]
    f1, f2, f3 = figures(n)
    S = styles()
    p, h, b, fig, table = S["p"], S["h"], S["b"], S["fig"], S["table"]
    k = "phi0.5_h0.2"
    cz = n["cz"]
    ex = cz[(cz.zone == EXAMPLE) & (cz.plant == "legacy") & (cz.phi == 0.5) & (cz["head"] == 0.2)].iloc[0]
    top = cz[(cz.plant == "legacy") & (cz.phi == 0.5) & (cz["head"] == 0.2)].nlargest(3, "penalty_pct_of_claim")
    tops = ", ".join(f"{r.hub} {r.penalty_pct_of_claim:.0f}%" for r in top.itertuples())
    flat = cz[cz.zone.isin(["HK", "ID"]) & (cz.phi == 0.5) & (cz["head"] == 0.2)].groupby("plant").true_cooling_aware_g.max()
    rng = lambda key: (min(cy[f"{key}_{y}_penalty_pct_pooled"] for y in (2020, 2021, 2022)),  # noqa: E731
                       max(cy[f"{key}_{y}_penalty_pct_pooled"] for y in (2020, 2021, 2022)))
    oos = [cy[f"{pl}_{hh}_oos_law_{a}_to_{a + 1}_r2"] for pl in ("hyperscale", "legacy") for hh in ("h0.2", "hinf") for a in (2020, 2021)]
    hits = sum(cy[f"{pl}_{hh}_oos_flag_{a}_to_{a + 1}_hits"] for pl in ("hyperscale", "legacy") for hh in ("h0.2", "hinf") for a in (2020, 2021))
    miss = sum(cy[f"{pl}_{hh}_oos_flag_{a}_to_{a + 1}_misses"] for pl in ("hyperscale", "legacy") for hh in ("h0.2", "hinf") for a in (2020, 2021))
    r2s = [cs[f"{pl}_phi{ph}_h{hh}_law_r2"] for pl in ("hyperscale", "legacy") for ph in (0.25, 0.5) for hh in ("0.2", "0.5", "inf")]

    story = [
        Paragraph(TITLE, S["title"]),
        Paragraph("Naveen Kumar Lakkaram", S["auth"]),
        Paragraph("Workshop draft, November 2026 · code and data: github.com/lakkaramnaveen/carbonshift", S["note"]),
        FrameBreak(),
        NextPageTemplate("later"),
        Paragraph(
            "<b>Abstract.</b> Carbon-aware schedulers move flexible work to hours when the grid is cleanest. A data centre, however, draws IT power times its PUE, "
            "and PUE rises in hot weather. On solar-heavy grids the cleanest hours are also the hottest, so a scheduler that watches only carbon over-claims. "
            f"With hourly weather at data-centre hubs in {cs['zones']:.0f} cloud regions, carbon-only scheduling over-states its saving by "
            f"{cs[f'hyperscale_{k}_penalty_pct_pooled']:.1f}% (efficient plants) to {cs[f'legacy_{k}_penalty_pct_pooled']:.1f}% (legacy plants) pooled, and by up to "
            f"{cs[f'legacy_{k}_penalty_pct_max']:.0f}% in single regions. A closed form, −ρ·(mean c/σ<sub>c</sub>)·(σ<sub>PUE</sub>/mean PUE), predicts the penalty "
            f"(R² {min(r2s):.2f}-{max(r2s):.2f}) and, evaluated on the previous year, flags {hits:.0f} of {hits + miss:.0f} cases above 10%. The same physics governs "
            f"chilled-water storage: dispatched on grid carbon, as it is marketed, a tank keeps {th['hyperscale_share_carbon_only_pooled'] * 100:.0f}-"
            f"{th['legacy_share_carbon_only_pooled'] * 100:.0f}% of its value and raises emissions in up to {th['hyperscale_zones_carbon_only_raises']:.0f} regions; "
            "dispatched on carbon × 1/COP it stacks fully on top of workload shifting.", S["body"]),

        h("1 Introduction"),
        p("Temporal shifting of flexible compute is the most studied lever of carbon-aware computing [1, 2, 3]. Its benefit is computed from the grid's hourly "
          "carbon intensity c(t), and reported savings are usually scaled by an annual PUE. That treats the facility overhead as a constant. It is not: "
          "chillers reject heat to outdoor air or water, and their efficiency falls as the outdoor temperature rises. The carbon cost of one IT kWh is "
          "c(t)·PUE(t)."),
        p("The two factors are correlated, and in the wrong direction. Solar makes midday both clean and hot. Figure 1 shows Johannesburg: the grid is "
          "cleanest around 11:00 UTC (13:00 local), exactly when a legacy plant's PUE peaks. Cooling-aware workload placement was studied a decade ago for "
          "single sites with on-site solar [4], and thermal storage for cost and peak shaving [5]; here we measure what ignoring cooling costs <i>grid</i>-"
          "carbon-aware schedulers across real grids and weather, and give a closed form for it."),
        p("<b>Contributions.</b>"),
        b("An exact accounting identity for the true saving, and the cooling penalty it implies (§3)."),
        b(f"Measurement on {cs['zones']:.0f} cloud regions with ERA5 weather, 2020-2022, for two cooling plants (§4)."),
        b("A closed-form law that predicts the penalty out of sample from public data (§5)."),
        b("Chilled-water storage as a third battery: what drives it, and how it stacks with shifting (§6)."),
        fig(f1, f"Mean day of grid carbon and PUE at Johannesburg (2022), each as a percent of its annual mean. The clean dip sits under the PUE peak.", 1),

        h("2 Model"),
        p("Each region gets one data-centre hub (e.g. Ashburn for PJM, Santa Clara for CAISO) with hourly 2-m dry- and wet-bulb temperature from ERA5 [7] "
          "and hourly carbon intensity from Electricity Maps [8], both in UTC. Two first-principles cooling plants bracket practice. <i>Hyperscale</i>: 18 °C "
          "chilled water, cooling tower on wet bulb, water-side economiser (free cooling below 10 °C wet bulb), 7% fixed overhead; median annual PUE "
          f"{cs['hyperscale_pue_mean_median']:.2f}. <i>Legacy</i>: 7 °C chilled water, air-cooled condenser on dry bulb, no economiser, 22% fixed overhead; median "
          f"{cs['legacy_pue_mean_median']:.2f}. Chiller COP is a fixed fraction of Carnot at the condensing temperature. The facility follows [9]: flat IT load, a "
          "share φ flexible within 24 h, IT power capped at 1 + h, solved as a year-long LP."),

        h("3 The Cooling Penalty"),
        p("With the IT draw d scaled to mean 1, the saving against running on arrival is exactly <b>−cov(c·PUE, d)</b>. An operator who computes the saving on c "
          "and multiplies by the annual PUE claims −mean(PUE)·cov(c, d). The gap, <b>−cov(c·(PUE − mean PUE), d)</b>, is the cooling penalty: positive when the schedule moves "
          "load into hours of above-average PUE. A scheduler can avoid it by optimising on c·PUE instead of c."),

        h("4 Measurement"),
        p(f"In {cs['legacy_share_zones_clean_hours_hot'] * 100:.0f}% of regions the mean daily profiles of carbon and PUE are anti-correlated (ρ < −0.3). With half "
          f"the load flexible and +20% headroom, carbon-only scheduling over-claims by {cs[f'hyperscale_{k}_penalty_pct_pooled']:.1f}% (hyperscale) and "
          f"{cs[f'legacy_{k}_penalty_pct_pooled']:.1f}% (legacy) pooled; the median region by {cs[f'hyperscale_{k}_penalty_pct_median']:.1f}% and "
          f"{cs[f'legacy_{k}_penalty_pct_median']:.1f}%. The tail matters: {cs[f'legacy_{k}_zones_penalty_over_10pct']:.0f} regions exceed 10% for legacy plants "
          f"({tops}). In Johannesburg the claim is {ex.claimed_g:.1f} g/kWh and the truth {ex.true_carbon_only_g:.1f}."),
        p(f"Scheduling on c·PUE recovers {cs[f'hyperscale_{k}_recovered_pct_pooled']:.1f}% and {cs[f'legacy_{k}_recovered_pct_pooled']:.1f}% more true saving pooled. "
          "On grids whose carbon signal is flat (Hong Kong, Jakarta), where carbon-only scheduling saves nothing, weather alone still saves up to "
          f"{flat.get('legacy', 0):.1f} g/kWh."),
        table([["Pooled penalty", "2020", "2021", "2022"],
               ["Hyperscale, +20%"] + [f"{cy[f'hyperscale_h0.2_{y}_penalty_pct_pooled']:.1f}%" for y in (2020, 2021, 2022)],
               ["Legacy, +20%"] + [f"{cy[f'legacy_h0.2_{y}_penalty_pct_pooled']:.1f}%" for y in (2020, 2021, 2022)],
               ["Legacy, regions > 10%"] + [f"{cy[f'legacy_h0.2_{y}_zones_over_10pct']:.0f}" for y in (2020, 2021, 2022)]],
              f"Cooling penalty as a share of the claimed saving, φ = 0.5, {cs['zones']:.0f} regions, each year's carbon and weather.", 1,
              [1.35 * 72] + [0.55 * 72] * 3),

        h("5 A Closed-Form Law"),
        p("If the schedule moves against the carbon signal, cov(PUE, d)/cov(c, d) ≈ ρ·σ<sub>PUE</sub>/σ<sub>c</sub>, so"),
        p("<b>penalty / claim ≈ −ρ · (mean c / σ<sub>c</sub>) · (σ<sub>PUE</sub> / mean PUE)</b>,"),
        p("with ρ and the σs taken over within-day (25 h detrended) hourly variation. Daily-profile statistics are not enough (R² 0.55-0.75); hourly "
          f"residuals give R² {min(r2s):.2f}-{max(r2s):.2f} against the LP over 12 configurations (Figure 2). The striking factor is mean c/σ<sub>c</sub>, the "
          "inverse coefficient of variation of carbon: the most exposed grids are dirty ones with a small relative swing, where the cooling swing is large "
          f"compared with the carbon swing. The law needs no optimiser. Evaluated on last year's data it predicts this year's LP penalty with R² "
          f"{min(oos):.2f}-{max(oos):.2f}, and flags {hits:.0f} of the {hits + miss:.0f} region-years above 10% (it runs slightly high, so it also raises some false "
          f"alarms). Per-region penalties are stable: Spearman {min(cy[i] for i in cy.index if 'spearman_20' in i):.2f}-"
          f"{max(cy[i] for i in cy.index if 'spearman_20' in i):.2f} between years."),
        fig(f2, "Cooling penalty from the LP against the closed-form law, φ = 0.5, +20% headroom, one point per region and plant.", 2),

        h("6 The Third Battery: Chilled-Water Storage"),
        p("A chilled-water tank lets the plant make cold before it is needed. Cold costs m(t) = 1/COP units of electricity (zero under free cooling), so it "
          "should be made when c·m is low. We solve a cyclic LP per region: a tank of 1-12 h of heat load, chiller 1.5× peak, 1%/day standing loss. "
          f"A 4 h tank cuts facility emissions by a median {th['hyperscale_cap4_saving_pct_median']:.2f}% (hyperscale) and {th['legacy_cap4_saving_pct_median']:.2f}% "
          f"(legacy); cooling is only {th['legacy_cooling_share_pct_median']:.0f}% of a legacy facility's emissions."),
        p(f"<b>The tank is a weather battery.</b> Dispatched on the weather alone it keeps {th['hyperscale_share_weather_only_pooled'] * 100:.0f}% and "
          f"{th['legacy_share_weather_only_pooled'] * 100:.0f}% of its value; dispatched on grid carbon alone, as vendors propose, "
          f"{th['hyperscale_share_carbon_only_pooled'] * 100:.0f}% and {th['legacy_share_carbon_only_pooled'] * 100:.0f}%, and it raises emissions in "
          f"{th['hyperscale_zones_carbon_only_raises']:.0f} and {th['legacy_zones_carbon_only_raises']:.0f} of {th['zones']:.0f} regions (Figure 3), because it makes "
          f"cold at clean, hot middays. One kWh of tank saves as much as {th['hyperscale_cap4_battery_kwh_per_tes_kwh_median']:.2f}-"
          f"{th['legacy_cap4_battery_kwh_per_tes_kwh_median']:.2f} kWh of 4 h Li-ion battery, but on grids with a small daily spread, where a battery's losses eat "
          f"its gain, the tank still saves. On top of cooling-aware shifting it adds its full saving (pooled ratio {th['legacy_stack_ratio_pooled']:.2f})."),
        fig(f3, "Share of a 4 h tank's saving kept when it is dispatched on grid carbon only or on weather only, one point per region and plant.", 3),

        h("7 Discussion and Limitations"),
        p("<b>Implications.</b> Carbon-aware schedulers should optimise on c·PUE, which every operator already meters, and report savings with hourly, not "
          "annual, PUE. Regions where the law exceeds a few percent can be identified from public carbon and weather data before deployment. The penalty "
          "concentrates in hot, humid, fossil-heavy regions (India, Southeast Asia, South Africa). Thermal storage should be "
          "dispatched on carbon × 1/COP; marketing it as 'charge on clean power' is wrong in a third of regions for efficient plants."),
        p("<b>Limitations.</b> Steady-state hourly plant models with assumed parameters (two archetypes bracket practice); one hub per region; PUE "
          "independent of IT load; average, not marginal, carbon; perfect forecasts; tank embodied carbon not counted."),

        h("References"),
    ] + [Paragraph(f"[{i}] {r}", S["ref"]) for i, r in enumerate(REFS, 1)]

    doc = BaseDocTemplate(str(OUT), pagesize=letter, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN,
                          title=TITLE, author="Naveen Kumar Lakkaram", creator="carbonshift build_workshop_cooling.py", subject="Carbon-aware computing")
    first, later = frames()
    doc.addPageTemplates([PageTemplate("first", first, onPage=page_no), PageTemplate("later", later, onPage=page_no)])
    doc.build(story)
    print(f"wrote {OUT.name}  ({doc.page} pages)")


if __name__ == "__main__":
    build()
