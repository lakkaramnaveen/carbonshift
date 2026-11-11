"""One-page PDF research summary (for applications and emails), built from results/*.csv.

    python scripts/build_summary.py      -> carbonshift_summary.pdf (project root)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
OUT = ROOT / "carbonshift_summary.pdf"


def numbers() -> dict:
    n = {}
    rv = pd.read_csv(RES / "repro_vs_paper.csv")
    bound = rv.claim.str.contains("<10") & (rv.ours < 10)
    n["repro"] = int(((rv.abs_diff <= 1.5) | (rv.abs_diff / rv.paper.abs() <= 0.03) | bound).sum())
    n["repro_total"] = len(rv)
    cz = pd.read_csv(RES / "facility_capped_per_zone.csv")
    n["peak"] = cz[(cz.phi == 0.5) & np.isinf(cz.cap) & (cz.slack == 24)].peak_draw.median()
    fs = pd.read_csv(RES / "facility_summary.csv")
    n["kept20"] = float(fs[(fs.phi == 0.5) & (fs.slack == 24) & np.isclose(fs.cap, 1.2)].retained_of_ideal.iloc[0]) * 100
    ds = pd.read_csv(RES / "diurnal_summary.csv", index_col=0)["value"]
    n.update(util=ds.mean_util_median * 100, head=ds.headroom_median_pct, kept_real=ds.kept_median_pct)
    wf = pd.read_csv(RES / "waterfill_summary.csv", index_col=0)["value"]
    vb = pd.read_csv(RES / "virtual_battery_summary.csv", index_col=0)["value"]
    n.update(wf_save_r2=wf.saving_r2_finite_caps, wf_val_r2=wf.value_r2, vb_r2=vb.horizon_law_slack24_r2,
             vb_lossless=vb["lossless_w0.7_raises_avg_pct"], vb_lossy=vb["eta90_w0.7_raises_avg_pct"])
    ex = pd.read_csv(RES / "exchange_summary.csv", index_col=0)["value"]
    n.update(x20=ex["X_median_phi0.5_cap1.2"], emb20=ex["embodied_ratio_battery_over_headroom_min_phi0.5_cap1.2"])
    vs = pd.read_csv(RES / "variability_summary.csv", index_col=0)["value"]
    n.update(rho_std=vs.rho_daily_std_2022, rho_mean=vs.rho_mean_ci_2022, rho_cv=vs.rho_daily_cv_2022,
             k=vs.loglog_constant_k, e=vs.loglog_elasticity_shadow_vs_std)
    ps = pd.read_csv(RES / "redundancy_payback_summary.csv")
    row = ps[(ps.slack == 24) & np.isclose(ps.cap, 1.2)].iloc[0]
    n.update(val20=row.kg_per_kw_year_median, grid_ok=row.grid_only_pays_back_in_life_pct, cpu_ok=row.cpu_servers_pays_back_in_life_pct)
    rd = pd.read_csv(RES / "redundancy_dividend_summary.csv")
    g = lambda t: float(rd[np.isclose(rd.u, 0.65) & (rd.topology == t)].perfect_pct.iloc[0])
    ideal = float(rd[np.isclose(rd.u, 0.65)].ideal_pct.iloc[0])
    n["gap_n1"] = (g("N+1 (N=4)") - g("design_only")) / (ideal - g("design_only")) * 100
    n["gap_2n"] = (g("2N") - g("design_only")) / (ideal - g("design_only")) * 100
    ms = pd.read_csv(RES / "marginal_summary.csv", index_col=0)["value"]
    n.update(m_share=ms.share_of_marg_optimum_pooled * 100, m_raise=ms.bas_where_avg_schedule_raises_marginal_pct,
             m_rho=ms.rho_shadow_marg_vs_std_marg, m_bas=int(ms.bas))
    jt = pd.read_csv(RES / "joint_summary.csv").set_index(["group", "cap"])
    n["joint_eu"] = jt.loc[("Europe", "real (1/0.70)"), "joint_extra_over_spatial_pct"]
    n["temp_eu"] = jt.loc[("Europe", "real (1/0.70)"), "temporal_pct"]
    ls = pd.read_csv(RES / "llm_power_summary.csv")
    e = lambda b: float(ls[ls.model.str.startswith("Qwen2.5-1.5B") & (ls.batch == b)].all_j_per_tok.iloc[0])
    n["llm_x"] = e(1) / e(32)
    lb = pd.read_csv(RES / "llm_batching_summary.csv")
    n["aware_max"] = lb.aware_vs_static_saving_pct_pooled.max()
    return n


def main() -> None:
    n = numbers()
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["Normal"], fontName="Times-Roman", fontSize=9.6, leading=11.6, alignment=TA_JUSTIFY, spaceAfter=4)
    h = ParagraphStyle("h", parent=body, fontName="Times-Bold", fontSize=10.5, spaceBefore=5, spaceAfter=2, alignment=0)
    title = ParagraphStyle("t", parent=body, fontName="Times-Bold", fontSize=13.5, leading=16, alignment=TA_CENTER, spaceAfter=2)
    sub = ParagraphStyle("s", parent=body, fontName="Times-Italic", fontSize=9.6, alignment=TA_CENTER, spaceAfter=6)
    cap = ParagraphStyle("c", parent=body, fontSize=8.4, leading=10, alignment=TA_CENTER, spaceAfter=4)
    bullet = ParagraphStyle("bl", parent=body, leftIndent=10, bulletIndent=0)

    def B(text):
        return Paragraph(text, bullet, bulletText="•")

    story = [
        Paragraph("Power Before Hours: Facility Limits, Embodied Carbon, and the Redundancy Dividend in Carbon-Aware Workload Shifting", title),
        Paragraph("Naveen Kumar Lakkaram · research summary · November 2026", sub),
        Paragraph("<b>Question.</b> Carbon-aware schedulers move flexible cloud work to cleaner hours and regions. Upper-bound studies "
                  "(Sukprasert, Souza et al., EuroSys'24) optimise each job in isolation, implicitly letting all shifted work pile into the same clean "
                  "hours. A real data centre's grid draw is capped by its electrical plant (utility service, transformers, switchgear, UPS, generators). "
                  "How much of the carbon saving survives that cap, what is one extra kilowatt of capacity worth, and is it worth building?", body),
        Paragraph("Approach", h),
        Paragraph(f"A clean-room reproduction of the EuroSys'24 analysis (123 grid zones, Electricity Maps 2022) matches <b>{n['repro']} of "
                  f"{n['repro_total']}</b> numeric claims, with three documented discrepancies. The core tool is an exact power-capped scheduling LP. Because "
                  "all work shares one slack, earliest-deadline-first equals FIFO, so feasibility reduces to cumulative bounds: one variable per hour, "
                  "proved equivalent to the per-arrival LP. Its dual variables are the <i>shadow carbon price of capacity</i>. Data: Google's public power "
                  "traces (57 power domains), EIA-930 marginal signals, manufacturer EPDs, and LLM measurements on an Apple M4 Pro GPU.", body),
        Paragraph("Key findings", h),
        B(f"<b>Power caps bound the savings.</b> The ideal temporal saving needs the median facility to draw {n['peak']:.0f}x its mean power for an "
          f"hour. +20% headroom keeps {n['kept20']:.0f}% of the ideal; at real Google headroom (median utilisation {n['util']:.0f}%, +{n['head']:.0f}%) "
          f"about {n['kept_real']:.0f}% survives."),
        B(f"<b>A unifying theory: capped flexible load is a lossless battery.</b> The LP is exactly a battery of energy slack × φ, discharge power φ "
          "and charge power h, so the optimum is water-filling the cleanest hours. Closed forms follow, with no optimiser: the share of the saving kept depends only on "
          f"headroom per unit of flexible load, h/φ (exact); saving ≈ φ·A·sinc(πφ/(φ+h)) for daily swing A (R² {n['wf_save_r2']:.2f} against the LP); and the value "
          f"of a kW is a daily water-level area (R² {n['wf_val_r2']:.2f}), which is linear in the swing and blind to the mean intensity. It also explains why UPS batteries barely help, and why only "
          f"losses stop a scheduler serving both average and marginal ledgers: a lossless battery on a blended signal harms {n['vb_lossless']:.0f}% of BAs, "
          f"versus {n['vb_lossy']:.0f}% at 90% round trip. Priced against a real 4 h battery, a half-flexible facility at +20% headroom is worth "
          f"{n['x20']:.1f} kWh of storage per kW of load, and that battery would embody at least {n['emb20']:.0f}x the carbon of the headroom."),
        B(f"<b>Build power, not servers.</b> One extra kW is worth {n['val20']:.0f} kg CO2/yr (median zone, +20% headroom). Grid-only electrical "
          f"headroom repays its embodied carbon within its life in {n['grid_ok']:.0f}% of zones; CPU servers bought to absorb shifted work, in only "
          f"{n['cpu_ok']:.0f}%. The ordering held in every one of 5,000 Monte Carlo draws."),
        B(f"<b>Redundancy is free carbon headroom.</b> Using the N+1 / 2N spare path for sheddable work closes {n['gap_n1']:.0f}% / {n['gap_2n']:.0f}% "
          "of the gap to the ideal, at zero embodied cost; maintenance and failures erode almost none of it."),
        B(f"<b>Signal choice matters (extends e-Energy'24).</b> Across {n['m_bas']} US balancing authorities (EIA-930), the swing law holds for marginal "
          f"signals (ρ = {n['m_rho']:.2f}), but a capped schedule tuned to average intensity captures only {n['m_share']:.0f}% of the marginal-optimal "
          f"saving and raises marginal emissions in {n['m_raise']:.0f}% of them."),
        B(f"<b>Under real caps, spatial and temporal flexibility compete.</b> In Europe, temporal shifting alone saves {n['temp_eu']:.1f}% but adds only "
          f"{n['joint_eu']:.1f} percentage points on top of migration, because migration fills the clean regions' headroom."),
        B(f"<b>For LLM inference, batch first.</b> Measured on an M4 Pro GPU, batching cuts energy per token {n['llm_x']:.1f}x; making batch size follow "
          f"the grid adds at most {n['aware_max']:.1f}% more."),
    ]
    fig_w = 3.45 * inch
    figs = Table([[Image(str(FIG / "facility_retained.png"), width=fig_w, height=fig_w * 0.375),
                   Image(str(FIG / "variability_drivers.png"), width=fig_w, height=fig_w * 0.38)],
                  [Paragraph("Share of the ideal saving kept as power headroom grows.", cap),
                   Paragraph("Value of +1 kW tracks the grid's daily swing (left), not its level (right).", cap)]],
                 colWidths=[3.6 * inch, 3.6 * inch])
    figs.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story += [Spacer(1, 4), figs]
    story += [
        Paragraph("Why it matters and next steps", h),
        Paragraph("Carbon-aware schedulers should see watts, not only hours. The natural next step is an Ecovisor-style interface that exposes both the "
                  "carbon signal and the facility's power budget to applications. Further steps: validate the capped model with real facility "
                  "telemetry, repeat the GPU study on data-centre accelerators with power capping, and design carbon-optimal headroom that weighs the "
                  "embodied carbon of the electrical plant against operational savings. My background in data-centre power equipment (UPS, "
                  "switchgear, generators) is what led to this facility-side view.", body),
        Paragraph("Artifact: github.com/lakkaramnaveen/carbonshift. It includes tested open-source code (88 tests), all data, an interactive web explorer, a full paper and two "
                  "workshop drafts. Every number above is generated from the result files.", body),
    ]
    doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch, topMargin=0.5 * inch, bottomMargin=0.5 * inch,
                            title="Power Before Hours: research summary", author="Naveen Kumar Lakkaram")
    doc.build(story)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
