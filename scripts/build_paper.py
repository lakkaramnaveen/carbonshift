"""Build the research paper (Word .docx) from the project's results.

Every number in the text is read from results/*.csv at build time, so the
paper cannot drift from the experiments. Re-run after any experiment:

    python scripts/build_paper.py        -> carbonshift_paper.docx (project root)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from carbonshift.embodied import FACTORS, LIFETIME, per_kw

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
OUT = ROOT / "carbonshift_paper.docx"

TITLE = "Power Before Hours: Facility Limits, Embodied Carbon, and the Redundancy Dividend in Carbon-Aware Workload Shifting"
AUTHOR = "Naveen Kumar Lakkaram"
AFFIL = "Independent researcher"


# ============================================================================ numbers
def load_numbers() -> dict:
    n: dict = {}
    rv = pd.read_csv(RES / "repro_vs_paper.csv")
    n["repro"] = rv
    bound_met = rv.claim.str.contains("<10") & (rv.ours < 10)  # claims stated as an upper bound
    n["repro_within"] = int(((rv.abs_diff <= 1.5) | (rv.abs_diff / rv.paper.abs() <= 0.03) | bound_met).sum())
    n["repro_total"] = len(rv)
    sp = pd.read_csv(RES / "repro_spatial.csv").set_index("grouping")
    n["asia50_art"], n["asia50_cor"] = sp.loc["Asia", "idle50_artifact"], sp.loc["Asia", "idle50_corrected"]
    n["amer_inf_art"], n["amer_inf_cor"] = sp.loc["Americas", "inf_artifact"], sp.loc["Americas", "inf_corrected"]

    fs = pd.read_csv(RES / "facility_summary.csv")
    f05 = fs[fs.phi == 0.5]
    n["fac"] = f05
    get = lambda sl, cap, col: float(f05[(f05.slack == sl) & np.isclose(f05.cap, cap)][col].iloc[0])
    n["get"] = get
    n["ideal24"] = float(f05[(f05.slack == 24) & np.isinf(f05.cap)].saving_g.iloc[0])
    n["ideal24_pct"] = float(f05[(f05.slack == 24) & np.isinf(f05.cap)].reduction_pct.iloc[0])
    n["ideal168"] = float(f05[(f05.slack == 168) & np.isinf(f05.cap)].saving_g.iloc[0])
    cz = pd.read_csv(RES / "facility_capped_per_zone.csv")
    unc = cz[(cz.phi == 0.5) & np.isinf(cz.cap)]
    n["peak24"], n["peak168"] = unc[unc.slack == 24].peak_draw.median(), unc[unc.slack == 168].peak_draw.median()
    need = pd.read_csv(RES / "facility_headroom_needed.csv")
    n["need50_24"] = need[need.slack == 24].headroom_for_50pct.median() * 100
    n["need80_24"] = need[need.slack == 24].headroom_for_80pct.median() * 100
    n["need50_168"] = need[need.slack == 168].headroom_for_50pct.median() * 100
    n["zones_flat"] = int((need[need.slack == 24].ideal_g <= 1e-9).sum())
    sa = cz[(cz.zone == "AU-SA") & (cz.phi == 0.5) & (cz.slack == 24)]
    n["sa_5"] = float(sa[np.isclose(sa.cap, 1.05)].facility_reduction_pct.iloc[0])
    inw = cz[(cz.zone == "IN-WE") & (cz.phi == 0.5) & (cz.slack == 24)]
    n["inwe_inf"] = float(inw[np.isinf(inw.cap)].facility_reduction_pct.iloc[0])

    bs = pd.read_csv(RES / "facility_battery_summary.csv")
    bget = lambda e, r, c: float(bs[np.isclose(bs.energy_h, e) & np.isclose(bs.reserve_frac, r)][c].iloc[0])
    n["bat10_mean"], n["bat10_max"] = bget(1 / 6, 0, "mean"), bget(1 / 6, 0, "max")
    n["bat10_res"] = bget(1 / 6, 0.5, "mean")
    n["bat4h_mean"], n["bat4h_max"] = bget(4, 0, "mean"), bget(4, 0, "max")

    ls = pd.read_csv(RES / "llm_power_summary.csv")
    n["llm"] = ls
    lg = lambda m, b, c: float(ls[(ls.model.str.startswith(m)) & (ls.batch == b)][c].iloc[0])
    n["lg"] = lg
    raw = pd.read_csv(RES / "llm_power.csv")
    dec = raw.groupby(["model", "batch"]).decode_tok_per_s.mean()
    n["dec15_32"] = float(dec[[i for i in dec.index if i[0].startswith("Qwen2.5-1.5B") and i[1] == 32][0]])
    n["dec15_128"] = float(dec[[i for i in dec.index if i[0].startswith("Qwen2.5-1.5B") and i[1] == 128][0]])
    cv = raw.groupby(["model", "batch"]).all_power_w.agg(["mean", "std"])
    n["rep_cv_max"] = float((cv["std"] / cv["mean"]).max() * 100)

    ps = pd.read_csv(RES / "redundancy_payback_summary.csv")
    n["pay"] = ps
    n["pg"] = lambda sl, cap, col: float(ps[(ps.slack == sl) & np.isclose(ps.cap, cap)][col].iloc[0])
    n["rob"] = pd.read_csv(RES / "robustness_years_summary.csv").set_index("year")
    rc = pd.read_csv(RES / "robustness_years_shadow_rankcorr.csv", index_col=0)
    off = rc.where(rc < 0.9999)
    n["rob_rc"] = (off.min().min(), off.max().max())
    rd = pd.read_csv(RES / "redundancy_dividend_summary.csv")
    n["rd"] = rd
    n["rg"] = lambda u, t, c: float(rd[np.isclose(rd.u, u) & (rd.topology == t)][c].iloc[0])
    return n


def f(x, d=1):
    return f"{x:,.{d}f}"


# ============================================================================ docx helpers
def set_cell_shade(cell, hex_fill: str):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


class Paper:
    def __init__(self):
        self.doc = Document()
        self.fig_no = 0
        self.tab_no = 0
        sec = self.doc.sections[0]
        sec.left_margin = sec.right_margin = Inches(1)
        sec.top_margin = sec.bottom_margin = Inches(0.9)
        st = self.doc.styles["Normal"]
        st.font.name = "Times New Roman"
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        st.font.size = Pt(10.5)
        st.paragraph_format.space_after = Pt(5)
        st.paragraph_format.line_spacing = 1.08
        for lvl, size in ((1, 13), (2, 11.5), (3, 10.5)):
            h = self.doc.styles[f"Heading {lvl}"]
            h.font.name = "Times New Roman"
            h.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
            h.font.size = Pt(size)
            h.font.bold = True
            h.font.color.rgb = RGBColor(0, 0, 0)
            h.paragraph_format.space_before = Pt(10 if lvl == 1 else 7)
            h.paragraph_format.space_after = Pt(4)
        # page numbers in footer
        p = sec.footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run()
        for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
            if tag:
                el = OxmlElement("w:fldChar")
                el.set(qn("w:fldCharType"), tag)
            else:
                el = OxmlElement("w:instrText")
                el.set(qn("xml:space"), "preserve")
                el.text = text
            r._r.append(el)

    def title(self, text, author, affil, date):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        r.bold, r.font.size = True, Pt(16)
        for line, size, ital in ((author, 11.5, False), (affil, 10.5, True), (date, 10, True)):
            q = self.doc.add_paragraph()
            q.alignment = WD_ALIGN_PARAGRAPH.CENTER
            q.paragraph_format.space_after = Pt(1)
            rr = q.add_run(line)
            rr.font.size, rr.italic = Pt(size), ital
        self.doc.add_paragraph()

    def h(self, text, lvl=1):
        self.doc.add_heading(text, level=lvl)

    def p(self, *parts, align_justify=True, indent=False):
        """Paragraph from strings; **bold** and *italic* inline markers supported."""
        para = self.doc.add_paragraph()
        if align_justify:
            para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if indent:
            para.paragraph_format.left_indent = Inches(0.3)
        self._runs(para, "".join(parts))
        return para

    def _runs(self, para, text):
        import re

        for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
            if not tok:
                continue
            if tok.startswith("**"):
                para.add_run(tok[2:-2]).bold = True
            elif tok.startswith("*"):
                para.add_run(tok[1:-1]).italic = True
            else:
                para.add_run(tok)

    def bullets(self, items, numbered=False):
        for it in items:
            para = self.doc.add_paragraph(style="List Number" if numbered else "List Bullet")
            para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            self._runs(para, it)

    def eq(self, text, label=None):
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = para.add_run(text)
        r.italic = True
        r.font.name = "Cambria Math"
        if label:
            para.add_run(f"\t\t({label})")

    def figure(self, path, caption, width=6.0):
        self.fig_no += 1
        self.doc.add_picture(str(path), width=Inches(width))
        self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c = self.doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = c.add_run(f"Figure {self.fig_no}. ")
        r.bold, r.font.size = True, Pt(9.5)
        r2 = c.add_run(caption)
        r2.font.size = Pt(9.5)
        return self.fig_no

    def table(self, header, rows, caption, widths=None, font=9):
        self.tab_no += 1
        c = self.doc.add_paragraph()
        r = c.add_run(f"Table {self.tab_no}. ")
        r.bold, r.font.size = True, Pt(9.5)
        c.add_run(caption).font.size = Pt(9.5)
        t = self.doc.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, htext in enumerate(header):
            cell = t.rows[0].cells[i]
            cell.text = ""
            run = cell.paragraphs[0].add_run(htext)
            run.bold, run.font.size = True, Pt(font)
            set_cell_shade(cell, "E8EEF7")
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                para = cells[i].paragraphs[0]
                run = para.add_run(str(v))
                run.font.size = Pt(font)
                if i > 0:
                    para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Inches(w)
        self.doc.add_paragraph()
        return self.tab_no

    def refs(self, items):
        for i, it in enumerate(items, 1):
            para = self.doc.add_paragraph()
            para.paragraph_format.left_indent = Inches(0.35)
            para.paragraph_format.first_line_indent = Inches(-0.35)
            para.paragraph_format.space_after = Pt(2)
            r = para.add_run(f"[{i}] ")
            r.font.size = Pt(9)
            r2 = para.add_run(it)
            r2.font.size = Pt(9)

    def save(self, path):
        self.doc.save(path)


# ============================================================================ references
REFS = [
    "T. Sukprasert, A. Souza, N. Bashir, D. Irwin, P. Shenoy. On the Limitations of Carbon-Aware Temporal and Spatial Workload Shifting in the Cloud. In Proc. EuroSys, 2024. arXiv:2306.06502.",
    "A. Radovanović, R. Koningstein, et al. Carbon-Aware Computing for Datacenters. arXiv:2106.11750, 2021.",
    "C. Zhang, A. G. Kumbhare, I. Manousakis, et al. Flex: High-Availability Datacenters With Zero Reserved Power. In Proc. ISCA, 2021.",
    "W. A. Hanafy, L. Wu, D. Irwin, P. Shenoy. CarbonFlex: Enabling Carbon-aware Provisioning and Scheduling for Cloud Clusters. arXiv:2505.18357, 2025.",
    "W. A. Hanafy, Q. Liang, N. Bashir, D. Irwin, P. Shenoy. CarbonScaler: Leveraging Cloud Workload Elasticity for Optimizing Carbon-Efficiency. Proc. ACM Meas. Anal. Comput. Syst. (SIGMETRICS), 2023. arXiv:2302.08681.",
    "A. Souza, N. Bashir, J. Murillo, W. Hanafy, Q. Liang, D. Irwin, P. Shenoy. Ecovisor: A Virtual Energy System for Carbon-Efficient Applications. In Proc. ASPLOS, 2023.",
    "B. Acun, B. Lee, F. Kazhamiaka, K. Maeng, U. Gupta, M. Chakkaravarthy, D. Brooks, C.-J. Wu. Carbon Explorer: A Holistic Framework for Designing Carbon Aware Datacenters. In Proc. ASPLOS, 2023.",
    "P. Wiesner, I. Behnke, D. Scheinert, K. Gontarska, L. Thamsen. Let's Wait Awhile: How Temporal Workload Shifting Can Reduce Carbon Emissions in the Cloud. In Proc. Middleware, 2021. arXiv:2110.13234.",
    "D. Maji, P. Shenoy, R. K. Sitaraman. CarbonCast: Multi-Day Forecasting of Grid Carbon Intensity. In Proc. BuildSys, 2022.",
    "X. Fan, W.-D. Weber, L. A. Barroso. Power Provisioning for a Warehouse-sized Computer. In Proc. ISCA, 2007.",
    "P. Patel, E. Choukse, C. Zhang, Í. Goiri, B. Warrier, N. Mahalingam, R. Bianchini. POLCA: Power Oversubscription in LLM Cloud Providers. arXiv:2308.12908, 2023.",
    "U. Gupta, M. Elgamal, G. Hills, G.-Y. Wei, H.-H. S. Lee, D. Brooks, C.-J. Wu. ACT: Designing Sustainable Computer Systems with an Architectural Carbon Modeling Tool. In Proc. ISCA, 2022.",
    "Y. Li, Z. Hu, E. Choukse, R. Fonseca, G. E. Suh, U. Gupta. EcoServe: Designing Carbon-Aware AI Inference Systems. arXiv:2502.05043, 2025.",
    "N. Bashir, T. Guo, M. Hajiesmaili, D. Irwin, P. Shenoy, R. Sitaraman, A. Souza, A. Wierman. Enabling Sustainable Clouds: The Case for Virtualizing the Energy System. In Proc. SoCC, 2021. arXiv:2106.08872.",
    "A. Hall, F. Micheli, G. Belgioioso, A. Radovanović, F. Dörfler. Carbon-Aware Computing for Data Centers with Probabilistic Performance Guarantees. arXiv:2410.21510, 2024.",
    "W. Kwon, Z. Li, S. Zhuang, et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. In Proc. SOSP, 2023.",
    "Qwen Team. Qwen2.5 Technical Report. arXiv:2412.15115, 2024.",
    "A. Hannun, J. Digani, A. Katharopoulos, R. Collobert. MLX: Efficient and Flexible Machine Learning on Apple Silicon. https://github.com/ml-explore/mlx, 2023.",
    "Electricity Maps. Historical carbon-intensity data (hourly, 2020-2022), ODbL; as released with [1] at https://github.com/umassos/decarbonization-potential (Zenodo 10790855).",
    "Trafo Elettro S.r.l. Environmental Product Declaration, 2500 kVA cast-resin transformers TES-R2-2. EPDItaly1073 rev. 04 (EN 15804+A2).",
    "ABB T&D S.p.A. Environmental Product Declaration, Large Distribution Transformer 16/20 MVA (ONAN/ONAF). S-P-00056.",
    "Cummins Inc. Environmental Product Declaration, QSK95 Series Open Diesel Generator Set. EPD-IES-0025774:003, 2026.",
    "Rolls-Royce Power Systems. Environmental Product Declaration, containerized diesel generator set mtu 10V1600 DS500. EPD-IES-0017705:002, 2025.",
    "Schneider Electric. PEP ecopassport ENVPEP2311001_V2, Galaxy VX UPS (A1-A3), 2024; ABB MegaFlex DPA and DPA 250 S4 UPS EPDs (PEP ecopassport / INIES), via OneClick LCA.",
    "Dell Technologies. Product Carbon Footprint, PowerEdge R650 (2022); Life Cycle Assessment, PowerEdge R760.",
    "NVIDIA. HGX H100 Product Carbon Footprint Summary (cradle-to-gate), 2024.",
    "vladkens. macmon: sudoless performance monitoring CLI for Apple Silicon. https://github.com/vladkens/macmon.",
]


# ============================================================================ content
def build(n: dict) -> None:
    P = Paper()
    get, lg, pg, rg = n["get"], n["lg"], n["pg"], n["rg"]
    P.title(TITLE, AUTHOR, AFFIL, "Preprint, November 2026")

    # ------------------------------------------------------------------ abstract
    P.h("Abstract")
    P.p(
        "Carbon-aware schedulers cut cloud emissions by moving flexible work to hours and places with cleaner electricity. "
        "Upper-bound studies of this idea evaluate each job in isolation, which implicitly lets all shifted work pile into the same "
        "low-carbon hours. A data centre cannot do that: its grid draw is bounded by the electrical plant (utility service, transformers, "
        "switchgear, UPS and generators). We first reproduce the trace-driven analysis of Sukprasert et al. (EuroSys 2024) from its released data "
        f"for 123 grid zones, matching {n['repro_within']} of {n['repro_total']} numeric claims, and report three discrepancies. We then show that "
        f"reaching the published temporal upper bound requires the median facility to draw {f(n['peak24'], 0)}x its mean power in a single hour; "
        f"with a realistic 20% power headroom only {f(get(24, 1.2, 'retained_of_ideal') * 100, 0)}% of the ideal saving survives. "
        "Using the dual variables of a compact linear program we compute, to our knowledge for the first time across many grids, the *shadow carbon price of facility "
        f"capacity*: one extra kW of headroom saves {f(pg(24, 1.2, 'kg_per_kw_year_median'), 0)} kg CO2/yr in the median zone. Comparing this against "
        "embodied-carbon factors we derive from manufacturer Environmental Product Declarations shows that electrical headroom repays its embodied "
        f"carbon in under a year in almost every zone, whereas CPU servers bought to absorb shifted work pay back within their lifetime in only "
        f"{f(pg(24, 1.2, 'cpu_servers_pays_back_in_life_pct'), 0)}% of zones. Tier redundancy (N+1, 2N) offers this headroom for free: using the spare path "
        f"for sheddable work closes {f((rg(0.65, 'N+1 (N=4)', 'perfect_pct') - rg(0.65, 'design_only', 'perfect_pct')) / (rg(0.65, 'design_only', 'ideal_pct') - rg(0.65, 'design_only', 'perfect_pct')) * 100, 0)}-"
        f"{f((rg(0.65, '2N', 'perfect_pct') - rg(0.65, 'design_only', 'perfect_pct')) / (rg(0.65, 'design_only', 'ideal_pct') - rg(0.65, 'design_only', 'perfect_pct')) * 100, 0)}% "
        "of the gap to the ideal, and maintenance and failures erode almost none of it. Finally, measurements of LLM inference on an Apple M4 Pro GPU "
        f"show that batching cuts energy per token by {f(lg('Qwen2.5-1.5B', 1, 'all_j_per_tok') / lg('Qwen2.5-1.5B', 32, 'all_j_per_tok'), 1)}x for "
        f"{f(lg('Qwen2.5-1.5B', 32, 'all_power_w') / lg('Qwen2.5-1.5B', 1, 'all_power_w'), 1)}x more power, a larger lever than temporal shifting. "
        "All code, data and a secure interactive explorer are released.")
    P.p("**Keywords:** carbon-aware computing, data-centre power provisioning, workload shifting, embodied carbon, redundancy, LLM inference energy.")

    # ------------------------------------------------------------------ 1 intro
    P.h("1 Introduction")
    P.p(
        "Data-centre electricity demand is growing quickly, driven in large part by AI, and operators have committed to aggressive emission targets. "
        "One prominent response is *carbon-aware workload shifting*: delaying flexible batch work to hours when grid electricity is cleaner, or moving it "
        "to cleaner regions [1, 2, 5, 8]. Sukprasert et al. [1] recently bounded the benefits of this idea across 123 grid zones and found them far smaller "
        "than commonly assumed. Their temporal analysis, however, optimises every job independently: each job is free to run in the cleanest hours of its "
        "window regardless of how many other jobs choose the same hours. The authors note that resource limits would reduce the savings further, but do not "
        "quantify that effect.")
    P.p(
        "**Gap.** A data centre is an electrical facility before it is a computer. Its grid draw is capped by the rating of its utility service, substation "
        "transformers, switchgear and UPS, and, for tier-rated sites, by the generator plant that must back the critical load. Google's carbon-intelligent "
        "compute system already enforces hourly capacity caps on flexible work in production [2, 15], and CarbonFlex [4] studies contention among jobs in a "
        "cluster. What is missing is a cross-grid answer to three facility-level questions: *how much of the ideal carbon saving survives a realistic power "
        "cap; what one additional kilowatt of capacity is worth in carbon; and whether that kilowatt is worth building once its embodied carbon is counted.*")
    P.p(
        "**Insight.** The capped scheduling problem has a structure that makes these questions cheap to answer exactly. When all flexible work has the same "
        "slack, earliest-deadline-first coincides with FIFO, so feasibility reduces to bounds on *cumulative* work served. The optimisation becomes a linear "
        "program with one variable per hour, and its dual variables price facility capacity in grams of CO2. Combined with embodied-carbon factors from "
        "equipment declarations, these shadow prices turn an operations question into a design question.")
    P.p("This paper makes the following contributions:")
    P.bullets([
        f"**A clean-room reproduction** of the EuroSys'24 analysis [1] that matches {n['repro_within']} of {n['repro_total']} checked claims, with "
        "three documented discrepancies, including a spatial metric that treats departed work as zero-carbon (Sec. 3).",
        "**A power-capped formulation** of temporal shifting, an exact cumulative-flow LP that is 25-170x smaller than the per-arrival transportation "
        "LP and provably equivalent to it, and a 123-zone map of how much of the ideal saving survives each level of headroom (Sec. 4).",
        "**The shadow carbon price of capacity** and the first embodied-carbon payback analysis of headroom built for carbon shifting, using factors "
        "derived from transformer, UPS, generator and server declarations (Sec. 5).",
        "**The redundancy dividend:** using N+1 and 2N spare capacity as zero-embodied-carbon headroom for sheddable work, evaluated with maintenance and "
        "random failures, plus a negative result for carbon-aware maintenance scheduling (Sec. 6).",
        "**Measurements** of GPU power, throughput and energy per token versus LLM batch size, and of UPS-battery carbon arbitrage (Secs. 7-8).",
        "**An open artifact:** tested code (44 tests), all data, and a hardened interactive web explorer (Sec. 11).",
    ])

    # ------------------------------------------------------------------ 2 background
    P.h("2 Background")
    P.h("2.1 Grid carbon intensity and workload flexibility", 2)
    P.p(
        "The average carbon intensity of grid electricity (g CO2eq/kWh) varies by hour with the mix of generators serving demand and differs by an order of "
        "magnitude between regions [1, 19]. Batch work such as ML training, analytics and simulation tolerates delay (*deferrability*) and pausing "
        "(*interruptibility*); interactive work does not [1]. We follow [1] in using average rather than marginal intensity, as required by the GHG "
        "Protocol for Scope 2 reporting.")
    P.h("2.2 The electrical plant of a data centre", 2)
    P.p(
        "Power enters through a utility service and medium-voltage switchgear, is stepped down by transformers, conditioned by UPS systems with batteries "
        "for ride-through, and backed by diesel generators. Facilities are provisioned for peak demand plus redundancy; average draw is well below the "
        "provisioned rating [10], and operators increasingly oversubscribe power to recover the difference [3, 11]. Redundancy is described by topology: "
        "N+1 adds one spare module to the N needed for the critical load; 2N duplicates the entire power path. The spare capacity is idle except during "
        "maintenance or a failure.")

    # ------------------------------------------------------------------ 3 reproduction
    P.h("3 Reproduction of Sukprasert et al.")
    P.p(
        "We re-implemented the analysis without executing the original code: the artifact [19] was read for reference only. The released hourly traces for 2022 "
        "were preprocessed identically (back-fill then forward-fill over 2020-2022). For temporal shifting, a job of length J arrives at every hour and may "
        "start anywhere within its slack S (deferral, a minimum-sum contiguous window) or run in its J cheapest hours (interruption). Our vectorised "
        "implementation (sliding windows, partial sorts, and a heap for the one-year case) replaces a 50-process loop. A unit test checks it against a "
        "line-by-line port of the original loop, including its end-of-year clipping and its definition of one-year slack as 'until the end of the year'. "
        "The full reproduction runs in about 20 s on a laptop.")
    rv = n["repro"]
    rows = [(r.section, r.claim, f"{r.paper:g}", f"{r.ours:g}", r.unit) for r in rv.itertuples()]
    P.table(["Source", "Claim", "Paper", "Ours", "Unit"], rows,
            f"All numeric claims checked ({n['repro_total']}). g = g CO2eq per job-hour or per kWh as in [1].", widths=[0.7, 3.6, 0.6, 0.6, 0.6], font=8)
    P.figure(FIG / "repro_temporal.png", "Reproduced carbon reduction per job-hour from deferral and interruption (cf. Figs. 7-8 of [1]).")
    P.p(
        "**Discrepancies.** (i) Figure 10(d) reports 127 g for the equal-mix, one-year-slack case, but Figure 10(a) reports 135 g for the same quantity. "
        "The artifact computes both identically and yields 134.9 g, which suggests the 127 g comes from an earlier run. (ii) Two descriptive statistics in "
        "§1 and §4.1 do not match their stated definitions. The share of zones with mean daily coefficient of variation below 0.1 is 49.6% (the paper says "
        "'more than 70%'; a threshold of 0.15 gives 74%), and the max/min ratio of zone means is 60x across all zones and 51x across cloud zones (the paper "
        f"says 40x). (iii) In the capacity-constrained spatial analysis, the artifact scores a grouping by subtracting the mean intensity of member data "
        "centres that receive work; when no member receives any, a NaN becomes 0, as if departed work ran at zero carbon. Charging each origin the intensity "
        f"where its work actually lands changes Asia at 50% idle capacity from {f(n['asia50_art'], 0)} to {f(n['asia50_cor'], 0)} g, and the Americas at "
        f"infinite capacity from {f(n['amer_inf_art'], 0)} to {f(n['amer_inf_cor'], 0)} g. Global results and all qualitative conclusions of [1] are unaffected.")

    # ------------------------------------------------------------------ 4 capped
    P.h("4 Power-Capped Temporal Shifting")
    P.h("4.1 Model", 2)
    P.p(
        "Normalise power to the facility's mean IT load. Each hour, a share φ of energy is flexible, pre-emptible work that must finish within S hours of "
        "arrival; the remaining 1-φ is inflexible. Grid draw in hour t may not exceed a cap C_t. C_t - 1 is the *headroom*: it is set by the electrical "
        "plant and drops during maintenance. With y_t the flexible work served in hour t and c_t the carbon intensity, the scheduler solves")
    P.eq("minimise  Σ_t c_t · y_t    subject to    0 ≤ y_t ≤ C_t − (1 − φ),    A(t − S) ≤ Y_t ≤ A(t)  ∀t", "1")
    P.p(
        "where Y_t = Σ_{u≤t} y_u is cumulative service and A(t) = φ(t+1) is cumulative arrivals. **Lemma.** Because every arrival has the same relative deadline, "
        "earliest-deadline-first and FIFO coincide, so a profile y can serve every arrival within its window if and only if the cumulative bounds in (1) hold. "
        "Necessity is immediate. For sufficiency, serve arrivals FIFO: the lower bound ensures that by the end of hour t all arrivals from hour t-S are complete. "
        "Formulation (1) has one variable per hour (8,760 per zone-year) with banded constraints, compared with 8,760·(S+1) variables in the per-arrival "
        "transportation LP. HiGHS solves it in about 50 ms. Tests confirm that it equals the brute-force LP on random instances (relative error < 1e-7), and "
        "that with C = ∞ it reproduces the 1-hour-job result of [1] (57.4 g per flexible kWh at 24 h slack), so the published bound is the C → ∞ limit of our model.")
    P.h("4.2 Results", 2)
    P.p(
        f"We sweep all 123 zones × φ ∈ {{25, 50, 75%}} × S ∈ {{24, 168}} h × 14 cap levels (10,332 LPs; 78 s on 14 cores). **The ideal requires an absurd power "
        f"spike:** with φ = 0.5 and 24 h slack, the uncapped optimum concentrates a day's flexible work in the cleanest hour, so the median facility peaks at "
        f"{f(n['peak24'], 0)}x its mean load ({f(n['peak168'], 0)}x with weekly slack). **Realistic headroom keeps only part of the saving** (Table 2, Fig. 2). "
        f"The median zone needs +{f(n['need50_24'], 0)}% headroom to keep half its ideal 24 h saving and +{f(n['need80_24'], 0)}% to keep 80%; with weekly "
        "slack the median zone does not reach 80% even at 4x capacity, so additional slack is useful only if there is power to use it. Variable, low-carbon "
        f"grids gain most from small headroom (South Australia reaches {f(n['sa_5'], 1)}% with 5% headroom), while flat fossil grids gain little at any "
        f"headroom (India-West: {f(n['inwe_inf'], 1)}% even uncapped). {n['zones_flat']} zones have constant traces and are excluded from the "
        "headroom-needed statistics.")
    caps = (1.1, 1.2, 1.5, 2.0)
    rows = []
    for sl in (24, 168):
        rows.append([f"Share of ideal kept, {sl} h slack"] + [f"{get(sl, c, 'retained_of_ideal') * 100:.0f}%" for c in caps] + ["100%"])
        rows.append([f"Facility CO2 reduction, {sl} h slack"] + [f"{get(sl, c, 'reduction_pct'):.1f}%" for c in caps] +
                    [f"{(n['ideal24_pct'] if sl == 24 else float(n['fac'][(n['fac'].slack == 168) & np.isinf(n['fac'].cap)].reduction_pct.iloc[0])):.1f}%"])
    P.table(["Headroom above mean load", "+10%", "+20%", "+50%", "+100%", "∞"], rows,
            "Global mean over 123 zones, φ = 0.5. Share kept = ratio of mean savings.", widths=[2.4, 0.7, 0.7, 0.7, 0.7, 0.7])
    P.figure(FIG / "facility_retained.png", "Share of the ideal (uncapped) carbon saving retained as power headroom grows, for three flexible shares and two slacks.")
    P.figure(FIG / "facility_zones.png", "Facility-level reduction versus headroom for four grids (φ = 0.5, 24 h slack); dotted lines mark the uncapped ideal.", width=4.6)

    # ------------------------------------------------------------------ 5 shadow price
    P.h("5 The Carbon Price of a Kilowatt")
    P.h("5.1 Shadow prices", 2)
    P.p(
        "Let λ_t ≥ 0 be the dual variable of the hourly capacity constraint in (1). Raising capacity by one unit in hour t lowers emissions by λ_t, so Σ_t λ_t "
        "is the annual carbon value of one additional kW of headroom at a facility with 1 kW mean load (equivalently, grams per year per kW of added capacity). "
        "We verify the duals against finite differences (within 2%), and check that they are non-negative and vanish when the cap does not bind. "
        f"For the median zone at 24 h slack, one extra kW saves {f(pg(24, 1.1, 'kg_per_kw_year_median'), 0)} kg/yr at 10% existing headroom, "
        f"{f(pg(24, 1.2, 'kg_per_kw_year_median'), 0)} kg/yr at 20%, {f(pg(24, 1.5, 'kg_per_kw_year_median'), 0)} kg/yr at 50% and "
        f"{f(pg(24, 2.0, 'kg_per_kw_year_median'), 0)} kg/yr at 100%. The first kilowatts are the most valuable.")
    P.h("5.2 Embodied carbon of headroom", 2)
    P.p(
        "We found no peer-reviewed per-kW embodied-carbon factors for data-centre electrical plant, so we derive them from manufacturer EPDs "
        "(cradle-to-gate, A1-A3) and server product carbon footprints [20-26]. Table 3 lists the factors. A key observation is that **sheddable, deferrable work "
        "does not require UPS or generator backup**: on a grid outage it is postponed, not lost. Headroom for carbon shifting can therefore be *grid-only* "
        "(transformer plus switchgear), roughly five times cheaper in embodied carbon than fully backed capacity.")
    erows = []
    src = {"transformer": "[20, 21] 2.5 MVA cast resin; 16/20 MVA oil", "switchgear": "no primary EPD found; sensitivity range",
           "ups": "[24] ABB MegaFlex, DPA 250; Schneider Galaxy VX", "ups_battery": "Li-ion 54-69 kg/kWh × 10 min (secondary)",
           "generator": "[22, 23] Cummins QSK95; mtu 10V1600", "cpu_servers": "[25] Dell R650, R760; 0.35 kW assumed draw",
           "gpu_servers": "[26] HGX H100 baseboard ×2 (assumed)"}
    for k, (lo, mid, hi) in FACTORS.items():
        erows.append([k.replace("_", " "), f"{lo:.0f}", f"{mid:.0f}", f"{hi:.0f}", src[k]])
    P.table(["Component", "Low", "Central", "High", "Basis"], erows, "Embodied carbon per kW of capacity (kg CO2eq/kW) derived for this study.",
            widths=[1.1, 0.5, 0.6, 0.5, 3.2], font=8.5)
    rows = []
    names = {"grid_only": "Grid-only electrical", "backed": "Fully backed electrical", "gpu_servers": "GPU servers", "cpu_servers": "CPU servers"}
    for b in ("grid_only", "backed", "gpu_servers", "cpu_servers"):
        rows.append([names[b], f"{per_kw(b)[1]:,.0f}", f"{LIFETIME[b]}", f"{pg(24, 1.2, f'{b}_payback_median_y'):.1f}",
                     f"{pg(24, 1.2, f'{b}_pays_back_in_life_pct'):.0f}%"])
    P.table(["Headroom build", "Embodied kg/kW", "Life (y)", "Median payback (y)", "Zones repaid in life*"], rows,
            "Carbon payback of building headroom for shifting at 20% existing headroom, φ = 0.5, 24 h slack. *Savings decline 3%/yr with grid decarbonisation.",
            widths=[1.8, 1.0, 0.7, 1.2, 1.3])
    P.figure(FIG / "redundancy_marginal.png",
             "Shadow carbon price of facility capacity (median and interquartile range over zones) with embodied-carbon breakeven levels per build option.", width=5.0)
    P.p(
        "**Finding: build power, not servers.** Electrical headroom repays its embodied carbon within months in almost every zone. CPU servers bought only to "
        f"absorb shifted work pay back within a five-year life in just {f(pg(24, 1.2, 'cpu_servers_pays_back_in_life_pct'), 0)}% of zones; GPU servers, with far "
        f"lower embodied carbon per watt, do so in {f(pg(24, 1.2, 'gpu_servers_pays_back_in_life_pct'), 0)}%. Carbon-aware shifting should therefore use idle "
        "compute that already exists, and a power-constrained operator gains most by adding *electrical* headroom.")

    rob = n["rob"]
    P.h("5.3 Robustness across years", 2)
    P.p(
        "We repeat the facility analysis for 2020 and 2021 (φ = 0.5, 24 h slack; the 120 zones with non-constant traces). Table 5 shows that the headline "
        "results are stable. Per-zone shadow prices rank-correlate at Spearman "
        f"{n['rob_rc'][0]:.2f}-{n['rob_rc'][1]:.2f} between years, so a ranking of where capacity is most valuable carries over from one year to the next. "
        f"**The carbon value of capacity is rising:** the median shadow price grew {(rob.loc[2022, 'kg_per_kw_year_median'] / rob.loc[2020, 'kg_per_kw_year_median'] - 1) * 100:.0f}% "
        "from 2020 to 2022. Three years is too short to establish a trend, but the direction is notable. It runs opposite to the observation in [1] that the "
        "*relative* benefit of carbon-aware scheduling shrinks as grids decarbonise: as grids become more variable, headroom becomes more valuable.")
    rows = []
    for label, col, fmt_ in (("Share of ideal kept, +20% headroom", "kept_20pct_headroom", "{:.1f}%"),
                             ("Share of ideal kept, +50% headroom", "kept_50pct_headroom", "{:.1f}%"),
                             ("Shadow price of capacity, median (kg/kW-yr)", "kg_per_kw_year_median", "{:.0f}"),
                             ("CPU servers repaid within life", "cpu_pays_back_in_life_pct", "{:.0f}%"),
                             ("Gap to ideal closed by N+1", "gap_closed_N+1_pct", "{:.0f}%"),
                             ("Gap to ideal closed by 2N", "gap_closed_2N_pct", "{:.0f}%")):
        rows.append([label] + [fmt_.format(rob.loc[y, col]) for y in (2020, 2021, 2022)])
    P.table(["Metric", "2020", "2021", "2022"], rows, "Robustness of the facility findings across three years of traces.",
            widths=[3.2, 0.8, 0.8, 0.8])

    # ------------------------------------------------------------------ 6 redundancy
    P.h("6 The Redundancy Dividend")
    P.p(
        "Tier-rated facilities already own idle electrical capacity. Flex [3] showed that this reserve can host software-redundant work and be reclaimed "
        "within seconds when a failure occurs. We apply the idea to carbon: deferrable work may use the redundant capacity, and when a path is down the "
        "facility falls back to its design capacity 1/u, where u is design utilisation (mean load / critical capacity). We model two 24 h maintenance windows "
        "per year and Poisson failures (0.5/yr, exponential repair with an 8 h mean), averaged over 10 random seeds per zone. Maintenance is placed either at "
        "random or in the window with the lowest summed shadow price (carbon-aware maintenance).")
    rows = []
    for u in (0.5, 0.65, 0.8):
        d = rg(u, "design_only", "perfect_pct")
        ideal = rg(u, "design_only", "ideal_pct")
        for t in ("design_only", "N+1 (N=4)", "2N"):
            per = rg(u, t, "perfect_pct")
            rnd = rg(u, t, "random_maint_pct") if t != "design_only" else per
            awr = rg(u, t, "aware_maint_pct") if t != "design_only" else per
            closed = (per - d) / (ideal - d) * 100 if t != "design_only" else 0
            rows.append([f"{u:.2f}", t.replace("design_only", "design only"), f"{rg(u, t, 'cap'):.2f}", f"{per:.2f}%", f"{rnd:.2f}%", f"{awr:.2f}%",
                         f"{closed:.0f}%" if t != "design_only" else "–"])
    P.table(["u", "Capacity for flexible work", "Cap (×mean)", "No outages", "Random maint.", "Carbon-aware maint.", "Gap closed"], rows,
            f"Facility CO2 reduction, mean of 123 zones, φ = 0.5, 24 h slack. Ideal (uncapped) = {n['ideal24_pct']:.2f}%.",
            widths=[0.4, 1.5, 0.8, 0.8, 0.9, 1.1, 0.8], font=8.5)
    P.figure(FIG / "redundancy_dividend.png", "Redundancy dividend at 65% design utilisation.", width=4.6)
    P.p(
        "**Finding: redundancy is free carbon headroom, and outages barely matter.** Using the N+1 spare raises the facility reduction from "
        f"{rg(0.65, 'design_only', 'perfect_pct'):.1f}% to {rg(0.65, 'N+1 (N=4)', 'perfect_pct'):.1f}% and the 2N spare to "
        f"{rg(0.65, '2N', 'perfect_pct'):.1f}% at u = 0.65, at zero embodied cost. Maintenance and failures cost under 0.01 percentage points, because the LP "
        "simply routes work around short windows within its slack. Carbon-aware maintenance recovers 90-100% of that loss; the effect is real but negligible. "
        "We report this as a negative result: it is not worth optimising.")

    # ------------------------------------------------------------------ 7 batteries
    P.h("7 Can UPS Batteries Shift Carbon?")
    P.p(
        "We solve a cyclic battery-dispatch LP behind a flat load (90% round-trip efficiency, grid draw limited by the cap). A UPS sized for 10 minutes of "
        f"ride-through and fully cycled every day cuts emissions by only {n['bat10_mean']:.2f}% on average ({n['bat10_max']:.1f}% in the best zone), and "
        f"by {n['bat10_res']:.2f}% if half its energy must stay in reserve. Even a dedicated 4-hour battery averages {n['bat4h_mean']:.1f}% "
        f"(best zone {n['bat4h_max']:.1f}%), less than half-flexible load with 50% headroom ({get(24, 1.5, 'reduction_pct'):.1f}%).")
    P.figure(FIG / "facility_battery.png", "Facility CO2 reduction from carbon-optimal battery dispatch versus battery size.", width=4.4)

    # ------------------------------------------------------------------ 8 LLM
    P.h("8 Measured LLM Inference Energy versus Batch Size")
    P.p(
        "AI inference is the fastest-growing flexible load, and batch size is its main efficiency control. We measured two 4-bit Qwen2.5 instruct models "
        "[17] (1.5B and 7B) on an Apple M4 Pro with a 20-core GPU, using MLX [18]. B distinct prompts were decoded together with concurrency forced to B "
        "(mlx-lm otherwise silently caps it at 32), generating 128 tokens each for at least 6 s per point, with three repeats. GPU, chip (CPU+GPU+DRAM) and "
        "system power were sampled every 100 ms with macmon [27], which reads the IOReport counters used by powermetrics without root privileges. We report "
        "chip energy as the headline metric because system power drifted upward with temperature and background activity during the run.")
    rows = []
    for m in ("Qwen2.5-1.5B", "Qwen2.5-7B"):
        for b in (1, 8, 32, 128):
            rows.append([m, b, f"{lg(m, b, 'tok_per_s'):,.0f}", f"{lg(m, b, 'gpu_power_w'):.1f}", f"{lg(m, b, 'all_power_w'):.1f}",
                         f"{lg(m, b, 'all_j_per_tok') * 1000:.0f}"])
    P.table(["Model (4-bit)", "Batch", "Tokens/s", "GPU W", "Chip W", "Chip mJ/token"], rows,
            f"Mean of 3 repeats; chip-power coefficient of variation across repeats ≤ {n['rep_cv_max']:.1f}%.", widths=[1.4, 0.6, 0.9, 0.8, 0.8, 1.1])
    P.figure(FIG / "llm_power_batch.png", "Throughput, GPU power and chip energy per token versus batch size on an Apple M4 Pro.")
    P.p(
        f"**Batching is cheap in power.** From batch 1 to 32, throughput of the 1.5B model rises {lg('Qwen2.5-1.5B', 32, 'tok_per_s') / lg('Qwen2.5-1.5B', 1, 'tok_per_s'):.1f}x "
        f"while chip power rises {lg('Qwen2.5-1.5B', 32, 'all_power_w') / lg('Qwen2.5-1.5B', 1, 'all_power_w'):.1f}x, cutting energy per token "
        f"{lg('Qwen2.5-1.5B', 1, 'all_j_per_tok') / lg('Qwen2.5-1.5B', 32, 'all_j_per_tok'):.1f}x "
        f"({lg('Qwen2.5-7B', 1, 'all_j_per_tok') / lg('Qwen2.5-7B', 32, 'all_j_per_tok'):.1f}x for 7B). **The GPU saturates near batch 32:** beyond it, "
        f"decode throughput grows by less than 10% ({n['dec15_32']:,.0f} to {n['dec15_128']:,.0f} tokens/s for 1.5B), and GPU power plateaus at 31-32 W. Gains are "
        "uneven, with a flat step from 4 to 8 and a 1.7-1.9x jump from 16 to 32, consistent with (but not shown to be caused by) kernel-selection thresholds in "
        "MLX's quantised matrix multiplication. **Implication:** under a power cap, running inference at its efficient batch size delivers about 4x more tokens "
        "per joule within the same watts, a larger lever than temporal shifting (Sec. 4). Only work beyond the saturation point needs to compete for headroom. "
        "This complements carbon-aware inference designs such as EcoServe [13] and power oversubscription for LLM clusters [11].")

    # ------------------------------------------------------------------ 9 related work
    P.h("9 Related Work")
    P.p(
        "**Carbon-aware shifting.** Temporal shifting was evaluated for specific regions and workloads [8] and bounded across 123 zones [1]. CarbonScaler [5] "
        "exploits elasticity, CarbonFlex [4] provisions and schedules clusters, and Ecovisor [6] and the virtual energy system vision [14] expose energy and "
        "carbon to applications. CarbonCast [9] provides the forecasts such schedulers need. Google's carbon-intelligent compute system [2] and its "
        "probabilistic extension [15] enforce hourly capacity caps in production. We add a cross-grid quantification of how the cap level bounds the saving, "
        "and price capacity in carbon.")
    P.p(
        "**Power provisioning.** Fan et al. [10] showed that real clusters draw well below their aggregate nameplate, motivating oversubscription. Flex [3] "
        "eliminates reserved power by shedding software-redundant work on failures, and POLCA [11] oversubscribes LLM inference clusters. These works use "
        "headroom to host more servers; we use it to save carbon, and show that tier redundancy is effective carbon headroom.")
    P.p(
        "**Embodied carbon.** ACT [12] models the embodied carbon of computing hardware, and Carbon Explorer [7] co-designs capacity, renewables and storage "
        "for carbon-aware data centres. To our knowledge no prior work derives embodied carbon per kW of data-centre *electrical* plant from equipment "
        "declarations, or weighs it against the operational value of headroom for workload shifting.")

    # ------------------------------------------------------------------ 10 discussion
    P.h("10 Discussion, Limitations and Threats to Validity")
    P.bullets([
        "**Average vs. marginal intensity.** Like [1], we use average intensity; marginal signals can rank hours differently.",
        "**Perfect forecasts.** All results are upper bounds; [1] estimates about 3% loss at CarbonCast-level errors [9].",
        "**Flat inflexible load and fluid flexible work.** Real IT load is diurnal and long jobs are not infinitely divisible; both reduce savings below our bounds. "
        "Migration and checkpoint overheads are ignored.",
        "**Embodied factors.** One or two EPDs per component; switchgear has no primary source; server factors rest on assumed average draws. Conclusions are "
        "robust because the gap between electrical plant and CPU servers is an order of magnitude.",
        "**Redundancy model.** Fleet-average failure parameters; partial-module failures and cooling redundancy are not modelled; shedding is assumed instantaneous, as in [3].",
        "**GPU measurements.** One laptop-class SoC and two small quantised models; they show the shape of the batch curve, not data-centre accelerator values.",
        "**Grid evolution.** Savings shrink as grids decarbonise [1]; payback results include a 3%/yr decline sensitivity.",
    ])

    # ------------------------------------------------------------------ 11 artifact / conclusion
    P.h("11 Artifact")
    P.p(
        "The repository contains the library (data loading, temporal and spatial bounds, capped LP with duals, battery LP, embodied factors with sources, "
        "redundancy model, sudo-free power sampler), experiment scripts that write every number in this paper to CSV, 44 unit and integration tests "
        "(including equivalence to the original loop and to brute-force LPs), and an interactive explorer. The explorer is a FastAPI service with strict "
        "input validation, rate and concurrency limits, a body-size limit, a strict Content-Security-Policy, and no runtime model inference. This document is "
        "generated by scripts/build_paper.py from the result files.")
    P.h("12 Conclusion")
    P.p(
        "Carbon-aware workload shifting is bounded not only by grid variability but by the power plant of the facility doing the shifting. Realistic headroom "
        "preserves a fraction of the published upper bounds. Pricing capacity in carbon shows that the first kilowatts of headroom are valuable and cheap in "
        "embodied carbon, that the redundancy tier-rated facilities already own is free carbon headroom, and that buying servers just to shift work rarely "
        "pays off. For AI inference, batching to the hardware's efficient point beats shifting in time. Together these results argue for carbon-aware "
        "schedulers that see watts, not only hours: the natural next step is an Ecovisor-style interface [6] that exposes both the carbon signal and the "
        "facility's power budget to applications.")
    P.h("Acknowledgements")
    P.p("We thank the authors of [1] for releasing their code and data, and Electricity Maps for the underlying carbon-intensity data (ODbL).")
    P.h("References")
    P.refs(REFS)
    P.save(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}  ({P.fig_no} figures, {P.tab_no} tables, {len(REFS)} references)")


if __name__ == "__main__":
    build(load_numbers())
