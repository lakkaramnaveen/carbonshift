"""Generate paper/sigmetrics/refs.bib from the REFS lists of the workshop drafts.

Each REFS entry is a plain string "Authors. Title. Venue details, Year." The lists are read
with ast (the draft scripts are not executed), de-duplicated by title, and written as
@misc entries with author / title / howpublished / year, plus a stable key
(first author's surname + year + first title word). The keys are printed so that main.tex
can \\cite them.

Run: python scripts/build_bib.py
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted((ROOT / "scripts").glob("build_workshop*.py"))
OUT = ROOT / "paper" / "sigmetrics" / "refs.bib"

# references used only by the paper (F31, the waterbed), in the same plain format
EXTRA = [
    "G. Perino. New EU ETS Phase 4 Rules Temporarily Puncture Waterbed. Nature Climate Change 8(4), 262-264, 2018.",
    "K. E. Rosendahl. EU ETS and the Waterbed Effect. Nature Climate Change 9, 734-735, 2019.",
    "G. Perino, R. A. Ritz, A. van Benthem. Overlapping Climate Policies. NBER Working Paper 25643, 2019.",
    "European Parliament and Council. Directive (EU) 2023/959 amending Directive 2003/87/EC and Decision (EU) 2015/1814 on the Market Stability Reserve. Official Journal of the EU, 2023.",
    "European Commission. Publication of the Total Number of Allowances in Circulation in 2025 for the Purposes of the Market Stability Reserve. Communication C/2026/2957, 2026.",
    "J. Lopez Prol, W.-P. Schill. The Economics of Variable Renewable Energy and Electricity Storage. Annual Review of Resource Economics 13, 443-467, 2021.",
    "B. Acun, B. Lee, F. Kazhamiaka, K. Maeng, U. Gupta, M. Chakkaravarthy, D. Brooks, C.-J. Wu. Carbon Explorer: A Holistic Framework for Designing Carbon Aware Datacenters. ASPLOS, 2023.",
    "Council of the EU. ETS Market Stability Reserve: Council and Parliament Reach Provisional Agreement on Targeted Amendment. Press release, 21 October 2026.",
    "L. A. Barroso, U. Holzle. The Case for Energy-Proportional Computing. IEEE Computer 40(12), 33-37, 2007.",
    "N. Bashir, V. Gohil, A. Belavadi Subramanya, M. Shahrad, D. Irwin, E. Olivetti, C. Delimitrou. The Sunk Carbon Fallacy: Rethinking Carbon Footprint Metrics for Effective Carbon-Aware Scheduling. ACM SoCC, 542-551, 2024.",
    "H. Tropgen, R. Schone, T. Ilsche, D. Hackenberg. 16 Years of SPEC Power: An Analysis of x86 Energy Efficiency Trends. IEEE Cluster Workshops, 2024.",
    # F34 (price of a shifted tonne)
    "U.S. Environmental Protection Agency. Report on the Social Cost of Greenhouse Gases: Estimates Incorporating Recent Scientific Advances. EPA report, 2023.",
    "Bloomberg New Energy Finance. Energy Storage Systems Cost Survey 2025. BNEF report, 2025.",
    "Turner & Townsend. Data Centre Construction Cost Index 2025-2026. Industry report, 2025.",
    "J. Lindberg, Y. Abdennadher, J. Chen, B. C. Lesieutre, L. Roald. A Guide to Reducing Carbon Emissions through Data Center Geographical Load Shifting. ACM e-Energy, 2021.",
    # F35 (who produced the saving)
    "L. Han, J. Kakadia, B. C. Lee, U. Gupta. Fair-CO2: Fair Attribution for Cloud Carbon Emissions. ISCA, 646-663, 2025.",
    "D. Maji, N. Bashir, D. Irwin, P. Shenoy, R. K. Sitaraman. The Green Mirage: Impact of Location- and Market-based Carbon Intensity Estimation on Carbon Optimization Efficacy. ACM e-Energy, 2024.",
    "P. H. Wicksteed. An Essay on the Co-ordination of the Laws of Distribution. Macmillan, 1894.",
    "R. J. Aumann, L. S. Shapley. Values of Non-Atomic Games. Princeton University Press, 1974.",
]

CONFERENCES = ("EuroSys", "e-Energy", "ASPLOS", "Middleware", "BuildSys", "SIGMETRICS", "FOCS", "HotCarbon",
               "HPCA", "SoCC", "Cluster", "ISCA", "NSDI", "OSDI", "SOSP")


def entry_type(venue: str) -> tuple[str, str]:
    """(BibTeX type, field) for a venue string: conference -> inproceedings, journal -> article, else misc."""
    head = re.sub(r",?\s*(?:18|19|20)\d\d.*$", "", venue).strip()
    if any(c in head for c in CONFERENCES):
        return "inproceedings", "booktitle"
    if re.search(r"\d+\(\d+\)|\b\d+, \d", venue) and not re.search(r"Working Paper|Communication|White paper", venue):
        return "article", "journal"
    return "misc", "howpublished"


def read_refs(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "REFS" for t in node.targets):
            return list(ast.literal_eval(node.value))
    return []


def parse(ref: str) -> dict:
    parts = [p.strip() for p in re.split(r"(?<=[a-z0-9\)\]'])\.\s+(?=[A-Z0-9\"'])", ref.strip()) if p.strip()]
    authors = parts[0] if parts else ref
    title = parts[1] if len(parts) > 1 else ""
    rest = ". ".join(parts[2:]).rstrip(".")
    years = re.findall(r"(?:18|19|20)\d\d", ref)
    year = years[-1] if years else ""
    return {"authors": authors, "title": title.rstrip("."), "venue": rest, "year": year}


def bib_authors(a: str) -> str:
    a = re.sub(r",?\s+et al\.?", " and others", a)
    names = [n.strip() for n in re.split(r",\s*|\s+&\s+|\s+and\s+(?!others)", a) if n.strip()]
    return " and ".join(names)


def key(e: dict) -> str:
    first = re.split(r",|\s+and\s+|&", re.sub(r",?\s+et al\.?", "", e["authors"]))[0].strip()
    surname = re.sub(r"[^A-Za-z]", "", first.split()[-1]) if first else "anon"
    word = re.sub(r"[^A-Za-z]", "", (e["title"].split() or ["x"])[0]).lower()
    return f"{surname.lower()}{e['year']}{word}"


def esc(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def main() -> None:
    seen, entries = set(), []
    refs = [r for src in SOURCES for r in read_refs(src)] + EXTRA
    for r in refs:
        if "Workshop draft" in r:  # unpublished self-citations break anonymity
            continue
        e = parse(r)
        if not e["title"]:  # unparseable: keep the whole string as the title
            e = {"authors": "", "title": r.rstrip("."), "venue": "", "year": e["year"]}
        t = e["title"].lower()
        if t in seen:
            continue
        seen.add(t)
        e["key"] = key(e)
        entries.append(e)
    keys = {}
    for e in entries:  # make keys unique
        k = e["key"]
        keys[k] = keys.get(k, 0) + 1
        if keys[k] > 1:
            e["key"] = f"{k}{chr(ord('a') + keys[k] - 1)}"
    lines = ["% Generated by scripts/build_bib.py from the REFS lists in scripts/build_workshop*.py.",
             "% Conference and journal entries are typed from their venue strings; check them against the",
             "% publishers' records (and add DOIs) before submission.", ""]
    for e in entries:
        kind, field = entry_type(e["venue"])
        venue = e["venue"]
        if kind != "misc":  # the year has its own field
            venue = re.sub(r",?\s*(?:19|20)\d\d\s*$", "", re.sub(r"^(?:ACM|IEEE)\s+(?=e-Energy|SoCC|Cluster)", "", venue)).strip()
        lines += [f"@{kind}{{{e['key']},",
                  f"  author = {{{esc(bib_authors(e['authors']))}}},",
                  f"  title = {{{{{esc(e['title'])}}}}},",
                  f"  {field} = {{{esc(venue)}}},",
                  f"  year = {{{e['year']}}},", "}", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"{len(entries)} entries -> {OUT.relative_to(ROOT)}")
    for e in entries:
        print(f"  {e['key']}: {e['title'][:70]}")


if __name__ == "__main__":
    main()
