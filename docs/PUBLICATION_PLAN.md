# Publication plan

Last updated 2026-11-20 (research run 30, F28). Nothing has been submitted anywhere; every submission is the author's decision.

## 1. Venues and deadlines (checked online 2026-11-20)

| Venue | Format | Deadline | Status of the date | Fit |
|---|---|---|---|---|
| **ACM SIGMETRICS 2027, winter cycle** (POMACS journal track; conference June 7-11, 2027, Atlanta, at FCRC) | full paper, double-blind | abstract **Jan 4, 2027**, paper **Jan 11, 2027** (AoE); notification Mar 10, 2027 | from the SIGMETRICS 2027 site (sigmetrics.org/sigmetrics2027) and two deadline trackers that agree. The fall cycle (Oct 9, 2026) has passed. The spring cycle is not yet posted | theory + measurement: the closed forms and virtual-battery theorem (F21-F23) |
| **ACM e-Energy 2027** (ACM Sustainability Week, with BuildSys) | full paper | **not yet announced**. 2026 was Jan 22, 2026 (Banff, June 22-25, 2026) | unverified for 2027; expect late January 2027 by pattern. Check energy.acm.org | energy-systems audience; the F28 accounting result fits best here, or the battery paper as fallback |
| **ACM HotCarbon 2027** | 5-page workshop paper, double-blind | **not yet announced**. 2026 was May 11 (registration) / May 18, 2026 (submission) | unverified; hotcarbon.org still shows the 2026 call | short, provocative results: F28 ("Moved, Not Cut") is the best candidate |
| ACM BuildSys 2027 | full paper | not yet announced; 2026 was Jan 22, 2026 (co-located with e-Energy) | unverified | weaker fit (buildings); only the cooling papers (F26-F27) |
| arXiv (cs.DC or eess.SY) | preprint | none | first-time submitters may need an endorsement | post the preprint when a venue version exists (the author's call) |

## 2. The strongest single-paper story

**Full paper: "Your Data Centre Is Already a Battery: A Closed-Form Theory of Carbon-Aware Shifting under Power Caps."**
This is the deepest original contribution and has one idea throughout:

1. *Problem* (F3, F4, F12): the EuroSys'24 upper bounds need 13x peak draw. Real facilities have about +42% headroom and keep about half the ideal saving.
2. *Theory* (F21, F22): the water-filling law (sinc closed form, homogeneity in h/φ, water-level shadow price) and the theorem that capped flexible load is a lossless battery. Together they explain F3, F4, F5, F7 and F15 as corollaries.
3. *Price* (F23): the exchange rate against real batteries (4.3 kWh per kW at +20%), plus embodied carbon (F8, F16).
4. *Fleet consequence* (F24): the power cap is an anti-herding device.
5. *Validation*: 4,680 LP solves, 3 years, out-of-year checks.

**Workshop paper: "Moved, Not Cut: Who Bears the Cost of Carbon-Aware Computing?"** (F28 with F17, F19 and F20). This is the result most likely to change how people think: location-based savings on most US grids are transfers onto other consumers, and a two-part rule fixes the incentive. It is timely because the GHG Protocol's Scope 2 revision is in progress. Draft: `scripts/build_workshop_incidence.py` builds a 2-page double-blind PDF.

The other drafts are kept for reuse: `build_workshop.py` (signals, F17/F19/F20; superseded by the incidence draft), `build_workshop_battery.py` (F21-F23; the seed of the full paper) and `build_workshop_cooling.py` (F26-F27; a BuildSys/e-Energy candidate).

## 3. Still needed before a full submission

- [ ] **Template.** Port to LaTeX `acmart` (sigconf for e-Energy/HotCarbon; the SIGMETRICS/POMACS format for SIGMETRICS). No LaTeX is installed in the build machine; the reportlab drafts are for content only.
- [ ] **Proofs.** Write formal statements and proofs in an appendix: the homogeneity theorem, the virtual-battery equivalence (and its boundary term), and the water-level shadow price. For F28, the incidence identity and the two-part properties: budget balance, and exact alignment under linear response.
- [ ] **Related work, deeper.** Virtual storage for flexible loads (Callaway & Hiskens 2011; Hao et al. 2015), valley filling for EV charging (Gan, Topcu & Low 2013), water-filling in power allocation, carbon accounting (Gorka, Rhodes & Roald 2024; WattTime 2022; GHG Protocol consultation 2025-26; Aumann-Shapley allocation). Read the full texts: they could not be fetched in this session, so the novelty claims are scoped to the abstracts and summaries.
- [ ] **Threats to validity.** One section collecting the following:
  - average vs marginal signals;
  - season × hour MEFs, 2022, US only;
  - fluid work and perfect forecasts;
  - modelled cooling plants;
  - simulated meters (F25);
  - the constant gas margin (F24, F28 CAISO check).
- [ ] **Figures.** Vector PDF figures with one consistent style, sized for two columns. The draft PNGs are at 200 dpi.
- [ ] **Anonymisation.** Double-blind venues need the drafts stripped of author names and the repository link. The new incidence draft is already anonymous; the other three are not.
- [ ] **Artifact.** A tagged release with `reproduce.py` + the experiment scripts and the test suite (124 tests). Consider a Zenodo DOI at camera-ready.
- [ ] **External data gap.** F28 and F17-F20 need marginal factors outside the US. Any free source would strengthen the claim that the result generalises.

## 4. Single next concrete step

**Decide between SIGMETRICS (Jan 11, 2027) and e-Energy (expected late Jan 2027) for the battery paper, then port `build_workshop_battery.py`'s content into an `acmart` LaTeX skeleton with a formal theorem/proof appendix.** Register the SIGMETRICS abstract by Jan 4 if that venue is chosen. In parallel, keep the F28 workshop draft ready for HotCarbon 2027 once its call is posted.
