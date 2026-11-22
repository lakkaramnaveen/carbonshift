# Publication plan

Last updated 2026-11-22 (research run 32, F30). Previous updates 2026-11-21 (run 31, F29) and 2026-11-20 (run 30, F28). Nothing has been submitted anywhere; every submission is the author's decision.

## 1. Venues and deadlines (checked online 2026-11-20, re-checked 2026-11-21 and 2026-11-22)

| Venue | Format | Deadline | Status of the date | Fit |
|---|---|---|---|---|
| **ACM SIGMETRICS 2027, winter cycle** (re-checked 2026-11-22: the official CFP page lists a winter deadline and three cycles; trackers give paper Jan 11, 2027 and abstract Jan 4, 2027 (abstract date not on the official page); the official CFP page could not be fetched from this session, so confirm it there) (POMACS journal track; conference June 7-11, 2027, Atlanta, at FCRC) | full paper, double-blind | abstract **Jan 4, 2027**, paper **Jan 11, 2027** (AoE); notification Mar 10, 2027 | from the SIGMETRICS 2027 site (sigmetrics.org/sigmetrics2027) and two deadline trackers that agree. The fall cycle (Oct 9, 2026) has passed. The spring cycle is not yet posted | theory + measurement: the closed forms and virtual-battery theorem (F21-F23) |
| **ACM e-Energy 2027** (ACM Sustainability Week, with BuildSys) | full paper | **not yet announced**. 2026 was Jan 22, 2026 (Banff, June 22-25, 2026) | unverified for 2027; expect late January 2027 by pattern. Check energy.acm.org | energy-systems audience; the F28 accounting result fits best here, or the battery paper as fallback |
| **ACM HotCarbon 2027** | 5-page workshop paper, double-blind | **not yet announced**. 2026 was May 11 (registration) / May 18, 2026 (submission) | unverified; on 2026-11-22 only a mirror copy of the 2026 call with 2027 dates was found, so it is not trusted | short, provocative results: F28 ("Moved, Not Cut") is the best candidate |
| ACM BuildSys 2027 | full paper | not yet announced; 2026 was Jan 22, 2026 (co-located with e-Energy) | unverified | weaker fit (buildings); only the cooling papers (F26-F27) |
| arXiv (cs.DC or eess.SY) | preprint | none | first-time submitters may need an endorsement | post the preprint when a venue version exists (the author's call) |

## 2. The strongest single-paper story

**Full paper: "Your Data Centre Is Already a Battery: A Closed-Form Theory of Carbon-Aware Shifting under Power Caps."**
This is the deepest original contribution and has one idea throughout:

1. *Problem* (F3, F4, F12): the EuroSys'24 upper bounds need 13x peak draw. Real facilities have about +42% headroom and keep about half the ideal saving.
2. *Theory* (F21, F22): the water-filling law (sinc closed form, homogeneity in h/φ, water-level shadow price) and the theorem that capped flexible load is a lossless battery. Together they explain F3, F4, F5, F7 and F15 as corollaries.
3. *Price* (F23): the exchange rate against real batteries (4.3 kWh per kW at +20%), plus embodied carbon (F8, F16).
4. *Boundary of the theory* (F29, new): not every knob is a battery. Deferral is linear in utilisation, so its saving is first order in the daily swing; carbon-aware speed scaling (DVFS, GPU power limits) is convex, so its saving is the power-mean gap 1 - M_p(c)/M_1(c), second order in CV (a Harberger triangle): a median 0.79% vs 14.6% uncapped. This gives the paper a sharp, quotable contrast ("the carbon knob is *when*, not *how fast*") and pre-empts the reviewer question "why not just throttle?".
5. *Fleet consequence* (F24): the power cap is an anti-herding device.
6. *Validation*: 4,680 LP solves, 3 years, out-of-year checks; power-mean law exact against the optimiser and a generic solver.

**Workshop (or short e-Energy) paper: "Moved, Not Cut: Who Bears the Cost of Carbon-Aware Computing?"** (F28 + F30, with F17, F19 and F20). This is the result most likely to change how people think, and F30 makes it much stronger: location-based savings on most US grids are transfers onto other consumers (83%), and the stricter standard now proposed, hourly matching / 24/7 CFE, is *worse* for shifting (94% transfer, reported savings 5.6x larger). One rule fixes both: the two-part rule with portfolios credited at the margin. It is timely because the GHG Protocol's October 2026 status update still has hourly matching ('Method 1') under consideration. Draft: `scripts/build_workshop_incidence.py` builds a 3-page double-blind PDF (§6 is F30). With F30 the story could also stand as a full e-Energy 2027 paper if the related work on 24/7 CFE (Riepin et al. 2025; Xu et al. 2024) is read in full and a non-US check is added.

The other drafts are kept for reuse: `build_workshop.py` (signals, F17/F19/F20; superseded by the incidence draft), `build_workshop_battery.py` (F21-F23; the seed of the full paper) and `build_workshop_cooling.py` (F26-F27; a BuildSys/e-Energy candidate).

## 3. Still needed before a full submission

- [~] **Template.** Started 2026-11-22: `paper/sigmetrics/main.tex` (acmart acmsmall, POMACS, anonymous) has the title, abstract, model, formal statements and section skeleton for the battery paper; it has not been compiled (no LaTeX here). Remaining: port the text and figures, add refs.bib. Port to LaTeX `acmart` (sigconf for e-Energy/HotCarbon; the SIGMETRICS/POMACS format for SIGMETRICS). No LaTeX is installed in the build machine; the reportlab drafts are for content only.
- [~] **Proofs.** Drafted 2026-11-22 in `paper/sigmetrics/main.tex` (appendix): homogeneity, sinusoidal water-filling with the 1 + 24φ peak, virtual-battery equivalence with its boundary term, and the power-mean law with its second-order expansion and α -> 1 limit. Still missing: the water-level shadow price and a tight bound on the boundary term. Originally: the homogeneity theorem, the virtual-battery equivalence (and its boundary term), the water-level shadow price, and (F29) the power-mean law with its second-order expansion χ CV²/2 and the α -> 1 limit. For F28, the incidence identity and the two-part properties: budget balance, and exact alignment under linear response.
- [ ] **Related work, deeper.** Speed scaling with time-varying costs (Yao, Demers & Shenker 1995; Antoniadis et al. 2017) and carbon-aware DVFS / GPU power limits (Hanafy et al., HotCarbon 2023; arXiv:2303.02508; FREESH), for F29. Virtual storage for flexible loads (Callaway & Hiskens 2011; Hao et al. 2015), valley filling for EV charging (Gan, Topcu & Low 2013), water-filling in power allocation, 24/7 CFE and hourly matching (Riepin, Brown & Zavala 2025; Xu, Ricks, Manocha, Patankar & Jenkins, Joule 2024; WattTime on additionality; GHG Protocol Scope 2 status update, Oct 2026), carbon accounting (Gorka, Rhodes & Roald 2024; WattTime 2022; GHG Protocol consultation 2025-26; Aumann-Shapley allocation). Read the full texts: they could not be fetched in this session, so the novelty claims are scoped to the abstracts and summaries.
- [ ] **Threats to validity.** One section collecting the following:
  - average vs marginal signals;
  - season × hour MEFs, 2022, US only;
  - fluid work and perfect forecasts;
  - modelled cooling plants;
  - simulated meters (F25);
  - the constant gas margin (F24, F28 CAISO check);
  - an assumed power exponent α and dynamic-power-only model for speed scaling (F29); a measured DVFS curve would remove this;
  - clear-sky solar portfolios at approximate BA centroids and the grid average as residual mix (F30); real plant profiles (EIA-930 solar by BA, not in the committed derived data) would remove the first.
- [ ] **Figures.** Vector PDF figures with one consistent style, sized for two columns. The draft PNGs are at 200 dpi.
- [ ] **Anonymisation.** Double-blind venues need the drafts stripped of author names and the repository link. The new incidence draft is already anonymous; the other three are not.
- [ ] **Artifact.** A tagged release with `reproduce.py` + the experiment scripts and the test suite (132 tests). Consider a Zenodo DOI at camera-ready.
- [ ] **External data gap.** F28 and F17-F20 need marginal factors outside the US. Any free source would strengthen the claim that the result generalises.

## 4. Single next concrete step

**Compile `paper/sigmetrics/main.tex` on a machine with TeX Live (`latexmk -pdf`), then port §1, §5 and the figures from `build_workshop_battery.py` into it and generate `refs.bib` from the REFS lists, so a complete anonymous POMACS draft exists before the SIGMETRICS winter abstract date (Jan 4, 2027 per trackers; paper Jan 11, 2027, AoE).** In parallel, the *Moved, Not Cut* draft (now with F30) is the e-Energy 2027 / HotCarbon 2027 candidate; check energy.acm.org in December for the e-Energy call. Submitting anywhere is the author's decision.
