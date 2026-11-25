# LaTeX paper drafts

- `sigmetrics/main.tex`: `acmart` (acmsmall, POMACS, anonymous review) skeleton of the full paper
  *Your Data Centre Is Already a Battery* (F21-F24, F29), with formal statements and a proof
  appendix (homogeneity, sinusoidal water-filling, virtual-battery equivalence, power-mean law).
  Sections marked TODO are to be ported from `scripts/build_workshop_battery.py`.

Build with `latexmk -pdf main.tex` on a machine with TeX Live (`acmart` is part of it). It compiles
cleanly as of 2026-11-25 (7 pages, no undefined references); the compiled draft is `sigmetrics/main.pdf`.
Regenerate the bibliography with `python scripts/build_bib.py`.
