# LaTeX paper drafts

- `sigmetrics/main.tex`: `acmart` (acmsmall, POMACS, anonymous review) skeleton of the full paper
  *Your Data Centre Is Already a Battery* (F21-F24, F29), with formal statements and a proof
  appendix (homogeneity, sinusoidal water-filling, virtual-battery equivalence, power-mean law).
  Sections marked TODO are to be ported from `scripts/build_workshop_battery.py`.

Build with `latexmk -pdf main.tex` on a machine with TeX Live (`acmart` is part of it). The research
machine has no LaTeX, so the skeleton has not been compiled yet; check it compiles before relying on it.
