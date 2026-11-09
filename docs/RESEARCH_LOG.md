# Research log

A running record of every finding, with its evidence, and a ranked backlog of open questions.
Each loop iteration takes the top backlog item, checks it is new, implements it with tests, runs it, verifies the numbers, and records the result here.

## Findings

| ID | Finding | Evidence | Status |
|---|---|---|---|
| F1 | 27 of 30 numeric claims of Sukprasert et al. (EuroSys'24) reproduce (within ~1 g / 0.6 pp / 3%, or a stated bound is met) from the released data | `results/repro_vs_paper.csv`; `tests/test_temporal.py` (equivalence to the original loop) | verified |
| F2 | Three discrepancies: Fig 10d vs 10a (127 vs 135 g); CV<0.1 share is 49.6%, not >70%, and the max/min ratio is 60x, not 40x; the spatial metric treats departed work as zero-carbon (Asia at 50% idle: 556 -> 387 g) | `results/repro_spatial.csv`, REPORT §2 | verified |
| F3 | The uncapped temporal optimum needs a median peak draw of 13x mean load (24 h slack) and 85x (168 h) | `results/facility_capped_per_zone.csv` | verified |
| F4 | Realistic headroom keeps part of the ideal: +20% keeps 36% and +50% keeps 58% (phi=0.5, 24 h). The median zone needs +34% for half and +125% for 80% | `results/facility_summary.csv`, `facility_headroom_needed.csv` | verified |
| F5 | A 10-min UPS battery cycled daily cuts emissions 0.19% (best zone 0.9%); a 4 h battery 3.5% | `results/facility_battery_summary.csv` | verified |
| F6 | Batching 1 -> 32 on an M4 Pro GPU: 6.5x throughput for 1.5x power, i.e. 4.4x less energy per token (1.5B); saturation at batch ~32 (~31 W) | `results/llm_power*.csv` | measured |
| F7 | Shadow carbon price of capacity: one extra kW of headroom saves 228 kg CO2/yr (median zone, +20% headroom, 24 h) | `results/redundancy_marginal_per_zone.csv`; dual variables validated by finite differences | verified |
| F8 | Build power, not servers: electrical headroom pays back in <1 yr in 94-98% of zones; CPU servers bought for shifting pay back within life in only 23% | `results/redundancy_payback_summary.csv`; EPD sources in `embodied.py` | verified (server factors rest on assumed draws) |
| F9 | Redundancy dividend: N+1 / 2N spare closes 36-73% of the gap to the ideal; outages cost <0.01 pp; carbon-aware maintenance is negligible (negative result) | `results/redundancy_dividend_summary.csv` | verified |
| F10 | Facility findings hold for 2020, 2021 and 2022: +20% headroom keeps 36-37% of the ideal; N+1 closes 35-36% and 2N 72-73% of the gap. Per-zone shadow prices rank-correlate at Spearman 0.93-0.97 between years. The median shadow price **rose 17%** (198 -> 232 kg/kW-yr), so capacity became more valuable for carbon shifting | `results/robustness_years_*.csv`, `scripts/robustness_years.py` | verified (3 years only; trend is suggestive) |

## Backlog (ranked by value ÷ cost)

1. **Forecast error under a power cap.** With a cap, mistakes are harder to undo. Does a CarbonCast-level error (MAPE 5-15%) hurt capped shifting more than uncapped shifting?
2. **Diurnal IT load.** Real inflexible load peaks in the daytime, which uses up headroom exactly when solar makes the grid clean. Model a diurnal base load and measure the extra loss.
3. **Where should the next MW go? Siting headroom across a cloud's regions.** Compare the shadow price of capacity per region, combining temporal value with spatial-destination value under latency limits.
4. **Power-capped LLM serving.** Use the measured batch-energy curves with a power cap and carbon signal: batch big in clean hours, small in dirty hours, under a latency SLO.
5. **Grid decarbonisation trajectory.** How do shadow prices change as zones add solar and wind? (Paper [1] shows relative benefits shrink; capacity value may behave differently.)

## Iteration notes

- **Iteration 1 (2026-11-09):** multi-year robustness check (F10). No novelty search was needed, since this is a validity check. One definition was corrected (share kept = ratio of mean absolute savings, matching `facility_summary.csv`).
