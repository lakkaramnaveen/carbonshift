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
| F11 | Forecast error and power caps compound almost independently at realistic error. At 5-15% MAPE (CarbonCast range), +20% headroom keeps 35-38% of the equally mis-forecast uncapped saving (36% with perfect forecasts). Only at 30% MAPE does the cap hedge (keeps 52%), because capped schedules spread work over many hours. Uncapped forecast error costs 0.8-2.9% of emissions, consistent with [1]. Shifting almost never backfires | `results/forecast_*.csv`, `src/carbonshift/forecast.py`, `tests/test_forecast.py` | verified (synthetic AR(1) errors, rho = 0.9) |
| F12 | Real Google power domains (PowerData2019, 57 domains) run at a median 70% of power capacity (52-78%), i.e. +42% headroom. At their real headroom, carbon shifting keeps a median 54% of the ideal saving (44-75%). The daily load shape barely matters: the worst alignment of the fleet-median shape with the grid loses 1.4% of the saving, and the most diurnal domain (21% swing) loses 5.9%; averaged over alignments it equals the flat model | `results/diurnal_*.csv`, `src/carbonshift/google_power.py`, `tests/test_facility.py::test_varying_load_equals_brute_force` | verified (one month, 2019; utilisation includes cooling) |

## Backlog (ranked by value ÷ cost)

1. **Where should the next MW go? Siting headroom across a cloud's regions.** Compare the shadow price of capacity per region, combining temporal value with spatial-destination value under latency limits.
2. **Power-capped LLM serving.** Use the measured batch-energy curves with a power cap and carbon signal: batch big in clean hours, small in dirty hours, under a latency SLO.
3. **Grid decarbonisation trajectory.** How do shadow prices change as zones add solar and wind? (Paper [1] shows relative benefits shrink; capacity value may behave differently.)

## Iteration notes

- **Iteration 1 (2026-11-09):** multi-year robustness check (F10). No novelty search was needed, since this is a validity check. One definition was corrected (share kept = ratio of mean absolute savings, matching `facility_summary.csv`).
- **Iteration 2 (2026-11-09):** forecast error under a power cap (F11). Novelty check: prior work quantifies forecast-error loss for uncapped shifting (arXiv:2508.14625; FM-CAC arXiv:2604.16448), but I found none for power-capped facilities. The headline metric was cross-checked against [1]'s 'about 3% of emissions' at CarbonCast-level error.
- **Iteration 3 (2026-11-09):** real facility load from Google PowerData2019 (F12). Novelty check: carbon-aware work (CICS, VCC) uses internal Google data; I found no public study that combines the released power traces with cross-grid carbon data. The documented meaning of `bad_measurement_data` contradicts the data (24 of 508,896 rows are True), so True rows were treated as bad. The library now supports time-varying load (brute-force tested).
