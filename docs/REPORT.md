# Carbon-Aware Workload Shifting Meets the Facility's Power Limit

**A reproduction of Sukprasert, Souza, Bashir, Irwin & Shenoy, "On the Limitations of Carbon-Aware Temporal and Spatial Workload Shifting in the Cloud" (EuroSys 2024), extended with facility power-capacity and UPS-battery limits and with GPU power measured against LLM batch size.**

Naveen Lakkaram · November 2026 · code and data: this repository (`scripts/reproduce.py`, `scripts/facility_experiments.py`)

---

## 1. Motivation

Sukprasert et al. bound the carbon saved by moving cloud work in **time** (deferring or pausing jobs into low-carbon hours) and **space** (migrating work to greener regions). They use hourly Electricity Maps traces for 123 grid zones. Their central message is that the practical savings are far below the ideal. The spatial analysis already applies capacity limits, but the temporal analysis treats each job alone and so lets every job fall into the same low-carbon hours. The authors name this gap themselves (§5.2.5, "resource constraints that prevent running many jobs during low carbon periods").

A data centre is an electrical facility before it is a computer. Its grid draw is bounded by the utility service, the substation transformers, switchgear, the UPS rating with N+1 or 2N redundancy, and the generator plant sized to back it up. This project asks:

> **How much of the ideal temporal carbon saving survives once all shifted work must fit under the facility's power rating? And can the UPS batteries a facility already owns shift carbon on their own?**

## 2. Reproduction

I re-implemented the analysis from scratch: vectorised NumPy, about 20 s for every figure on a laptop instead of a 50-process loop. I ran it on the authors' released 2022 traces. A unit test checks my temporal code against a line-by-line port of the artifact's loop, so edge handling matches exactly: windows are clipped at the end of the year, and "one-year slack" means "until the end of the year".

**Result: 27 of 30 numeric claims reproduce.** Each lands within about 1 g·CO₂eq, 0.6 percentage points, or 3% of the stated value. The single "< 10" bound is also met (`results/repro_vs_paper.csv`):

| Claim | Paper | Reproduced |
|---|---:|---:|
| Global mean carbon intensity (g/kWh) | 368.39 | 368.38 |
| Deferral, 1 h → 168 h job, 1-year slack (g/job-h) | 154 → 70 | 153.9 → 69.9 |
| Deferral, 1 h → 168 h job, 24 h slack | 57 → 3 | 57.4 → 2.9 |
| Extra from interruption, 168 h job, 1-yr slack | 43 | 42.5 |
| Interruption peak (24 h job, 24 h slack) | 18.4 | 18.4 |
| Equal / Azure / Google job mix, 1-yr slack | 135 / 100 / 112 | 134.9 / 100.6 / 111.9 |
| Spatial, infinite capacity (Global / Europe / Asia) | 352 / 281 / 556 | 352.5 / 281.2 / 555.8 |
| Spatial, 50% idle capacity (Global) | 190 | 189.7 |
| Global reduction at 50% / 99% idle | 51.5% / 95.68% | 51.50% / 95.68% |
| Latency SLO 250 ms, infinite / 50% util. | 92.5% / 45.7% | 92.5% / 45.5% |
| Latency SLO 50 ms, 50% utilisation | 31% | 31.1% |
| 1-migration vs ∞-migration gap | < 10 g | ≤ 7.9 g |

![Temporal reproduction](../figures/repro_temporal.png)

**Three observations from the reproduction:**

1. **Fig. 10d, 1-year slack (127 g)** is the same quantity as Fig. 10a's equal-mix value (135 g). The artifact code computes both identically and gives 134.9, so the 127 likely comes from an earlier run.
2. **Two §1/§4 statistics don't match the stated definitions.** The share of zones with daily CV < 0.1 is 49.6%, not "> 70%"; the threshold CV < 0.15 gives 74%. The max/min mean-intensity ratio is 60× across all zones and 51× across cloud zones, not 40×. Neither changes any conclusion.
3. **Spatial metric for groups whose work all leaves the group.** The artifact computes a group's reduction as its mean intensity minus the mean of member DCs that receive work. When no member receives work, that term becomes 0 (`NaN → 0`), as if the work landed somewhere zero-carbon. For Asia at 50% idle this gives the paper's 556 g. If each origin is instead charged the intensity where its work actually lands (dirtiest origin → greenest spare capacity), Asia gets **387 g** and the Americas at infinite capacity get 364 g instead of 380 g. The global numbers are unaffected, as are the paper's qualitative claims. Both metrics are in `results/repro_spatial.csv`.

## 3. Extension: facility power limits

**Model.** Power is normalised to the facility's mean IT load (= 1):
- Each hour a share φ of the energy is flexible, pre-emptible work that must finish within `slack` hours; the remaining 1 − φ is inflexible.
- Total grid draw in every hour must stay at or below a cap *C*. *C* − 1 is the **power headroom** the electrical plant has above average load.

Minimising Σₜ CIₜ · drawₜ is a transportation LP with one variable per (arrival, hour) pair. Because every arrival has the same slack, earliest-deadline-first equals FIFO. A served profile is therefore feasible exactly when its cumulative sum stays between "arrivals delayed by the slack" and "arrivals so far". That turns the problem into an LP over 8,760 cumulative variables with banded constraints, solved by HiGHS in about 50 ms. Unit tests show:
- it equals the brute-force per-arrival LP on random instances;
- with *C* = ∞ it reproduces the paper's 1-hour-job number (57.4 g per flexible kWh at 24 h slack), so the paper's result is the *C* → ∞ limit of this model.

The sweep covers all 123 zones × φ ∈ {25, 50, 75%} × slack ∈ {24 h, 168 h} × 14 caps (10,332 LPs, 78 s on 14 cores).

*Relation to prior work.* Google's Carbon-Intelligent Compute System already applies hourly capacity caps ("virtual capacity curves") to flexible work in production (Radovanović et al., 2021), and CarbonFlex studies cluster contention for low-carbon hours. This section complements them with a cross-grid map: how much of the ideal saving each level of headroom keeps, in each of 123 grid zones.

**Finding 1: the ideal saving needs an absurd power spike.** With 50% flexible load and 24 h slack, the uncapped optimum crams a day's flexible work into the single cleanest hour. The median facility's peak draw is **13× its mean load**, and **85×** with 168 h slack. No facility is built that way.

**Finding 2: realistic headroom keeps only part of the saving.** These are global means at φ = 0.5:

| Headroom above mean load | 10% | 20% | 50% | 100% | ∞ |
|---|---:|---:|---:|---:|---:|
| Share of ideal kept, 24 h slack | 23% | 36% | 58% | 76% | 100% |
| Facility CO₂ reduction, 24 h slack | 2.3% | 3.5% | 5.4% | 6.8% | 8.8% |
| Share of ideal kept, 168 h slack | 17% | 27% | 44% | 59% | 100% |

The median zone needs **+34% headroom to keep half** its ideal 24 h saving, and **+125% to keep 80%**. With weekly slack, the median zone stays below 80% even at 4× capacity: more slack only helps if there is spare power to use it.

![Share of the ideal saving retained](../figures/facility_retained.png)

**Forecast error under a cap.** Real schedulers act on forecasts. I optimised schedules on noisy forecasts and charged them the true intensity. The forecasts used autocorrelated (AR(1), rho = 0.9) multiplicative errors calibrated to a target MAPE (`src/carbonshift/forecast.py`; 123 zones, 3 seeds).

| Forecast MAPE | 5% | 10% | 15% | 30% |
|---|---:|---:|---:|---:|
| Saving lost, uncapped (% of saving) | 9% | 23% | 36% | 66% |
| Saving lost, uncapped (% of emissions) | 0.8% | 1.9% | 2.9% | 5.3% |
| Saving lost, +20% headroom (% of saving) | 10% | 23% | 33% | 50% |
| +20% headroom, share of uncapped saving kept | 35% | 36% | 38% | 52% |

**Finding 2b: at realistic forecast quality, the power cap and forecast error compound almost independently.** The capped facility keeps about the same share of the uncapped saving (35-38%) as with perfect forecasts (36%). Only with poor forecasts does the cap act as a hedge: capped schedules spread work over many hours instead of betting on the predicted minimum. The absolute cost of forecast error (0.8-2.9% of emissions at 5-15% MAPE) agrees with the paper's estimate of about 3%.

![Forecast error vs cap](../figures/forecast_regret.png)

**Finding 3: green, variable grids gain the most from headroom.** South Australia gets 6% from just 5% headroom. India-West has a flat, coal-heavy trace and stays near 5% even with unlimited power.

![Selected zones](../figures/facility_zones.png)

**Finding 4: UPS batteries can't do the job.** A UPS sized for 10 minutes of ride-through, fully cycled once a day with 90% round-trip efficiency, cuts emissions by **0.19% on average** (0.9% in the best zone). Keeping half of it in reserve for ride-through halves that. Even a dedicated 4-hour battery averages 3.5%, less than half-flexible load with 50% headroom (5.4%).

![Battery shifting](../figures/facility_battery.png)

## 4. Extension 2: GPU power vs LLM batch size, measured

AI inference is the fastest-growing flexible load, and batch size is its main efficiency knob. I measured it on real hardware: an Apple M4 Pro (20-core GPU) running MLX 0.32 with two 4-bit Qwen2.5 instruct models (1.5B and 7B).
- **Workload:** B different prompts decoded together, with concurrency forced to B; 128 new tokens each, at least 6 s per point, 3 repeats.
- **Power:** sampled every 100 ms with `macmon`, which reads the same IOReport counters as `powermetrics` but needs no sudo.
- **Headline metric:** chip energy (CPU + GPU + DRAM). Whole-system power drifted upward during the run as the machine warmed, so it is reported as an upper bound only. Raw data: `results/llm_power.csv`.

| Model | Batch | Tokens/s | GPU W | Chip mJ/token |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B | 1 | 198 | 15.3 | 99 |
| | 8 | 604 | 25.4 | 45 |
| | 32 | 1,292 | 28.5 | **23** |
| | 128 | 1,392 | 30.7 | 23 |
| Qwen2.5-7B | 1 | 53 | 18.2 | 377 |
| | 8 | 149 | 28.4 | 194 |
| | 32 | 318 | 31.2 | **100** |
| | 128 | 326 | 32.1 | 101 |

Throughput repeats agree within about 1%; chip power varies by up to 8% between repeats (7B model, batch 1). Error bars are in the figure.

![LLM power vs batch](../figures/llm_power_batch.png)

**Finding 5: batching buys throughput far more cheaply than it costs power.** For the 1.5B model, going from batch 1 to batch 32 raises throughput **6.5×** while chip power rises only **1.5×**. Energy per token falls **4.3×**, and **3.8×** for the 7B model.

**Finding 6: the GPU saturates at batch ≈ 32.** Past it, decode throughput rises under 10% (1,616 → 1,770 tokens/s for 1.5B, decode only) and GPU power levels off at 31–32 W. Bigger batches then add queueing delay without adding efficiency. The steps are uneven: 4 → 8 is nearly flat, then 16 → 32 jumps 1.7–1.9×. That pattern suggests a kernel-selection threshold in MLX's quantised matrix multiplication; I have not verified this.

**Link to Extension 1.** On a power-capped facility, the first lever is to run inference at its efficient batch size, which delivers about 4× more tokens per joule within the same watts. Only work beyond the saturation point should be shifted in time, and that shifting is what draws on headroom. In a California-like grid (264 g/kWh mean), the 1.5B model emits about 7.3 g CO₂ per million tokens at batch 1 and 1.7 g at batch 32. That 4× gap exceeds anything temporal shifting achieved in Section 3.

## 5. Extension 3: the carbon price of a kilowatt, and the redundancy dividend

Section 3 showed that headroom limits carbon-aware shifting. Two questions follow: **what is one more kW of headroom worth, and is it worth building?**

**Shadow carbon price of capacity.** The dual variables of the capped LP's hourly power constraints give, for each hour, the grams of CO₂ saved by one more kW of capacity in that hour. They are verified against finite differences in `tests/test_facility.py`. Summed over a year, they give the **annual carbon value of one extra kW of headroom**. With half the load flexible and 24 h slack, the median zone saves:
- **346 kg/yr** at 10% existing headroom;
- **228 kg/yr** at 20%;
- **112 kg/yr** at 50%;
- **48 kg/yr** at 100%.

The 90th-percentile zone saves 2.5–3× more. The value falls steeply as headroom grows: the first kilowatts are worth most.

**What headroom costs to build.** I found no published per-kW embodied-carbon factors for data-centre electrical plant, so I derived them from manufacturer EPDs (A1–A3, `src/carbonshift/embodied.py`):
- **Transformers:** 16.3 kg/kVA (Trafo Elettro 2,500 kVA cast resin) and 7.65 kg/kVA (ABB 16/20 MVA).
- **UPS:** 17–74 kg/kW across three products (ABB MegaFlex, ABB DPA 250, Schneider Galaxy VX).
- **Generator sets:** 53 kg/kVA (Cummins QSK95) and 61 kg/kVA (mtu 10V1600).
- **Servers** (product carbon footprints): Dell R650 and R760 for CPU servers, and the NVIDIA HGX H100 baseboard for GPU servers.

Sheddable, deferrable work doesn't need UPS or generator backup, because on a grid outage it is postponed, not lost. So the cheapest headroom for carbon shifting is **grid-only** capacity (transformer + switchgear, about 28 kg/kW). Fully backed capacity costs about 150 kg/kW.

| Headroom build (median zone, 20% existing headroom, 24 h slack) | Embodied kg/kW | Payback | Zones that pay back within service life* |
|---|---:|---:|---:|
| Grid-only electrical (25-yr life) | 28 | **0.1 yr** | 98% |
| Fully backed electrical (25-yr life) | 150 | 0.7 yr | 94% |
| GPU servers (5-yr life) | ~470 | 2.1 yr | 79% |
| CPU servers (5-yr life) | ~2,000 | 8.8 yr | **23%** |

\*With savings shrinking 3%/yr as grids decarbonise.

**Finding 7: build power, not servers.** Extra electrical capacity for carbon shifting repays its embodied carbon within months in almost every zone. Buying CPU servers just to absorb shifted work fails to pay back within a server's life in more than three-quarters of zones. Carbon-aware shifting is therefore worth doing with **idle compute that already exists**, and a power-constrained operator gains most from electrical headroom, not from more servers. The server factors rest on an assumed average draw (0.35 kW per CPU server; a full GPU server at twice the baseboard), and are flagged as assumptions in the code.

![Shadow price of capacity](../figures/redundancy_marginal.png)

**The redundancy dividend.** Tier-rated facilities already own idle electrical capacity: the spare module in N+1, or the whole second path in 2N. Microsoft's Flex (ISCA'21) showed this reserve can host sheddable work and be reclaimed within seconds when a failure occurs. I apply that idea to carbon: put deferrable work on the redundant capacity, and fall back to design capacity during maintenance (2 × 24 h per year) and random failures (0.5 per year, 8 h mean repair).

Results at 65% design utilisation, mean over 123 zones, with half the load flexible and 24 h slack:

| Capacity available to flexible work | Facility CO₂ reduction | Gap to ideal (8.8%) closed |
|---|---:|---:|
| Design capacity only | 5.5% | – |
| + N+1 spare (N = 4) | 6.7% | 36% |
| + 2N spare | 7.9% | 73% |

**Finding 8: redundancy is free carbon headroom, and outages barely matter.** Maintenance and failures cost less than 0.01 percentage points: the dividend survives outages almost intact, and it adds no embodied carbon. Scheduling maintenance in the hours with the lowest shadow price recovers 90–100% of that tiny loss. The effect is real but negligible, an honest negative result for carbon-aware maintenance scheduling.

![Redundancy dividend](../figures/redundancy_dividend.png)

**Robustness across years.** The facility findings are not an artefact of 2022 (`scripts/robustness_years.py`; phi = 0.5, 24 h slack, the 120 zones with non-constant traces).

| Metric | 2020 | 2021 | 2022 |
|---|---:|---:|---:|
| Share of ideal kept, +20% headroom | 36.7% | 36.0% | 35.6% |
| Share of ideal kept, +50% headroom | 58.7% | 58.0% | 58.0% |
| Shadow price of capacity, median (kg/kW-yr) | 198 | 217 | 232 |
| CPU servers repaid within life | 15% | 18% | 23% |
| Gap closed by N+1 / 2N | 35% / 72% | 35% / 72% | 36% / 73% |

Per-zone shadow prices rank-correlate at Spearman 0.93-0.97 between years, so a zone that values capacity highly in one year does so in the next. **Finding 9: the carbon value of capacity is rising.** The median shadow price grew 17% from 2020 to 2022. Three years is too short to call it a trend, but it runs opposite to the paper's observation that the *relative* benefit of carbon-aware scheduling shrinks as grids get greener. As variability grows, headroom becomes worth more.

**Related work and novelty.** Google's Carbon-Intelligent Compute System caps flexible capacity hour by hour with virtual capacity curves (Radovanović et al., 2021), and the UMass group studies contention when many jobs pick the same low-carbon window (CarbonFlex 2025; Hanafy et al. 2026). Flex (Zhang et al., ISCA 2021) reclaims reserved power to deploy more servers. I have not found prior work that does any of the following:
1. derives the shadow carbon price of facility capacity across many grids;
2. compares it with EPD-based embodied carbon of electrical plant and servers;
3. uses tier redundancy as carbon-shifting headroom.

## 6. Takeaways

1. The EuroSys'24 temporal upper bounds are also **power** upper bounds: reaching them assumes a facility can draw many times its average load. A useful practical bound is the **saving-vs-headroom curve**, which depends on the electrical design.
2. Headroom is the real currency. Stranded power capacity is common because facilities are provisioned for peak and redundancy. Carbon-aware schedulers should treat the *remaining electrical headroom* as the scarce resource. The marginal gram saved per extra kW of headroom falls steeply after about 50%.
3. The headroom curve sets operational savings against **embodied carbon and cost** of extra electrical plant. That trade-off connects to Ecovisor's idea of exposing power and energy to applications, and to CarbonScaler's elasticity.

## 7. Limitations

- Grid **average** carbon intensity is used, as in the paper. Marginal intensity can change conclusions.
- Forecasts are perfect. The paper estimates about a 3% loss at CarbonCast-level errors.
- Inflexible load is flat. Real IT load has a diurnal shape that uses up some headroom at the very hours where solar makes the grid cleanest.
- Flexible work is modelled as fluid and pre-emptible; long, non-parallel jobs would retain less. Migration and checkpoint overheads are ignored.
- Batteries are charged at grid average intensity, and degradation and the embodied carbon of extra headroom are not counted.
- The embodied factors come from a small number of EPDs: one or two per component, and switchgear has none. Server factors rest on assumed average power draws. The redundancy model treats failures as fleet-average events and ignores partial-module failures and cooling redundancy.
- The GPU numbers come from one laptop-class Apple chip running two small 4-bit models. They show the shape of the batch curve, not data-centre accelerator numbers. Background apps were running during the measurement, which is why whole-system power is reported only as an upper bound.

## 8. Next steps

1. Use a real facility load shape and per-cluster power caps, such as the Google power-trace data, in place of the flat base load.
2. Add a joint spatial-temporal model in which each region has its own power cap, so that low-carbon regions saturate.
3. Make the trade-off explicit: the embodied carbon of extra transformer, UPS and generator capacity against the operational carbon saved, giving a carbon-optimal headroom.
4. Model facility-aware control in an Ecovisor-style virtual energy system that exposes "power budget" alongside "carbon signal" to applications.
5. Repeat the batch-size study on a data-centre GPU (an A100 or H100 with `nvidia-smi` power capping), adding a power-cap sweep. The question is whether "batch first, then shift" holds when the chip itself is capped.

## 9. Interactive explorer

`carbonshift-web` serves a small app for experimenting with every model in this report: pick a grid region and slide the job length, slack, flexible share, power headroom, battery size or LLM batch, and the results update live. See the README for its security design.

## Reproducing

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest              # 44 tests
.venv/bin/python scripts/reproduce.py   # ~20 s; results/repro_*.csv, figures/repro_*.png
.venv/bin/python scripts/facility_experiments.py  # ~80 s; results/facility_*.csv, figures/facility_*.png
.venv/bin/python scripts/llm_power_batch.py       # ~20 min, Apple Silicon + `brew install macmon`
.venv/bin/python scripts/redundancy_experiments.py # ~2 min; shadow prices, embodied payback, redundancy dividend
```
