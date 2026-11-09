"""Measure Apple-Silicon GPU power, throughput and energy/token of LLM inference vs batch size.

For each model and batch size B, B different prompts are decoded together with
MLX (`mlx_lm.batch_generate`, concurrency forced to B). GPU, SoC and whole-system
power are sampled by `macmon` every 100 ms. Each point repeats the batch until
at least --min-seconds of generation have run, and the whole thing is repeated
--repeats times. Idle power is measured first so dynamic energy can be reported.

Outputs: results/llm_power.csv (one row per repeat), results/llm_power_summary.csv,
results/llm_power_meta.json, figures/llm_power_*.png

Run: python scripts/llm_power_batch.py            (~15-25 min on an M4 Pro)
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd
from mlx_lm import load
from mlx_lm.generate import batch_generate

from carbonshift import plotting as P
from carbonshift.power import PowerSampler

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ("mlx-community/Qwen2.5-1.5B-Instruct-4bit", "mlx-community/Qwen2.5-7B-Instruct-4bit")
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
TOPICS = ["the history of the electric grid", "how transformers work", "photosynthesis", "the Roman Empire",
          "black holes", "data centre cooling", "the water cycle", "jazz music", "volcanoes", "the printing press",
          "wind turbines", "the human immune system", "chess strategy", "coral reefs", "the internet",
          "lithium-ion batteries", "ancient Egypt", "climate change", "quantum computing", "honey bees",
          "the Silk Road", "nuclear fusion", "the stock market", "glaciers", "the French Revolution",
          "solar panels", "octopus intelligence", "the Apollo program", "rainforests", "semiconductors",
          "the Olympic Games", "tides"]


def prompts_for(tokenizer, n: int) -> list[list[int]]:
    out = []
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        msg = [{"role": "user", "content": f"Write a long, detailed essay about {topic}. (Variant {i})"}]
        out.append(tokenizer.apply_chat_template(msg, add_generation_prompt=True))
    return out


def run_point(model, tok, sampler: PowerSampler, batch: int, max_tokens: int, min_seconds: float) -> dict:
    prompts = prompts_for(tok, batch)
    kwargs = dict(max_tokens=max_tokens, completion_batch_size=batch, prefill_batch_size=min(batch, 32))
    sampler.wait_for_next(2)
    t0 = time.monotonic()
    gen_tokens = prompt_tokens = 0
    decode_time = 0.0
    while True:
        r = batch_generate(model, tok, prompts, **kwargs)
        gen_tokens += r.stats.generation_tokens
        prompt_tokens += r.stats.prompt_tokens
        decode_time += r.stats.generation_time
        if time.monotonic() - t0 >= min_seconds:
            break
    mx.synchronize()
    t1 = time.monotonic()
    p = sampler.mean(t0, t1)
    wall = t1 - t0
    return {"batch": batch, "wall_s": wall, "gen_tokens": gen_tokens, "prompt_tokens": prompt_tokens,
            "tok_per_s": gen_tokens / wall, "decode_tok_per_s": gen_tokens / decode_time,
            **{f"{k}_w" if k.endswith("power") else k: v for k, v in p.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--batches", nargs="+", type=int, default=list(BATCHES))
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--min-seconds", type=float, default=6.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--summarise-only", action="store_true", help="rebuild summary/figure from results/llm_power.csv")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.summarise_only:
        df = pd.read_csv(RESULTS / "llm_power.csv")
        df["all_j_per_tok"] = df.all_power_w / df.tok_per_s
        summary = summarise(df)
        summary.to_csv(RESULTS / "llm_power_summary.csv", index=False)
        print(summary.round(3).to_string(index=False))
        figures(summary)
        return

    rows = []
    with PowerSampler(interval_ms=100) as s:
        time.sleep(1)
        t0 = time.monotonic(); time.sleep(10); idle = s.mean(t0, time.monotonic())
        print(f"idle: GPU {idle['gpu_power']:.2f} W  SoC {idle['all_power']:.2f} W  system {idle['sys_power']:.2f} W")
        for name in args.models:
            model, tok = load(name)
            batch_generate(model, tok, prompts_for(tok, 4), max_tokens=32)  # warm-up / kernel compile
            for rep in range(args.repeats):
                for b in args.batches:
                    r = run_point(model, tok, s, b, args.max_tokens, args.min_seconds)
                    r.update(model=name.split("/")[-1], repeat=rep)
                    rows.append(r)
                    print(f"{r['model']:28s} B={b:4d} rep={rep} {r['tok_per_s']:8.1f} tok/s  "
                          f"GPU {r['gpu_power_w']:5.1f} W  sys {r['sys_power_w']:5.1f} W  "
                          f"{r['gpu_power_w'] / r['tok_per_s'] * 1000:7.2f} mJ/tok(GPU)", flush=True)
            del model
            mx.clear_cache()

    df = pd.DataFrame(rows)
    for k in ("gpu", "all", "sys"):
        df[f"{k}_j_per_tok"] = df[f"{k}_power_w"] / df.tok_per_s
    df["sys_dynamic_j_per_tok"] = (df.sys_power_w - idle["sys_power"]) / df.tok_per_s
    df.to_csv(RESULTS / "llm_power.csv", index=False)
    summary = summarise(df)
    summary.to_csv(RESULTS / "llm_power_summary.csv", index=False)
    meta = {"chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
            "macos": platform.mac_ver()[0], "mlx": mx.__version__, "idle": idle, "args": vars(args),
            "date": time.strftime("%Y-%m-%d")}
    (RESULTS / "llm_power_meta.json").write_text(json.dumps(meta, indent=2))
    print(summary.round(3).to_string(index=False))
    figures(summary)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["tok_per_s", "gpu_power_w", "all_power_w", "sys_power_w", "gpu_j_per_tok", "all_j_per_tok", "sys_j_per_tok",
            "sys_dynamic_j_per_tok", "gpu_freq_mhz", "gpu_active_ratio"]
    g = df.groupby(["model", "batch"])[cols]
    out = g.mean().join(g.std().add_suffix("_std")).reset_index()
    return out


def figures(s: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    P.setup()
    models = list(dict.fromkeys(s.model))
    colors = (P.BLUE, P.ORANGE, P.AQUA)
    panels = (("tok_per_s", "Throughput (tokens/s)", True),
              ("gpu_power_w", "GPU power (W)", False),
              ("all_j_per_tok", "Chip energy per token\n(CPU + GPU + DRAM)", True))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (col, label, log) in zip(axes, panels):
        for k, m in enumerate(models):
            d = s[s.model == m]
            ax.errorbar(d.batch, d[col], yerr=d[f"{col}_std"], fmt="o-", ms=4, capsize=2, color=colors[k], label=m)
        ax.set_xscale("log", base=2)
        if log:
            ax.set_yscale("log")
            ticks = {"tok_per_s": [50, 100, 200, 500, 1000, 2000], "all_j_per_tok": [0.02, 0.05, 0.1, 0.2, 0.4]}[col]
            fmt = (lambda v: f"{v:g}") if col == "tok_per_s" else (lambda v: f"{v * 1000:g} mJ")
            ax.set_yticks(ticks, [fmt(t) for t in ticks])
            ax.minorticks_off()
        ax.set(xlabel="Batch size (concurrent sequences)", ylabel=label)
        ax.set_xticks(sorted(s.batch.unique()), [str(b) for b in sorted(s.batch.unique())])
        ax.minorticks_off()
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    P.save(fig, "llm_power_batch")


if __name__ == "__main__":
    main()
