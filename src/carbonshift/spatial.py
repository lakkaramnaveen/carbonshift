"""Spatial workload shifting upper bounds (Sukprasert et al., EuroSys'24, Sec. 5.1).

Capacity model from the original artifact: N identical data centres, one per
zone, each with ``N`` units of capacity. With idle fraction ``u`` every DC
starts with ``int(N*(1-u))`` units of work. All work is then packed into the
greenest DCs (by annual mean carbon intensity), filling each one completely
before moving to the next.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def greedy_fill(n_regions: int, idle_frac: float) -> np.ndarray:
    """Units of work hosted by each DC (DCs ordered greenest first)."""
    cap = n_regions
    remaining = int(cap * (1 - idle_frac)) * n_regions
    alloc = np.zeros(n_regions)
    for i in range(n_regions):
        if remaining <= 0:
            break
        alloc[i] = min(cap, remaining)
        remaining -= alloc[i]
    return alloc


def hosting_regions(zone_means: pd.Series, idle_frac: float) -> pd.Series:
    """Mean CI of every zone that hosts work after packing (NaN where it hosts none)."""
    means = zone_means.sort_values()
    alloc = greedy_fill(len(means), idle_frac)
    return means.where(alloc > 0)


def global_idle_curve(zone_means: pd.Series, idle_fracs) -> pd.DataFrame:
    """Figure 5(c): global reduction (%) as idle capacity grows.

    ``artifact`` is the original metric (unweighted mean CI of hosting DCs);
    ``work_weighted`` weights each hosting DC by the work it carries.
    """
    means = zone_means.sort_values()
    base = means.mean()
    rows = []
    for u in idle_fracs:
        alloc = greedy_fill(len(means), u)
        host = means[alloc > 0]
        weighted = float((alloc * means).sum() / alloc.sum())
        rows.append({
            "idle_pct": round(u * 100),
            "artifact_pct": (base - host.mean()) / base * 100,
            "work_weighted_pct": (base - weighted) / base * 100,
            "emission_g_per_kwh": weighted,
        })
    return pd.DataFrame(rows).set_index("idle_pct")


def grouping_reduction(zone_means: pd.Series, members: list[str], idle_frac: float) -> dict:
    """Figure 5(a)/(b): mean reduction for origins in a grouping.

    ``artifact``: the original metric, mean(origin CI) - mean(CI of hosting DCs
    *inside the grouping*), counted as 0 when no member hosts work. It treats the
    destination as zero-carbon whenever all of a grouping's work leaves it (e.g. Asia).

    ``corrected``: mean over origins of (origin CI - CI where its work actually
    lands). Work is assigned dirtiest origin -> greenest free capacity, which is
    the matching described in the paper's text (Sec. 5.1.2).
    """
    host = hosting_regions(zone_means, idle_frac)
    m = zone_means.loc[members]
    host_in_group = host.loc[members].mean()
    artifact = m.mean() - (0.0 if np.isnan(host_in_group) else host_in_group)

    landed = landing_ci(zone_means, idle_frac)
    corrected = float((zone_means - landed).loc[members].mean())
    return {"artifact": float(artifact), "corrected": corrected}


def landing_ci(zone_means: pd.Series, idle_frac: float) -> pd.Series:
    """CI where each origin's work runs: dirtiest origins get the greenest free capacity.

    Work is fluid. Each DC keeps capacity 1 and starts at load ``1-idle_frac``.
    Migrated work only moves to a greener DC, so a green DC's own work never
    leaves.
    """
    means = zone_means.sort_values()
    n = len(means)
    load = 1 - idle_frac
    free = np.full(n, idle_frac)  # spare capacity, greenest first
    landed_emis = means.to_numpy() * load  # start: every origin runs at home
    dest = 0
    for o in range(n - 1, -1, -1):  # dirtiest origin first
        work = load
        moved_emis = 0.0
        while work > 1e-12 and dest < o:
            take = min(work, free[dest])
            moved_emis += take * means.iloc[dest]
            work -= take
            free[dest] -= take
            if free[dest] <= 1e-12:
                dest += 1
        if work < load:  # some work moved
            landed_emis[o] = moved_emis + work * means.iloc[o]
    return pd.Series(landed_emis / load, index=means.index).reindex(zone_means.index)


def one_vs_inf_migration(ci: pd.DataFrame, members: list[str]) -> dict:
    """Figure 6(b): migrate once to the greenest member vs follow the hourly minimum."""
    df = ci[members]
    one_dest = df[df.mean().idxmin()]
    inf_trace = df.min(axis=1)
    return {
        "one": float(df.sub(one_dest, axis=0).mean().mean()),
        "inf": float(df.sub(inf_trace, axis=0).mean().mean()),
    }


def capacity_latency(zone_means: pd.Series, latency: pd.DataFrame, latency_limits, idle_fracs):
    """Figure 6(a): global average emissions under latency SLO + capacity (port of artifact).

    ``latency[dest][origin]`` holds the measured GCP inter-region RTT in ms.
    Origins are processed greenest first, each filling its greenest reachable
    destinations; capacity is 100 units per DC.
    """
    zones = [z for z in zone_means.sort_values().index if z in latency.columns]
    lat = latency.loc[latency.index.isin(zones), zones]
    means = zone_means.loc[zones]
    n = len(zones)
    out = pd.DataFrame(index=list(latency_limits), columns=[round(u * 100) for u in idle_fracs], dtype=float)
    for u in idle_fracs:
        base_work = round(100 * (1 - u))
        for lim in latency_limits:
            alloc = [base_work] * n
            for oi, origin in enumerate(zones):
                work = base_work
                allowed = lat[origin][lat[origin] <= lim].index
                for dest in means.loc[allowed].sort_values().index:
                    di = zones.index(dest)
                    if alloc[di] <= 100:
                        move = min(work, 100 - alloc[di])
                        alloc[di] += move
                        alloc[oi] -= move
                        work -= move
                    if work <= 0:
                        break
                if work > 0:
                    alloc[oi] = work
            out.loc[lim, round(u * 100)] = float((np.array(alloc) * means).sum() / (base_work * n))
    return out
