"""Storage exchange rate (F23): how many kWh of real battery a flexible load is worth.

F22 showed that capped flexible load is a lossless battery (energy s*phi, discharge phi,
charge h). A real grid battery has losses and a fixed duration. Its carbon saving per
kWh of capacity is, for daily cycling, the daily *spread law*

    b = (1/24) * mean_days[ max(0, sqrt(eta) * mean(top-D hours) - mean(bottom-D hours) / sqrt(eta)) ]

g per kWh of capacity per hour (D = duration in hours: a D-hour battery delivers its
full capacity over the D dirtiest hours and recharges in the D cleanest).
The exchange rate is then X = S_flex / b kWh of battery per kW of mean facility load.
"""

from __future__ import annotations

import numpy as np


def spread_saving_per_kwh(ci: np.ndarray, duration_h: float = 4.0, round_trip_eff: float = 0.90) -> float:
    """Closed-form carbon saving (g per kWh of capacity, per hour) of a daily-cycling battery.

    Each day the battery charges 1 kWh (grid side 1/sqrt(eta)) in the D cleanest hours and
    delivers sqrt(eta) kWh in the D dirtiest; it skips days whose spread does not cover the losses.
    """
    days = np.asarray(ci, dtype=float)[: len(ci) // 24 * 24].reshape(-1, 24)
    srt = np.sort(days, axis=1)
    k = float(duration_h)
    kf = int(np.floor(k))

    def part_mean(a):
        tot = a[:, :kf].sum(axis=1) + (k - kf) * (a[:, kf] if kf < 24 else 0.0)
        return tot / k

    low, high = part_mean(srt), part_mean(srt[:, ::-1])
    s = np.sqrt(round_trip_eff)
    gain = np.maximum(0.0, high * s - low / s)
    return float(gain.mean() / 24)


def exchange_rate(flex_saving_g: float, battery_saving_per_kwh_g: float) -> float:
    """kWh of battery per kW of mean load that saves as much carbon as the flexible load."""
    if battery_saving_per_kwh_g <= 0:
        return float("inf") if flex_saving_g > 0 else 0.0
    return float(flex_saving_g / battery_saving_per_kwh_g)
