"""Hedging between average and marginal carbon signals.

A capped facility schedules on the blended signal  w*avg + (1-w)*marginal  and is
scored on both. Because the LP is linear, sweeping w traces the Pareto frontier of
(average saving, marginal saving) achievable by any schedule (weighted-sum
scalarisation of a convex problem recovers every supported Pareto point).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from carbonshift.facility import capped_shift


def saving(ci: np.ndarray, served: np.ndarray, phi: float) -> float:
    """g/kWh-of-mean-load saved by schedule `served` when accounted on signal `ci`."""
    return float(ci.mean() - (ci * ((1 - phi) + served)).mean())


def frontier(avg: np.ndarray, mar: np.ndarray, phi: float, cap, slack: int, weights) -> pd.DataFrame:
    """Savings on each signal, absolute and as a share of that signal's own capped optimum, per weight w on avg."""
    if avg.shape != mar.shape:
        raise ValueError("signals must have equal length")
    best_a = saving(avg, capped_shift(avg, phi, cap, slack).served, phi)
    best_m = saving(mar, capped_shift(mar, phi, cap, slack).served, phi)
    rows = []
    for w in weights:
        s = capped_shift(w * avg + (1 - w) * mar, phi, cap, slack).served
        ga, gm = saving(avg, s, phi), saving(mar, s, phi)
        rows.append({"w_avg": float(w), "avg_saving_g": ga, "marg_saving_g": gm,
                     "avg_share": ga / best_a if best_a > 1e-9 else np.nan,
                     "marg_share": gm / best_m if best_m > 1e-9 else np.nan})
    return pd.DataFrame(rows)
