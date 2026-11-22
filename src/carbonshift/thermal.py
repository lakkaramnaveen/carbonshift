"""Chilled-water thermal storage (TES) as a third virtual battery (F27).

The cooling plant must remove the IT heat every hour, but with a tank it can
make the cold earlier. Producing one unit of cold in hour t costs m(t) units of
electricity (m = 1/COP, the weather-dependent part of PUE from
``carbonshift.cooling``; 0 under full free cooling), so its carbon cost is
c(t) * m(t). The tank moves cold production to hours where that product is
low: cool nights (high COP) or clean grid hours, whichever wins.

LP (hourly, cyclic over the year), all in units of mean IT heat load:
    min  sum_t c_t m_t q_t
    s.t. S_t = (1 - loss) S_{t-1} + q_t - Q_t        (tank balance, cyclic)
         0 <= q_t <= q_max,  0 <= S_t <= capacity_h
where Q_t is the heat load (= IT draw) and q_t the cold produced.

Facility grid draw is d * (1 + fixed + fans) + m * q, so with no tank (q = Q)
it reduces to d * PUE. Tank losses are standing losses (default 1% of the
stored cold per day, typical of stratified chilled-water tanks).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from carbonshift.cooling import Plant, chiller_cop


@dataclass(frozen=True)
class TESResult:
    baseline_g: float     # cooling carbon without a tank, g per hour per unit mean heat (= g/kWh of mean IT)
    optimised_g: float    # with carbon-optimal tank dispatch
    produced: np.ndarray  # cold produced each hour
    state: np.ndarray     # stored cold at the end of each hour

    @property
    def saving_g(self) -> float:
        return self.baseline_g - self.optimised_g


def mech_per_heat(t_out: np.ndarray, plant: Plant) -> np.ndarray:
    """Electricity per unit of heat removed by the chiller/economiser (the weather part of PUE)."""
    t = np.asarray(t_out, float)
    mech = 1.0 / chiller_cop(t, plant)
    if plant.econ_full is not None:
        mech = np.clip((t - plant.econ_full) / (plant.econ_none - plant.econ_full), 0.0, 1.0) * mech
    return mech


def tes_shift(ci: np.ndarray, mech: np.ndarray, capacity_h: float, heat: np.ndarray | None = None,
              q_max: float | None = None, loss_per_day: float = 0.01) -> TESResult:
    """Carbon-optimal dispatch of a chilled-water tank of ``capacity_h`` hours of mean heat load.

    ``q_max`` is the chiller capacity (default 1.5 x peak heat). Returns cooling carbon only;
    the IT and fixed overheads are untouched by the tank.
    """
    ci, m = np.asarray(ci, float), np.asarray(mech, float)
    L = len(ci)
    Q = np.ones(L) if heat is None else np.asarray(heat, float)
    if Q.shape != (L,) or m.shape != (L,):
        raise ValueError("ci, mech and heat must have one value per hour")
    price = ci * m
    baseline = float((price * Q).mean())
    if capacity_h <= 0:
        return TESResult(baseline, baseline, Q.copy(), np.zeros(L))
    qmax = 1.5 * Q.max() if q_max is None else q_max
    if qmax < Q.max() - 1e-12:
        raise ValueError("chiller capacity must cover the peak heat load")
    keep = 1.0 - loss_per_day / 24.0
    # x = [q (L), S (L)];  S_t - keep*S_{t-1} - q_t = -Q_t
    I = sparse.identity(L, format="csr")
    prev = sparse.diags([np.ones(L - 1)], [-1], shape=(L, L), format="lil")
    prev[0, L - 1] = 1.0
    A_eq = sparse.hstack([-I, I - keep * prev.tocsr()], format="csr")
    obj = np.concatenate([price, np.zeros(L)])
    bounds = [(0, qmax)] * L + [(0, capacity_h)] * L
    res = linprog(obj, A_eq=A_eq, b_eq=-Q, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    q, S = res.x[:L], res.x[L:]
    return TESResult(baseline, float((price * q).mean()), q, S)
