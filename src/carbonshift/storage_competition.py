"""Grid batteries compete with carbon-aware shifting (F32).

F22 showed that a flexible workload under a power cap *is* a lossless battery. A grid
that already has batteries then has two batteries, and they compete for the same
resource: the clean surplus of the middle of the day. Power-systems economics has a
standard no-arbitrage result for this. If a storage fleet with round-trip efficiency
eta charges in hour a and discharges in hour b, and none of its power or energy
limits bind, the marginal cost of energy in the two hours satisfies

    lambda_a = eta * lambda_b.

For an emission-minimising dispatch the same holds for marginal *emission* rates
(m_a = eta * m_b), so moving one kWh of load from b to a cuts emissions by only
(1 - eta) * m_b: the shift displaces battery charging, the battery discharges eta kWh
less in b, and that is replaced by the marginal generator. A carbon-aware scheduler's
saving on a storage-rich grid is the battery's round-trip loss, not the hourly swing.

Model (CAISO, the F24 setting): each hour has a clean surplus S_t (observed
curtailment, which is already net of the *existing* battery fleet) and otherwise a
constant gas margin. We add (i) a data-centre fleet of mean load P with flexible
share phi, power cap and 24 h slack, and (ii) an *additional* grid battery
(power Pb, energy Eb, efficiency eta) dispatched to minimise emissions. Per day
(state of charge empty at midnight), the LP is

    min  sum_t u_t
    s.t. u_t >= base + y_t + c_t - d_t - S_t,     u_t >= -G_t,
         0 <= y_t <= ycap,  sum_t y_t = 24 phi P,
         0 <= c_t, d_t <= Pb,  0 <= sum_{k<=t} (c_k - d_k / eta) <= Eb,

where u_t is the change in gas output (MWh) and G_t the gas that can be displaced
(thermal + imports in hours without surplus, zero otherwise).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

H = 24


@dataclass
class DayResult:
    gas: float  # change in gas output, MWh (negative = displaced)
    y: np.ndarray  # flexible DC energy served each hour (MWh)
    charge: np.ndarray
    discharge: np.ndarray
    marginal: np.ndarray  # d(gas)/d(load_t), MWh gas per MWh of load (dual of the balance rows)


def _day_lp(S, G, base, flex_total, ycap, Pb, Eb, eta, fixed_y=None) -> DayResult:
    """Solve one day. Variable order: y(24), c(24), d(24), u(24)."""
    S = np.asarray(S, float)
    G = np.asarray(G, float)
    n = 4 * H
    iy, ic, idd, iu = (slice(k * H, (k + 1) * H) for k in range(4))
    I = sparse.identity(H, format="csr")
    Z = sparse.csr_matrix((H, H))
    L = sparse.csr_matrix(np.tril(np.ones((H, H))))
    # balance: y + c - d - u <= S - base
    A_bal = sparse.hstack([I, I, -I, -I])
    b_bal = S - base
    # soc upper: cumsum(c - d/eta) <= Eb ; soc lower: -cumsum(c - d/eta) <= 0
    A_soc = sparse.hstack([Z, L, -L / eta, Z])
    A_ub = sparse.vstack([A_bal, A_soc, -A_soc]).tocsr()
    b_ub = np.concatenate([b_bal, np.full(H, Eb), np.zeros(H)])
    if fixed_y is not None:
        ylo = yhi = np.asarray(fixed_y, float)
        A_eq = b_eq = None
    else:
        ylo, yhi = np.zeros(H), np.full(H, ycap)
        A_eq = sparse.hstack([sparse.csr_matrix(np.ones((1, H))), sparse.csr_matrix((1, 3 * H))]).tocsr()
        b_eq = np.array([flex_total])
    bounds = ([(lo, hi) for lo, hi in zip(ylo, yhi)] + [(0, Pb)] * H + [(0, Pb)] * H
              + [(-g, None) for g in G])
    cost = np.zeros(n)
    cost[iu] = 1.0
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    x = res.x
    # dual of balance row t w.r.t. its rhs (S_t - base): raising load by 1 lowers the rhs by 1
    mg = -res.ineqlin.marginals[:H]
    return DayResult(gas=float(x[iu].sum()), y=x[iy], charge=x[ic], discharge=x[idd], marginal=mg)


def caiso_days(df):
    """Per-day arrays (S, G) in MWh from the F24 hourly CAISO frame; drops incomplete days."""
    d = df.copy()
    d = d[d.groupby("date").hour.transform("count") >= H]
    d = d[d.hour <= H]
    S = d.curtail.to_numpy().reshape(-1, H)
    G = np.where(S > 0, 0.0, (d.thermal + d.imports).clip(lower=0).to_numpy().reshape(-1, H))
    return S, G


def fleet_year(S, G, P, phi, cap, Pb, Eb, eta, shifted=True):
    """Total gas change (MWh) over all days for a DC fleet (flat or shifted) plus extra storage."""
    base = (1.0 - phi) * P
    ycap = (cap - (1.0 - phi)) * P if np.isfinite(cap) else 24 * phi * P
    tot = 0.0
    for s, g in zip(S, G):
        fy = None if shifted else np.full(H, phi * P)
        tot += _day_lp(s, g, base, 24 * phi * P, ycap, Pb, Eb, eta, fixed_y=fy).gas
    return tot


def shifting_saving(S, G, P, phi, cap, Pb, Eb, eta, m_gas=490.0):
    """Real saving of carbon-aware shifting, g per kWh of flexible energy, with storage re-optimised."""
    flat = fleet_year(S, G, P, phi, cap, Pb, Eb, eta, shifted=False)
    sh = fleet_year(S, G, P, phi, cap, Pb, Eb, eta, shifted=True)
    flex_mwh = 24 * phi * P * len(S)
    return m_gas * (flat - sh) / flex_mwh


def greedy_saving(S, G, P, phi, cap, Pb, Eb, eta, m_gas=490.0):
    """Independent closed-form-style estimate (no LP).

    Each day the DC takes surplus first (efficiency 1): flat it absorbs sum min(S, P);
    shifted it water-fills its flexible energy into the largest-surplus hours, up to
    the cap. The extra battery then charges the leftover surplus, limited by Pb per
    hour and Eb per day, and returns eta of it. Battery timing is ignored (assumes
    the discharge hours come later the same day), so this is an approximation.
    """
    base = (1.0 - phi) * P
    top = cap * P if np.isfinite(cap) else np.inf
    W = 24 * phi * P
    tot_flat = tot_sh = 0.0
    for s in S:
        rest_flat = np.maximum(s - P, 0.0)
        a_flat = np.minimum(s, P).sum()
        # shifted: always-on base, then flexible work placed on the largest surpluses
        a_base = np.minimum(s, base)
        room = np.minimum(np.maximum(s - base, 0.0), top - base)
        order = np.argsort(-room)
        left = W
        take = np.zeros(H)
        for t in order:
            if left <= 0 or room[t] <= 0:
                break
            take[t] = min(room[t], left)
            left -= take[t]
        a_sh = a_base.sum() + take.sum()
        rest_sh = s - a_base - take

        def batt(rest):
            return eta * min(np.minimum(rest, Pb).sum(), Eb)

        tot_flat += a_flat + batt(rest_flat)
        tot_sh += a_sh + batt(rest_sh)
    return m_gas * (tot_sh - tot_flat) / (W * len(S))
