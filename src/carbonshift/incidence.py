"""Who bears the cost of carbon-aware shifting? Incidence under grid carbon accounting (F28).

Location-based Scope 2 accounting charges every consumer the hour's *average*
intensity c_t = E_t / D_t. It is budget balanced: the attributions of all consumers
sum to the grid's emissions E. So when a carbon-aware load moves work in time, the
change in its own attribution plus the change in everyone else's equals the change
in real emissions, exactly:

    dA_shifter + dA_others = dE.

The "saving" a location-based scheduler reports is therefore its real effect on the
grid *plus* emissions moved onto other consumers' inventories. To first order (a small
consumer, grid emissions responding at the marginal rate m_t) the moved part is

    transfer = sum_t (m_t - c_t) * dL_t  =  claimed saving - real saving,

because the average rate decomposes as c_t = m_t + r_t: a marginal charge plus an
*hourly residual* r_t = c_t - m_t. Location-based accounting is marginal accounting
with the residual allocated hour by hour; where r_t is very negative (clean hours,
gas at the margin) it subsidises load that the grid serves with fossil energy.

A two-part rule (Coase's two-part tariff, applied to carbon) keeps the marginal charge
and allocates the residual by *annual* energy instead of hourly:

    a_i = sum_t m_t L_it + rho * sum_t L_it,    rho = (E - sum_t m_t D_t) / sum_t D_t.

It is budget balanced like location-based accounting (sum_i a_i = E), but a shift
that conserves energy changes a_i by sum_t m_t dL_t: the reported change equals the
real one, and nobody else's inventory moves (exactly, under a linear response).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def location_attribution(E: np.ndarray, D: np.ndarray, load: np.ndarray) -> float:
    """Location-based attribution: sum_t (E_t / D_t) * load_t."""
    E, D, load = (np.asarray(x, dtype=float) for x in (E, D, load))
    return float((E / D * load).sum())


def two_part_residual(E: np.ndarray, D: np.ndarray, m: np.ndarray) -> float:
    """Annual residual rate rho = (sum E - sum m D) / sum D (can be negative)."""
    E, D, m = (np.asarray(x, dtype=float) for x in (E, D, m))
    return float((E.sum() - (m * D).sum()) / D.sum())


def two_part_attribution(E: np.ndarray, D: np.ndarray, m: np.ndarray, load: np.ndarray) -> float:
    """Two-part attribution: marginal charge per hour plus the annual residual per unit of energy."""
    m, load = np.asarray(m, dtype=float), np.asarray(load, dtype=float)
    return float((m * load).sum() + two_part_residual(E, D, m) * load.sum())


def linear_incidence(avg: np.ndarray, mar: np.ndarray, draw: np.ndarray) -> dict:
    """First-order incidence of moving from a flat draw (mean 1) to ``draw``, per unit of mean load.

    Units follow the signals (g/kWh -> g per kWh of the shifter's mean load).
    claimed = location-based saving the shifter reports; real = drop in grid emissions;
    transfer = rise in everyone else's location-based attribution = claimed - real.
    """
    avg, mar, draw = (np.asarray(x, dtype=float) for x in (avg, mar, draw))
    if not (avg.shape == mar.shape == draw.shape):
        raise ValueError("signals and draw must have equal length")
    dL = draw - 1.0
    claimed = -float((avg * dL).mean())
    real = -float((mar * dL).mean())
    return {"claimed": claimed, "real": real, "transfer": claimed - real}


def exact_incidence(E0: np.ndarray, D0: np.ndarray, emis: Callable[[np.ndarray], np.ndarray],
                    flat: np.ndarray, shifted: np.ndarray, m: np.ndarray) -> dict:
    """Exact (finite-size) incidence of a new consumer moving from ``flat`` to ``shifted``.

    The grid serves others' load D0 with emissions E0; the new consumer drawing L adds
    ``emis(L)`` (hourly array). ``m`` is the marginal rate used by the two-part rule.
    Returns, for both rules, the change in the consumer's attribution, in everyone
    else's, and the change in real emissions (all in the units of E0).
    """
    E0, D0, m = (np.asarray(x, dtype=float) for x in (E0, D0, m))

    def state(L):
        L = np.asarray(L, dtype=float)
        E, D = E0 + emis(L), D0 + L
        loc_c, loc_o = location_attribution(E, D, L), location_attribution(E, D, D0)
        tp_c, tp_o = two_part_attribution(E, D, m, L), two_part_attribution(E, D, m, D0)
        return float(E.sum()), loc_c, loc_o, tp_c, tp_o

    a, b = state(flat), state(shifted)
    d = [y - x for x, y in zip(a, b)]
    return {"real": d[0], "location_self": d[1], "location_others": d[2],
            "two_part_self": d[3], "two_part_others": d[4]}
