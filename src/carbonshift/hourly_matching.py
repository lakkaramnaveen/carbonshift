"""Hourly matching (24/7 CFE) and load shifting: what the score rewards vs what the grid sees (F30).

Under hourly-matched market-based accounting (Google's 24/7 CFE; the "Method 1"
option of the GHG Protocol Scope 2 revision) a consumer with a contracted clean
portfolio P_t is charged only for its *unmatched* load u_t = max(L_t - P_t, 0),
at the hour's grid (residual-mix) intensity c_t. Surplus P_t - L_t > 0 earns
nothing: the CFE score ignores it. So the reported emissions are

    A_hm = sum_t c_t * max(L_t - P_t, 0),

and moving one kWh from a deficit hour into a surplus hour of the consumer's own
portfolio lowers A_hm by c_deficit, a full residual-mix kWh.

The grid sees something else. The portfolio is already built and its output P_t
does not depend on the consumer's schedule; a kWh of surplus that the consumer
absorbs is a kWh *not exported*, which the rest of the grid replaces at the
marginal rate m_t. With P fixed the real change is therefore

    dE = sum_t m_t * dL_t,

exactly as for a consumer with no portfolio (F28): grid signals, not the PPA's
shape, decide whether a shift helps. The difference between the reported and the
real change is moved onto other consumers' residual-mix inventories (hourly
market-based accounting is budget balanced, like location-based accounting), so

    transfer = reported saving - real saving.

A procurement decision (building P) can have real effects; *shifting against* a
fixed P is a private claim on clean energy that the neighbours would otherwise
have used.

This module provides the hourly-matching LP for a capped flexible facility (the
F17/F28 setting), a clear-sky solar profile for a portfolio sized to the annual
load, and the ledger of reported vs real vs transferred carbon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from carbonshift.facility import _cumulative_bounds

# Approximate service-territory centroids (lat, lon) of the US balancing authorities in
# data/eia930 (public geography, rounded to ~0.5 degree; used only to shape a solar PPA).
BA_LATLON: dict[str, tuple[float, float]] = {
    "AECI": (37.5, -93.0), "AVA": (47.5, -117.0), "AZPS": (33.5, -112.0), "BANC": (38.6, -121.4),
    "BPAT": (45.6, -121.0), "CHPD": (47.4, -120.3), "CISO": (36.0, -119.5), "CPLE": (35.5, -78.5),
    "CPLW": (35.6, -82.6), "DOPD": (47.8, -120.0), "DUK": (35.3, -80.8), "EPE": (31.8, -106.4),
    "ERCO": (31.0, -97.5), "FMPP": (28.5, -81.4), "FPC": (28.5, -82.2), "FPL": (26.8, -80.6),
    "GCPD": (47.1, -119.3), "GRID": (45.0, -116.0), "GVL": (29.7, -82.3), "IID": (33.0, -115.5),
    "IPCO": (43.6, -115.5), "ISNE": (42.5, -71.8), "JEA": (30.3, -81.7), "LDWP": (34.1, -118.3),
    "LGEE": (37.8, -85.5), "MISO": (40.0, -90.0), "NEVP": (36.5, -115.5), "NWMT": (46.5, -110.0),
    "NYIS": (42.5, -75.0), "PACE": (41.5, -111.5), "PACW": (44.0, -122.5), "PGE": (45.4, -122.7),
    "PJM": (39.8, -78.5), "PNM": (35.0, -106.5), "PSCO": (39.7, -105.0), "PSEI": (47.5, -122.2),
    "SC": (33.6, -80.5), "SCEG": (33.8, -81.0), "SCL": (47.6, -122.3), "SEC": (28.8, -82.0),
    "SOCO": (32.8, -85.5), "SPA": (35.5, -93.5), "SRP": (33.4, -111.9), "SWPP": (37.5, -98.0),
    "TAL": (30.4, -84.3), "TEC": (27.9, -82.4), "TEPC": (32.2, -110.9), "TIDC": (37.5, -120.8),
    "TPWR": (47.2, -122.4), "TVA": (35.8, -86.5), "WACM": (40.5, -106.0), "WALC": (34.5, -113.5),
    "WAUW": (47.0, -105.0),
}


def clear_sky_solar(utc: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    """Hourly clear-sky irradiance shape (W/m^2, Haurwitz model) at hour-centre, UTC index.

    Solar position from the standard declination / equation-of-time approximations
    (accurate to ~0.5 degree, ample for a PPA *shape*).
    """
    t = pd.DatetimeIndex(utc) + pd.Timedelta(minutes=30)
    doy = t.day_of_year.to_numpy()
    hour = t.hour.to_numpy() + t.minute.to_numpy() / 60
    g = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))   # minutes
    solar_time = hour * 60 + eot + 4 * lon                                   # minutes
    ha = np.deg2rad(solar_time / 4 - 180)
    phi = np.deg2rad(lat)
    cosz = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    cosz = np.clip(cosz, 0, None)
    with np.errstate(divide="ignore", over="ignore"):
        ghi = np.where(cosz > 0.01, 1098 * cosz * np.exp(-0.057 / np.maximum(cosz, 1e-3)), 0.0)
    return ghi


def portfolio(shape: np.ndarray, annual_match: float = 1.0) -> np.ndarray:
    """Clean portfolio output with the given shape, sized to ``annual_match`` x the mean load of 1."""
    shape = np.asarray(shape, dtype=float)
    if shape.sum() <= 0:
        raise ValueError("portfolio shape must have positive energy")
    return shape / shape.mean() * annual_match


def cfe_score(load: np.ndarray, P: np.ndarray) -> float:
    """Hourly CFE score: share of load matched by the portfolio in the same hour."""
    load, P = np.asarray(load, dtype=float), np.asarray(P, dtype=float)
    return float(np.minimum(load, P).sum() / load.sum())


def hm_attribution(c: np.ndarray, load: np.ndarray, P: np.ndarray) -> float:
    """Hourly-matched market-based emissions per hour of mean load: mean_t c_t max(L_t - P_t, 0)."""
    c, load, P = (np.asarray(x, dtype=float) for x in (c, load, P))
    return float((c * np.maximum(load - P, 0)).mean())


def hm_shift(c: np.ndarray, P: np.ndarray, phi: float, cap, slack: int,
             tiebreak: np.ndarray | None = None, eps: float = 1e-4) -> np.ndarray:
    """Schedule minimising hourly-matched emissions sum_t c_t max(L_t - P_t, 0).

    Same facility model as :func:`carbonshift.facility.capped_shift` (flat load of mean 1,
    ``phi`` flexible, arrivals due within ``slack`` hours, power cap ``cap``). The objective
    is flat wherever the portfolio has surplus, so the LP has many optima; ``tiebreak`` (an
    hourly signal, weighted by ``eps`` relative to its own scale) picks among them, e.g.
    the grid average (a scheduler that also follows the public signal) or the marginal rate
    (the best case for the grid). Returns the total draw L_t.
    """
    c, P = np.asarray(c, dtype=float), np.asarray(P, dtype=float)
    n = len(c)
    base = (1.0 - phi) * np.ones(n)
    arrivals = phi * np.ones(n)
    lo, hi = _cumulative_bounds(arrivals, slack)
    room = np.broadcast_to(np.asarray(cap, dtype=float), (n,)) - base
    if np.any(room < phi - 1e-12):
        raise ValueError("cap must be >= the load")
    room = np.where(np.isinf(room), arrivals.max() * (slack + 1), room)
    # variables [Y (cumulative served), u (unmatched)]; y = DY
    D = sparse.diags([np.ones(n), -np.ones(n - 1)], [0, -1], format="csr")
    I = sparse.identity(n, format="csr")
    Z = sparse.csr_matrix((n, n))
    A_ub = sparse.vstack([sparse.hstack([D, Z]), sparse.hstack([-D, Z]), sparse.hstack([D, -I])], format="csr")
    b_ub = np.concatenate([room, np.zeros(n), P - base])
    tb = np.zeros(n) if tiebreak is None else eps * np.asarray(tiebreak, dtype=float) / max(np.abs(tiebreak).mean(), 1e-12)
    # sum_t tb_t y_t = sum_t Y_t (tb_t - tb_{t+1})
    cost = np.concatenate([tb - np.append(tb[1:], 0.0), c])
    bounds = np.vstack([np.column_stack([lo, hi]), np.column_stack([np.zeros(n), np.full(n, np.inf)])])
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    return base + np.diff(res.x[:n], prepend=0.0)


def ledger(c: np.ndarray, m: np.ndarray, P: np.ndarray, draw: np.ndarray) -> dict:
    """Reported (hourly-matching) vs real (marginal) saving of ``draw`` against a flat draw of 1.

    g per kWh of the consumer's mean load. transfer = reported - real is the rise in other
    consumers' residual-mix inventories (first order, small consumer, fixed portfolio).
    """
    c, m, P, draw = (np.asarray(x, dtype=float) for x in (c, m, P, draw))
    flat = np.ones_like(draw)
    reported = hm_attribution(c, flat, P) - hm_attribution(c, draw, P)
    real = -float((m * (draw - flat)).mean())
    return {"reported": reported, "real": real, "transfer": reported - real,
            "cfe_flat": cfe_score(flat, P), "cfe_shifted": cfe_score(draw, P)}


def toy_grid_inventories(D0: np.ndarray, R: np.ndarray, P: np.ndarray, load: np.ndarray,
                         rate=lambda f: 400.0 * f + 0.05 * f ** 2) -> dict:
    """Exact hourly market-based inventories on a toy grid (independent check of the ledger).

    Others' demand D0_t, exogenous clean supply R_t, the consumer's portfolio P_t and its load
    L_t. Fossil output f_t = max(D0 + L - R - P, 0) with emissions rate(f) (convex merit order).
    The consumer claims min(L, P) of its portfolio; its unmatched load and all others' load
    share the residual mix E_t / (D0_t + u_t) (unclaimed clean energy, including the
    consumer's exported surplus, is in the residual mix). Returns total emissions E, the
    consumer's inventory and everyone else's (all summed over hours).
    """
    D0, R, P, load = (np.asarray(x, dtype=float) for x in (D0, R, P, load))
    f = np.maximum(D0 + load - R - P, 0)
    E = rate(f)
    u = np.maximum(load - P, 0)
    rm = E / (D0 + u)
    return {"E": float(E.sum()), "self": float((rm * u).sum()), "others": float((rm * D0).sum())}
