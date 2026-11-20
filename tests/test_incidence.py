import numpy as np
import pytest

from carbonshift.facility import capped_shift
from carbonshift.incidence import (exact_incidence, linear_incidence, location_attribution,
                                   two_part_attribution, two_part_residual)


def _grid(n=24 * 20, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    D0 = 1000 + 200 * np.sin(2 * np.pi * (t - 6) / 24) + rng.normal(0, 10, n)        # MWh
    clean = np.clip(np.sin(2 * np.pi * (t - 6) / 24), 0, None) * 500                    # midday solar
    E0 = 490 * np.maximum(D0 - clean, 0.3 * D0)                                         # kg, gas covers the rest
    m = np.full(n, 490.0)
    return D0, E0, m


def test_both_rules_are_budget_balanced():
    D0, E0, m = _grid()
    loads = [0.3 * D0, 0.7 * D0]
    assert sum(location_attribution(E0, D0, L) for L in loads) == pytest.approx(E0.sum(), rel=1e-12)
    assert sum(two_part_attribution(E0, D0, m, L) for L in loads) == pytest.approx(E0.sum(), rel=1e-12)


def test_residual_closes_the_books():
    D0, E0, m = _grid()
    rho = two_part_residual(E0, D0, m)
    assert (m * D0).sum() + rho * D0.sum() == pytest.approx(E0.sum(), rel=1e-12)


def test_linear_transfer_is_claimed_minus_real_and_sums_hourly_residual():
    D0, E0, m = _grid()
    avg = E0 / D0
    draw = 1 - 0.5 + capped_shift(avg, 0.5, 1.4, 24).served
    r = linear_incidence(avg, m, draw)
    assert r["transfer"] == pytest.approx(r["claimed"] - r["real"])
    assert r["transfer"] == pytest.approx(-((avg - m) * (draw - 1)).mean())             # -sum r_t dL_t
    # constant marginal rate: shifting changes nothing real, so the whole claim is transfer
    assert r["real"] == pytest.approx(0.0, abs=1e-9)
    assert r["claimed"] > 1.0


def test_exact_identity_and_two_part_alignment_under_linear_response():
    D0, E0, m = _grid(seed=2)
    P = 1.0                                                                               # 0.1% of grid load
    avg = E0 / D0
    flat = np.full(len(D0), P)
    shifted = P * (0.5 + capped_shift(avg, 0.5, 1.4, 24).served)
    mt = 300 + 200 * np.cos(2 * np.pi * np.arange(len(D0)) / 24)                         # time-varying margin
    res = exact_incidence(E0, D0, lambda L: mt * L, flat, shifted, mt)
    # budget identity holds exactly for both rules
    assert res["location_self"] + res["location_others"] == pytest.approx(res["real"], abs=1e-6)
    assert res["two_part_self"] + res["two_part_others"] == pytest.approx(res["real"], abs=1e-6)
    # two-part: the shifter's reported change is the real change, others untouched
    assert res["two_part_self"] == pytest.approx(res["real"], abs=1e-6)
    assert res["two_part_others"] == pytest.approx(0.0, abs=1e-6)
    # location-based: first-order transfer matches the exact one for a small consumer
    lin = linear_incidence(avg, mt, shifted / P)
    assert res["location_others"] / (P * len(D0)) == pytest.approx(lin["transfer"], rel=0.005)


def test_two_part_is_gaming_proof_for_energy_preserving_shifts():
    """Under two-part accounting, an energy-preserving shift on a flat-margin grid reports zero, whatever the average does."""
    D0, E0, m = _grid(seed=4)
    P = 5.0
    avg = E0 / D0
    shifted = P * (0.5 + capped_shift(avg, 0.5, 1.4, 24).served)
    res = exact_incidence(E0, D0, lambda L: 490 * L, np.full(len(D0), P), shifted, m)
    assert res["real"] == pytest.approx(0.0, abs=1e-6)
    assert res["two_part_self"] == pytest.approx(0.0, abs=1e-6)
    assert res["location_self"] < -0.01 * P * len(D0)                                   # reports a saving that is not there
    assert res["location_others"] == pytest.approx(-res["location_self"], rel=1e-9)


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        linear_incidence(np.ones(3), np.ones(3), np.ones(4))
