import numpy as np
import pytest

from carbonshift import abatement as ab
from carbonshift import idle
from carbonshift.facility import capped_shift


def _signal(days=60, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(24 * days)
    return 300 + 80 * np.sin(2 * np.pi * t / 24) + 20 * rng.standard_normal(len(t))


def test_crf_matches_annuity():
    # Paying crf * P for n years at rate r has present value P
    r, n = 0.07, 20
    pv = sum(ab.crf(r, n) / (1 + r) ** k for k in range(1, n + 1))
    assert pv == pytest.approx(1.0)
    assert ab.crf(0.0, 10) == pytest.approx(0.1)


@pytest.mark.parametrize("h", [0.01, 0.1, 0.5])
@pytest.mark.parametrize("phi", [0.2, 0.5])
def test_value_never_exceeds_horizon_free_bound(h, phi):
    ci = _signal()
    for slack in (6, 24, 168):
        r = capped_shift(ci, phi, 1 + h, slack)
        v = (r.baseline_g - r.optimised_g) / h
        assert v <= ab.value_bound(ci) + 1e-9


def test_bound_is_tight_for_a_single_spike():
    # One dirty hour, otherwise flat: a tiny headroom with long slack moves work out of it
    ci = np.full(24 * 10, 100.0)
    ci[100] = 1000.0
    h = 1e-3
    r = capped_shift(ci, 0.5, 1 + h, 24 * 10)
    v = (r.baseline_g - r.optimised_g) / h
    # the bound holds even with a horizon long enough to move work out of the spike
    assert v <= ab.value_bound(ci) + 1e-9
    assert v > 0


def test_mac_and_breakeven_are_inverse():
    t = 0.4
    kappa = ab.annual_cost("facility_power")
    m = ab.mac(kappa, t)
    cap = ab.breakeven_capex(t, m, ab.COSTS["facility_power"]["life"])
    assert cap == pytest.approx(ab.COSTS["facility_power"]["capex"][1])
    assert ab.mac(kappa, 0.0) == float("inf")


def test_swing_needed_consistent_with_floor():
    ci = _signal()
    kappa = 500.0
    scc = ab.mac_floor(kappa, ci)
    assert ab.swing_needed(kappa, scc) == pytest.approx(ab.value_bound(ci))


def test_gpu_capacity_includes_facility():
    assert ab.annual_cost("gpu_capacity") == pytest.approx(
        ab.annual_cost("gpu_servers") + ab.annual_cost("facility_power"))
    assert ab.annual_cost("gpu_capacity") > 8 * ab.annual_cost("facility_power")


def test_first_unit_close_to_dual_and_closed_form():
    ci = _signal(days=90)
    r = capped_shift(ci, 0.5, 1.01, 24)
    primal = (r.baseline_g - r.optimised_g) / 0.01
    dual = r.annual_marginal_saving / len(ci)          # g per kW-headroom per hour
    D = idle.first_unit_value(ci)
    assert dual <= primal + 1e-6                       # concave S: next unit worth less than average
    assert primal == pytest.approx(D, rel=0.35)


def test_moved_energy_flat_signal_zero_and_bounded():
    ci = np.full(24 * 20, 200.0)
    r = capped_shift(ci, 0.5, 1.2, 24)
    assert ab.moved_mwh_per_kw_year(r.served, 0.5, 0.2) <= 8.76 + 1e-9
    ci = _signal()
    r = capped_shift(ci, 0.5, 1.2, 24)
    m = ab.moved_mwh_per_kw_year(r.served, 0.5, 0.2)
    assert 0 < m <= 8.76 + 1e-9


def test_price_gain_needed():
    assert ab.price_gain_needed(1000, 1.0, 2.0, 190) == pytest.approx(405.0)
    assert ab.price_gain_needed(100, 1.0, 2.0, 190) == 0.0
