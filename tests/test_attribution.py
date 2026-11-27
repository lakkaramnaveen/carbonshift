import numpy as np
import pytest

from carbonshift import attribution as at
from carbonshift import waterfill as wf


def _sinusoid(days=30, amp=80.0):
    t = np.arange(24 * days)
    return 300 - amp * np.cos(2 * np.pi * (t + 0.5) / 24)


def _noisy(days=40, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(24 * days)
    return 300 + 70 * np.sin(2 * np.pi * t / 24) + 25 * rng.standard_normal(len(t))


@pytest.mark.parametrize("h", [0.05, 0.2, 0.42, 1.0, 3.0])
def test_euler_identity_is_exact_for_waterfilling(h):
    ci = _noisy()
    e = at.euler_waterfill(ci, 0.5, h)
    assert abs(e["residual"]) < 1e-9 * max(1.0, e["S"])


@pytest.mark.parametrize("h", [0.15, 0.42, 0.9])  # k = 24 phi / (phi + h) not an integer (S is kinked there)
def test_analytic_marginals_match_finite_differences(h):
    ci = _noisy()
    a, f = at.euler_waterfill(ci, 0.5, h), at.euler_fd(ci, 0.5, h, eps=1e-6)
    assert a["S_phi"] == pytest.approx(f["S_phi"], rel=1e-4, abs=1e-6)
    assert a["S_h"] == pytest.approx(f["S_h"], rel=1e-4, abs=1e-6)


@pytest.mark.parametrize("h", [0.2, 0.42])
def test_sinc_closed_form_matches_waterfilling_and_lp(h):
    ci = _sinusoid()
    s = float(at.capacity_share_sinc(0.5, h))
    assert at.euler_waterfill(ci, 0.5, h)["sigma_h"] == pytest.approx(s, abs=0.015)
    lp = at.euler_lp(ci, 0.5, h)
    assert lp["sigma_h"] == pytest.approx(s, abs=0.015)
    assert abs(lp["residual"]) < 0.01 * lp["S"]


def test_share_depends_only_on_headroom_per_flexible_load():
    ci = _noisy()
    a = at.euler_waterfill(ci, 0.5, 0.2)["sigma_h"]
    b = at.euler_waterfill(ci, 0.25, 0.1)["sigma_h"]
    assert a == pytest.approx(b, rel=1e-9)


def test_limits_of_the_capacity_share():
    assert float(at.capacity_share_sinc(0.5, 1e-6)) == pytest.approx(1.0, abs=1e-4)
    big = 1e4
    q = 0.5 / (0.5 + big)
    assert float(at.capacity_share_sinc(0.5, big)) == pytest.approx((np.pi * q) ** 2 / 3, rel=1e-3)
    hs = np.linspace(0.01, 5, 200)
    s = at.capacity_share_sinc(0.5, hs)
    assert np.all(np.diff(s) < 0) and np.all((s > 0) & (s < 1))


def test_marginal_flexibility_never_negative_and_complements():
    ci = _noisy()
    for h in (0.05, 0.2, 0.5):
        assert at.euler_waterfill(ci, 0.5, h)["S_phi"] >= -1e-12
    # complements: the marginal value of flexibility rises with headroom
    sp = [at.euler_waterfill(ci, 0.5, h)["S_phi"] for h in (0.05, 0.2, 0.5, 1.0)]
    assert np.all(np.diff(sp) > 0)


def test_overclaim_factor():
    assert at.overclaim_factor(0.8) == pytest.approx(5.0)
    assert at.overclaim_factor(0.0) == pytest.approx(1.0)


def test_sum_of_job_counterfactuals_is_flexibility_share():
    # Split flexible load among 50 equal jobs; removing one job's flexibility (it becomes
    # inflexible, so the flexible rate falls by its share at fixed h) loses ~ S_phi * phi_i.
    ci = _noisy()
    phi, h, n = 0.5, 0.2, 50
    S = wf.saving_waterfill(ci, phi, 1 + h)
    loss = S - wf.saving_waterfill(ci, phi - phi / n, 1 + h)
    e = at.euler_waterfill(ci, phi, h)
    assert n * loss == pytest.approx((1 - e["sigma_h"]) * S, rel=0.05)
    assert n * loss < 0.5 * S
