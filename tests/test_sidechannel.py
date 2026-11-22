import numpy as np
import pytest

from carbonshift.facility import capped_shift
from carbonshift.sidechannel import auc, detection_stat, infer_battery, saving_from_cov, time_of_day_signal

L = 24 * 120
HOURS = np.arange(L) % 24
MONTHS = 1 + (np.arange(L) // (24 * 30)) % 12


def _carbon(seed=0):
    rng = np.random.default_rng(seed)
    daily = np.repeat(rng.normal(0, 60, L // 24), 24)  # day-to-day weather swings
    return 400 + 120 * np.cos(2 * np.pi * HOURS / 24) + daily + rng.normal(0, 10, L)


def test_saving_equals_minus_covariance():
    ci = _carbon()
    r = capped_shift(ci, 0.5, 1.3, 24)
    draw = 0.5 + r.served
    assert saving_from_cov(ci, draw) == pytest.approx(r.baseline_g - r.optimised_g, abs=1e-9)


def test_detector_flags_realtime_but_not_diurnal_load():
    ci = _carbon(1)
    aware = 0.5 + capped_shift(ci, 0.5, 1.3, 24).served
    diurnal = 1 + 0.3 * np.cos(2 * np.pi * HOURS / 24)  # ordinary busy-by-day load, aligned with the carbon cycle
    assert detection_stat(ci, aware, HOURS) < -0.2
    assert abs(detection_stat(ci, diurnal, HOURS)) < 1e-9


def test_time_of_day_schedule_is_invisible_within_a_month():
    ci = _carbon(2)
    sig = time_of_day_signal(ci, HOURS, MONTHS)
    draw = 0.5 + capped_shift(sig, 0.5, 1.3, 24).served
    first_month = slice(0, 24 * 30)
    assert abs(detection_stat(ci[first_month], draw[first_month], HOURS[first_month])) < 0.05


def test_meter_reveals_the_virtual_battery():
    ci = _carbon(3)
    phi, s, cap = 0.3, 24, 1.5
    draw = (1 - phi) + capped_shift(ci, phi, cap, s).served
    e = infer_battery(draw)
    assert e["phi_hat"] == pytest.approx(phi, abs=0.01)
    assert e["h_hat"] == pytest.approx(cap - 1, abs=0.02)
    assert e["energy_hat"] == pytest.approx(s * phi, rel=0.15)


def test_auc_bounds():
    assert auc(np.array([-1.0, -2.0]), np.array([0.0, 1.0])) == 1.0
    assert auc(np.array([0.0]), np.array([0.0])) == 0.5
