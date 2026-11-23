import numpy as np
import pytest

from carbonshift.waterbed import (
    RULES, MSRRules, TNAC_PUBLISHED, cap_regime, fate_bruteforce, fate_linear, implied_net_supply,
    net_supply_path, simulate,
)
from carbonshift.data import load_carbon


def test_published_intakes_reproduced():
    r = RULES["law_2023 (400 Mt)"]
    # 2024 TNAC above 1,096 Mt -> 24% intake (Commission: 275,531,900)
    assert r.net_intake(TNAC_PUBLISHED[2024]) == pytest.approx(275.532, abs=0.01)
    # 2025 TNAC in the band -> TNAC - 833 (Commission: 190,494,202)
    assert r.net_intake(TNAC_PUBLISHED[2025]) == pytest.approx(190.494, abs=0.01)
    d = implied_net_supply()
    assert d[2025] == pytest.approx(145.0, abs=0.5)


def test_pure_cap_is_a_waterbed():
    d = net_supply_path(100, 30)
    for y in (2025, 2026, 2030):
        f = fate_bruteforce(d, RULES["pure cap"], y)
        assert f["real_upper"] == pytest.approx(0.0, abs=1e-9)
        assert f["leaked"] == pytest.approx(1.0)


def test_band_absorbs_and_invalidation_destroys():
    # TNAC held inside the band by zero net supply: every avoided tonne goes into the MSR,
    # which sits above the threshold, so it is destroyed
    hist = {2024: 1000.0, 2025: 1000.0}
    r = MSRRules(split=False, inval=((2023, 0.0),))
    d = {y: 0.0 for y in range(2026, 2040)}
    base = simulate(d, r, tnac_hist=hist)
    pert = simulate(d, r, abate={2027: 1.0}, tnac_hist=hist)
    assert pert.total_invalidated - base.total_invalidated == pytest.approx(1.0)


def test_above_upper_threshold_geometric_absorption():
    # TNAC kept far above 1,096 Mt, calendar-year intake: an avoided tonne is absorbed at
    # 24%/yr, so after n intake years 1 - 0.76^n of it has entered the MSR
    hist = {2024: 5000.0, 2025: 5000.0}
    r = MSRRules(split=False, inval=((2023, 0.0),))
    d = {y: 0.24 * 5000.0 for y in range(2026, 2036)}  # keeps TNAC stationary
    f = fate_linear(d, r, 2026, tnac_hist=hist)
    n = 2035 - 2026  # intake years 2027..2035
    assert f["destroyed"] + f["locked"] == pytest.approx(1 - 0.76 ** n, abs=1e-9)
    fb = fate_bruteforce(d, r, 2026, size=1e-3, tnac_hist=hist)
    assert fb["real_upper"] == pytest.approx(f["real_upper"], abs=1e-6)


def test_below_band_no_absorption():
    hist = {2024: 700.0, 2025: 700.0}
    r = MSRRules(split=False)
    d = {y: 0.0 for y in range(2026, 2035)}
    base = simulate(d, r, tnac_hist=hist)
    pert = simulate(d, r, abate={2027: 1.0}, tnac_hist=hist)
    assert pert.total_invalidated == base.total_invalidated
    assert pert.msr_end == base.msr_end


@pytest.mark.parametrize("rules", ["law_2023 (400 Mt)", "deal_2026 (700 Mt)", "proposal_2026 (none)"])
def test_two_derivations_agree(rules):
    rng = np.random.default_rng(0)
    for _ in range(6):
        d = net_supply_path(float(rng.uniform(0, 200)), float(rng.uniform(0, 60)))
        for y in (2025, 2027, 2030):
            a = fate_linear(d, RULES[rules], y)
            b = fate_bruteforce(d, RULES[rules], y, size=1e-4)
            assert a["real_upper"] == pytest.approx(b["real_upper"], abs=1e-4)


def test_symmetry_extra_demand():
    # new demand (negative abatement) adds emissions in exactly the proportion an avoided tonne cuts them
    d = net_supply_path(125, 30)
    r = RULES["deal_2026 (700 Mt)"]
    up = fate_bruteforce(d, r, 2027, size=1e-3)["real_upper"]
    dn = fate_bruteforce(d, r, 2027, size=-1e-3)["real_upper"]
    assert up == pytest.approx(dn, abs=1e-6)


def test_every_zone_has_a_regime():
    zones = load_carbon(2022).columns
    regs = {z: cap_regime(z) for z in zones}
    assert regs["DE"] == "EU ETS (MSR)" and regs["GB"] == "UK ETS" and regs["US-CAL-CISO"].startswith("WCI")
    assert regs["US-TEX-ERCO"] == "no cap" and regs["GB-NIR"] == "EU ETS (MSR)"
    assert sum(r == "EU ETS (MSR)" for r in regs.values()) == 43


def test_split_intake_overshoots_but_converges_to_one():
    # Sep-Aug intake overlaps calendar years, so in the band an avoided tonne is absorbed with a
    # damped oscillation (roots of r^2 - 2r/3 + 2/3, |r| = sqrt(2/3)): transient overshoot, limit 1
    hist = {2024: 1000.0, 2025: 1000.0}
    r = MSRRules(split=True, inval=((2023, 0.0),))
    d = {y: 167.0 for y in range(2026, 2090)}  # intake = TNAC - 833 = 167: stationary in the band
    f = fate_linear(d, r, 2026, tnac_hist=hist)
    assert f["real_upper"] == pytest.approx(1.0, abs=1e-4)
    short = fate_linear({y: 167.0 for y in range(2026, 2029)}, r, 2026, tnac_hist=hist)
    assert short["real_upper"] > 1.0  # overshoot when the path ends mid-oscillation
