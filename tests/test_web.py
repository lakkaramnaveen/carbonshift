"""API behaviour and security tests for the explorer (no network, no GPU)."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from carbonshift.web import server
from carbonshift.web.server import RateLimiter, app

BASE = "http://localhost"
ZONE = "US-CAL-CISO"


@pytest.fixture()
def client():
    app.fastapi.state.limiter = RateLimiter(10_000)
    return TestClient(app, base_url=BASE)


def test_health_and_security_headers(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    h = r.headers
    assert "default-src 'self'" in h["content-security-policy"]
    assert "script-src 'self'" in h["content-security-policy"] and "unsafe-inline" not in h["content-security-policy"]
    assert h["x-content-type-options"] == "nosniff"
    assert h["x-frame-options"] == "DENY"
    assert h["referrer-policy"] == "no-referrer"
    assert h["cache-control"] == "no-store"
    assert "server" not in {k.lower() for k in h if k.lower() == "x-powered-by"}


def test_meta_lists_all_zones(client):
    d = client.get("/api/meta").json()
    assert len(d["zones"]) == 123
    assert {"code", "name", "region", "mean_ci"} <= d["zones"][0].keys()


def test_temporal_matches_library(client):
    from carbonshift.data import load_carbon
    from carbonshift.temporal import shift_one_zone

    r = client.post("/api/temporal", json={"zone": ZONE, "job_hours": 24, "slack_hours": 24})
    assert r.status_code == 200
    expect = shift_one_zone(load_carbon(2022)[ZONE].to_numpy(), 24, 24).per_unit(24)
    assert r.json()["combined_g"] == pytest.approx(expect["combined"], abs=0.01)


def test_facility_and_battery(client):
    f = client.post("/api/facility", json={"zone": ZONE, "flexible_pct": 50, "slack_hours": 24, "headroom_pct": 20, "week": 3})
    assert f.status_code == 200
    d = f.json()
    assert 0 < d["reduction_pct"] < d["ideal_reduction_pct"]
    assert len(d["week"]["ci"]) == 168 and max(d["week"]["draw"]) <= d["week"]["cap"] + 1e-6
    assert d["law"]["peak_draw"] == 13.0 and d["law"]["virtual_battery_hours"] == 12.0 and d["law"]["effective_horizon_hours"] == 60
    assert 0 < d["law"]["reduction_pct"] <= d["ideal_reduction_pct"] * 1.2
    assert all(c["law_reduction_pct"] is not None for c in d["curve"])
    assert d["law"]["battery_equivalent_kwh"] > 0 and d["law"]["battery_saves_nothing"] is False
    b = client.post("/api/battery", json={"zone": ZONE, "energy_minutes": 10, "reserve_pct": 0, "efficiency_pct": 90})
    assert b.status_code == 200 and 0 <= b.json()["reduction_pct"] < 2


@pytest.mark.parametrize("body", [
    {"zone": "NOPE", "job_hours": 24, "slack_hours": 24},                       # unknown zone
    {"zone": ZONE, "job_hours": 5, "slack_hours": 24},                          # not an allowed job length
    {"zone": ZONE, "job_hours": 24, "slack_hours": 100000},                     # huge slack
    {"zone": ZONE, "job_hours": 24, "slack_hours": 24, "extra": 1},             # unexpected field
    {"zone": "<script>alert(1)</script>", "job_hours": 24, "slack_hours": 24},  # injection attempt
    {"zone": ZONE, "job_hours": "24; drop", "slack_hours": 24},                 # wrong type
])
def test_rejects_bad_input(client, body):
    r = client.post("/api/temporal", json=body)
    assert r.status_code == 422
    assert r.json()["error"] == "invalid input"
    assert "<script>" not in r.text


def test_facility_bounds(client):
    for bad in ({"headroom_pct": 301}, {"headroom_pct": -1}, {"flexible_pct": 100}, {"slack_hours": 8760}, {"week": 52}):
        body = {"zone": ZONE, "flexible_pct": 50, "slack_hours": 24, "headroom_pct": 20, "week": 0} | bad
        assert client.post("/api/facility", json=body).status_code == 422


def test_body_size_limit(client):
    r = client.post("/api/temporal", content=b"{" + b" " * 10_000 + b"}", headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_untrusted_host_rejected():
    c = TestClient(app, base_url="http://evil.example")
    assert c.get("/api/health").status_code == 400


def test_rate_limit():
    app.fastapi.state.limiter = RateLimiter(per_minute=20)  # burst of 5
    c = TestClient(app, base_url=BASE)
    codes = [c.get("/api/health").status_code for _ in range(12)]
    assert 429 in codes and codes[0] == 200
    r = c.get("/api/health")
    assert r.status_code == 429 and r.headers["retry-after"]
    app.fastapi.state.limiter = RateLimiter(10_000)


def test_errors_do_not_leak(client, monkeypatch):
    def boom(*a):
        raise RuntimeError("secret internal path /etc/passwd")

    monkeypatch.setattr(server, "compute_battery", boom)
    r = client.post("/api/battery", json={"zone": ZONE, "energy_minutes": 10, "reserve_pct": 0, "efficiency_pct": 90})
    assert r.status_code == 500
    assert "secret" not in r.text and "Traceback" not in r.text


def test_docs_disabled_by_default(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_llm_endpoints(client, monkeypatch, tmp_path):
    missing = tmp_path / "none.csv"
    monkeypatch.setattr(server, "LLM_SUMMARY", missing)
    assert client.get("/api/llm").status_code == 404

    f = tmp_path / "s.csv"
    pd.DataFrame({"model": ["m"] * 3, "batch": [1, 4, 16], "tok_per_s": [100.0, 300.0, 800.0],
                  "gpu_power_w": [10.0, 15.0, 25.0], "sys_power_w": [30.0, 40.0, 60.0],
                  "all_power_w": [12.0, 18.0, 28.0], "gpu_j_per_tok": [0.1, 0.05, 0.03],
                  "all_j_per_tok": [0.12, 0.06, 0.035], "sys_j_per_tok": [0.3, 0.13, 0.075]}).to_csv(f, index=False)
    monkeypatch.setattr(server, "LLM_SUMMARY", f)
    assert len(client.get("/api/llm").json()["rows"]) == 3
    r = client.post("/api/llm/estimate", json={"model": "m", "batch": 4, "zone": ZONE, "million_tokens_per_day": 1})
    assert r.status_code == 200 and r.json()["chip_j_per_token"] == pytest.approx(0.06)
    assert client.post("/api/llm/estimate", json={"model": "x", "batch": 4, "zone": ZONE,
                                                  "million_tokens_per_day": 1}).status_code == 422


def test_index_has_no_inline_script(client):
    html = client.get("/").text
    assert '<script src="/static/app.js"' in html
    assert "<script>" not in html and "onclick" not in html and "style=" not in html


def test_capacity_endpoint(client):
    body = {"zone": ZONE, "utilization_pct": 70, "topology": "2N", "build": "grid_only", "flexible_pct": 50}
    r = client.post("/api/capacity", json=body)
    assert r.status_code == 200
    d = r.json()
    assert d["cap_x_mean"] == pytest.approx(2 / 0.7, rel=1e-3)
    assert 0 < d["reduction_pct"] <= d["ideal_reduction_pct"]
    assert d["value_kg_per_kw_year"] >= 0 and d["lifetime_years"] == 25
    for bad in ({"topology": "3N"}, {"build": "nuclear"}, {"utilization_pct": 99}, {"utilization_pct": 10}):
        assert client.post("/api/capacity", json=body | bad).status_code == 422


def test_siting_endpoint(client):
    r = client.get("/api/siting")
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        rows = r.json()["rows"]
        assert rows and {"zone", "temporal_kg_per_kw_yr", "spatial_100ms_kg_per_kw_yr"} <= rows[0].keys()


@pytest.fixture()
def fake_signal(monkeypatch):
    """Made-up hedge / battery tables for one fictitious balancing authority 'ZZZ'."""
    import numpy as np
    w = np.round(np.linspace(0, 1, 11), 1)
    hedge = pd.DataFrame({"ba": "ZZZ", "w_avg": w, "avg_share": w, "marg_share": 1 - w ** 2})
    oos = pd.DataFrame({"ba": "ZZZ", "w_avg": w, "marg_out_of_sample_g": 10 * (1 - w)})
    batt = pd.DataFrame({"ba": "ZZZ", "energy_h": [1 / 6, 1 / 6], "dispatch_on": ["average", "marginal"],
                         "avg_red_pct": [0.1, -0.3], "marg_red_pct": [-0.05, 1.0]})
    tables = {"hedge": hedge, "oos": oos, "batt": batt, "bas": frozenset({"ZZZ"}), "ba_list": ["ZZZ"]}
    day = pd.DataFrame({"ba": "ZZZ", "h": range(24), "average": 400.0, "marginal": 600.0}).set_index(["ba", "h"])
    monkeypatch.setattr(server, "signal_tables", lambda: tables)
    monkeypatch.setattr(server, "signal_days", lambda: day)
    server.compute_signal.cache_clear()
    yield
    server.compute_signal.cache_clear()


def test_signal_endpoint(client, fake_signal):
    assert client.get("/api/signal").json()["bas"] == [{"code": "ZZZ", "mean_average": 400.0, "mean_marginal": 600.0}]
    d = client.post("/api/signal", json={"ba": "ZZZ", "w_pct": 70}).json()
    assert d["selected"]["avg_share_pct"] == 70.0
    assert d["selected"]["marg_share_pct"] == pytest.approx(51.0)
    assert d["selected"]["marg_heldout_vs_marginal_only_pct"] == pytest.approx(30.0)
    assert len(d["w"]) == 11 and len(d["day"]["average"]) == 24
    assert {b["dispatch_on"] for b in d["battery"]} == {"average", "marginal"}
    for bad in ({"ba": "NOPE", "w_pct": 50}, {"ba": "ZZZ", "w_pct": 55}, {"ba": "ZZZ", "w_pct": 110},
                {"ba": "ZZZ", "w_pct": 50, "x": 1}, {"ba": "Z" * 40, "w_pct": 50}):
        assert client.post("/api/signal", json=bad).status_code == 422
