"""Interactive explorer API + static UI for the carbonshift project.

Security posture (read-only compute service, no user data stored, no secrets):

* Every input is validated by a strict Pydantic model (``extra="forbid"``).
  Numeric knobs are bounded ranges or fixed sets, so one request can't
  trigger an arbitrarily large computation.
* A per-client token-bucket rate limit, a global concurrency cap, a 4 KiB
  request-body limit and result caching (LRU) bound resource usage.
* Strict security headers: CSP with no inline script or style and same-origin
  only, frame-ancestors none, nosniff, a restrictive Referrer-Policy and
  Permissions-Policy, and HSTS when served over HTTPS.
* TrustedHost allow-list. CORS is off unless origins are explicitly configured.
* Errors are generic JSON; no stack traces. OpenAPI docs are off unless enabled.
* LLM results are served from measurements taken offline: the API never runs
  model inference, which would let anyone with the URL use the GPU.

Configuration (environment variables):
  CARBONSHIFT_ALLOWED_HOSTS   comma list, default "127.0.0.1,localhost"
  CARBONSHIFT_CORS_ORIGINS    comma list of allowed origins, default none
  CARBONSHIFT_RATE_PER_MIN    requests per minute per client, default 120
  CARBONSHIFT_MAX_CONCURRENT  simultaneous computations, default 4
  CARBONSHIFT_ENABLE_DOCS     "1" to expose /docs and /openapi.json
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.types import ASGIApp, Receive, Scope, Send

from carbonshift.data import load_carbon, region_groups
from carbonshift.embodied import LIFETIME, per_kw, payback_years
from carbonshift.facility import battery_shift, capped_shift
from carbonshift.redundancy import installed_cap
from carbonshift.temporal import shift_one_zone
from carbonshift.waterfill import effective_horizon, saving_law

log = logging.getLogger("carbonshift.web")

ROOT = Path(__file__).resolve().parents[3]
STATIC = Path(__file__).resolve().parent / "static"
LLM_SUMMARY = ROOT / "results" / "llm_power_summary.csv"
LLM_META = ROOT / "results" / "llm_power_meta.json"
SITING = ROOT / "results" / "siting_per_region.csv"
VARIABILITY = ROOT / "results" / "variability_summary.csv"
HEDGE = ROOT / "results" / "hedge_frontier_per_ba.csv"
HEDGE_OOS = ROOT / "results" / "hedge_oos_per_ba.csv"
BATT_MARG = ROOT / "results" / "battery_marginal_per_ba.csv"

ALLOWED_HOSTS = [h.strip() for h in os.getenv("CARBONSHIFT_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",") if h.strip()]
CORS_ORIGINS = [o.strip() for o in os.getenv("CARBONSHIFT_CORS_ORIGINS", "").split(",") if o.strip()]
RATE_PER_MIN = int(os.getenv("CARBONSHIFT_RATE_PER_MIN", "120"))
MAX_CONCURRENT = int(os.getenv("CARBONSHIFT_MAX_CONCURRENT", "4"))
ENABLE_DOCS = os.getenv("CARBONSHIFT_ENABLE_DOCS", "0") == "1"
MAX_BODY_BYTES = 4096

JOB_HOURS = (1, 2, 4, 6, 12, 24, 48, 96, 168)
SLACK_HOURS = (1, 6, 12, 24, 48, 72, 168, 336, 720, 8760)
FACILITY_SLACK = (1, 6, 12, 24, 48, 72, 168)
CURVE_HEADROOM_PCT = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300)
YEAR = 2022


# --------------------------------------------------------------------------- data (loaded once)
@lru_cache(maxsize=1)
def carbon() -> pd.DataFrame:
    return load_carbon(YEAR)


@lru_cache(maxsize=1)
def zones_meta() -> list[dict]:
    ci = carbon()
    groups = region_groups()
    out = []
    for z in ci.columns:
        row = groups.loc[z] if z in groups.index else None
        name = str(row["zone_name"]) if row is not None else z
        region = next((g for g in ("Americas", "Europe", "Asia", "Oceania") if row is not None and row.get(g) == 1), "Other")
        s = ci[z]
        daily = s.groupby(s.index.date)
        out.append({"code": z, "name": name, "region": region, "mean_ci": round(float(s.mean()), 1),
                    "daily_cv": round(float((daily.std() / daily.mean()).mean()), 3)})
    return sorted(out, key=lambda d: d["name"])


ZONE_CODES = frozenset(load_carbon(YEAR).columns)


def _zone(v: str) -> str:
    if v not in ZONE_CODES:
        raise ValueError("unknown zone")
    return v


# --------------------------------------------------------------------------- request models
class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_max_length=64)


class ZoneIn(Strict):
    zone: str = Field(max_length=32)

    @field_validator("zone")
    @classmethod
    def _known_zone(cls, v: str) -> str:
        return _zone(v)


class TemporalIn(ZoneIn):
    job_hours: int
    slack_hours: int


    @field_validator("job_hours")
    @classmethod
    def _job(cls, v: int) -> int:
        if v not in JOB_HOURS:
            raise ValueError(f"job_hours must be one of {JOB_HOURS}")
        return v

    @field_validator("slack_hours")
    @classmethod
    def _slack(cls, v: int) -> int:
        if v not in SLACK_HOURS:
            raise ValueError(f"slack_hours must be one of {SLACK_HOURS}")
        return v


class FacilityIn(ZoneIn):
    flexible_pct: int = Field(ge=5, le=95)
    slack_hours: int
    headroom_pct: int = Field(ge=0, le=300)
    week: int = Field(default=26, ge=0, le=51)


    @field_validator("slack_hours")
    @classmethod
    def _slack(cls, v: int) -> int:
        if v not in FACILITY_SLACK:
            raise ValueError(f"slack_hours must be one of {FACILITY_SLACK}")
        return v


class BatteryIn(ZoneIn):
    energy_minutes: int = Field(ge=5, le=480)
    reserve_pct: int = Field(ge=0, le=90)
    efficiency_pct: int = Field(ge=50, le=100)



class LLMIn(ZoneIn):
    model: str = Field(max_length=64)
    batch: int = Field(ge=1, le=1024)
    million_tokens_per_day: float = Field(ge=0.001, le=1e6)



class SignalIn(Strict):
    ba: str = Field(max_length=16)
    w_pct: int = Field(ge=0, le=100, multiple_of=10)

    @field_validator("ba")
    @classmethod
    def _known_ba(cls, v: str) -> str:
        t = signal_tables()
        if t is None or v not in t["bas"]:
            raise ValueError("unknown balancing authority")
        return v


class CapacityIn(ZoneIn):
    utilization_pct: int = Field(ge=40, le=95)
    topology: Literal["design_only", "N+1 (N=4)", "2N"]
    build: Literal["grid_only", "backed", "gpu_servers", "cpu_servers"]
    flexible_pct: int = Field(default=50, ge=5, le=95)


# --------------------------------------------------------------------------- cached computations
def _r(x: float, n: int = 2) -> float:
    return round(float(x), n) if math.isfinite(x) else None  # type: ignore[return-value]


@lru_cache(maxsize=512)
def compute_temporal(zone: str, job: int, slack: int) -> dict:
    ci = carbon()[zone].to_numpy()
    r = shift_one_zone(ci, job, slack)
    pu = r.per_unit(job)
    base = float(r.baseline.mean() / job)
    # distribution of per-arrival savings for a small histogram
    combined = (r.baseline - r.interrupt) / job
    hist, edges = np.histogram(combined, bins=20)
    return {
        "baseline_g_per_job_hour": _r(base),
        "defer_g": _r(pu["defer"]), "interrupt_extra_g": _r(pu["interrupt_extra"]), "combined_g": _r(pu["combined"]),
        "defer_pct": _r(pu["defer"] / base * 100, 1), "combined_pct": _r(pu["combined"] / base * 100, 1),
        "histogram": {"counts": hist.tolist(), "edges": [_r(e, 1) for e in edges]},
    }


@lru_cache(maxsize=256)
def _capped(zone: str, phi: float, cap: float, slack: int):
    return capped_shift(carbon()[zone].to_numpy(), phi, cap, slack)


@lru_cache(maxsize=512)
def compute_facility(zone: str, flex_pct: int, slack: int, headroom_pct: int, week: int) -> dict:
    phi = flex_pct / 100
    cap = 1 + headroom_pct / 100
    r = _capped(zone, phi, cap, slack)
    ideal = _capped(zone, phi, math.inf, slack)
    ideal_saving = ideal.baseline_g - ideal.optimised_g
    retained = (r.baseline_g - r.optimised_g) / ideal_saving * 100 if ideal_saving > 1e-9 else None
    ci = carbon()[zone].to_numpy()
    curve = []
    for h in CURVE_HEADROOM_PCT:
        c = _capped(zone, phi, 1 + h / 100, slack)
        curve.append({"headroom_pct": h, "reduction_pct": _r(c.facility_reduction_pct),
                      "law_reduction_pct": _r(saving_law(ci, phi, 1 + h / 100, slack) / c.baseline_g * 100) if h > 0 else 0.0,
                      "retained_pct": _r((c.baseline_g - c.optimised_g) / ideal_saving * 100, 1) if ideal_saving > 1e-9 else None})
    sl = slice(week * 168, week * 168 + 168)
    return {
        "baseline_g_per_kwh": _r(r.baseline_g), "optimised_g_per_kwh": _r(r.optimised_g),
        "reduction_pct": _r(r.facility_reduction_pct), "ideal_reduction_pct": _r(ideal.facility_reduction_pct),
        "retained_of_ideal_pct": _r(retained, 1) if retained is not None else None,
        "saving_g_per_flex_kwh": _r(r.saving_per_flex_kwh),
        "uncapped_peak_draw": _r((1 - phi) + ideal.served.max()),
        "law": {"reduction_pct": _r(saving_law(ci, phi, cap, slack) / r.baseline_g * 100) if headroom_pct > 0 else 0.0,
                "peak_draw": _r(1 + slack * phi), "virtual_battery_hours": _r(slack * phi, 1),
                "effective_horizon_hours": effective_horizon(phi, cap, slack) if headroom_pct > 0 else None},
        "curve": curve,
        "week": {"ci": [_r(v, 1) for v in ci[sl]], "draw": [_r((1 - phi) + v, 3) for v in r.served[sl]],
                 "cap": _r(cap, 3), "base": _r(1 - phi, 3)},
    }


@lru_cache(maxsize=256)
def compute_battery(zone: str, minutes: int, reserve_pct: int, eff_pct: int) -> dict:
    b = battery_shift(carbon()[zone].to_numpy(), energy_h=minutes / 60, reserve_frac=reserve_pct / 100,
                      round_trip_eff=eff_pct / 100)
    usable_h = minutes / 60 * (1 - reserve_pct / 100)
    return {"reduction_pct": _r(b.reduction_pct, 3), "baseline_g_per_kwh": _r(b.baseline_g),
            "optimised_g_per_kwh": _r(b.optimised_g),
            "full_cycles_per_year": _r(b.throughput_kwh / usable_h, 0) if usable_h > 0 else 0}


@lru_cache(maxsize=1)
def swing_predictor() -> tuple[float, float] | None:
    """(k, exponent) of value ~ k * daily_std**exponent from scripts/variability_drivers.py, if available."""
    if not VARIABILITY.exists():
        return None
    v = pd.read_csv(VARIABILITY, index_col=0)["value"]
    if "loglog_constant_k" not in v:
        return None
    return float(v["loglog_constant_k"]), float(v["loglog_elasticity_shadow_vs_std"])


@lru_cache(maxsize=512)
def compute_capacity(zone: str, util_pct: int, topology: str, build: str, flex_pct: int) -> dict:
    ci = carbon()[zone]
    phi = flex_pct / 100
    cap = installed_cap(util_pct / 100, topology)
    r = capped_shift(ci.to_numpy(), phi, cap, 24)
    ideal = _capped(zone, phi, math.inf, 24)
    g = r.annual_marginal_saving  # g CO2 per kW-year
    emb = per_kw(build)[1]
    pay = payback_years(emb, g, 0.03)
    daily = ci.groupby(ci.index.date)
    std = float(daily.std().mean())
    pred = swing_predictor()
    return {
        "cap_x_mean": _r(cap, 3), "headroom_pct": _r((cap - 1) * 100, 1),
        "reduction_pct": _r(r.facility_reduction_pct), "ideal_reduction_pct": _r(ideal.facility_reduction_pct),
        "value_kg_per_kw_year": _r(g / 1000, 1),
        "predicted_at_20pct_headroom_kg_per_kw_year": _r(pred[0] * std ** pred[1], 1) if pred else None,  # predictor is fitted at +20% headroom only
        "daily_std_g_per_kwh": _r(std, 1),
        "embodied_kg_per_kw": _r(emb, 0), "lifetime_years": LIFETIME[build],
        "payback_years": _r(pay, 2) if math.isfinite(pay) else None,
        "pays_back_in_life": bool(math.isfinite(pay) and pay <= LIFETIME[build]),
    }


@lru_cache(maxsize=1)
def siting_table() -> pd.DataFrame | None:
    return pd.read_csv(SITING) if SITING.exists() else None


@lru_cache(maxsize=1)
def signal_tables() -> dict | None:
    """Precomputed F19/F20 results (scripts/hedge_experiments.py, scripts/battery_marginal_experiments.py)."""
    if not (HEDGE.exists() and HEDGE_OOS.exists() and BATT_MARG.exists()):
        return None
    hedge, oos, batt = pd.read_csv(HEDGE), pd.read_csv(HEDGE_OOS), pd.read_csv(BATT_MARG)
    bas = sorted(set(hedge.ba) & set(oos.ba) & set(batt.ba))
    return {"hedge": hedge, "oos": oos, "batt": batt, "bas": frozenset(bas), "ba_list": bas}


@lru_cache(maxsize=1)
def signal_days() -> pd.DataFrame:
    """Mean UTC day (24 h) of the average and marginal signal per BA, from the derived EIA-930 file."""
    from carbonshift.eia930 import signals
    s = signals()
    return s.assign(h=s.utc.dt.hour).groupby(["ba", "h"])[["average", "marginal"]].mean()


@lru_cache(maxsize=128)
def compute_signal(ba: str, w_pct: int) -> dict:
    t = signal_tables()
    h = t["hedge"][t["hedge"].ba == ba].sort_values("w_avg")
    o = t["oos"][t["oos"].ba == ba].sort_values("w_avg")
    w = w_pct / 100
    sel = h[np.isclose(h.w_avg, w)].iloc[0]
    o0 = float(o[np.isclose(o.w_avg, 0)].marg_out_of_sample_g.iloc[0])
    osel = o[np.isclose(o.w_avg, w)].iloc[0]
    day = signal_days().loc[ba]
    b = t["batt"][t["batt"].ba == ba]
    return {
        "w": [_r(x, 1) for x in h.w_avg],
        "avg_share_pct": [_r(x * 100, 1) for x in h.avg_share],
        "marg_share_pct": [_r(x * 100, 1) for x in h.marg_share],
        "selected": {"avg_share_pct": _r(sel.avg_share * 100, 1), "marg_share_pct": _r(sel.marg_share * 100, 1),
                     "marg_heldout_vs_marginal_only_pct": _r(osel.marg_out_of_sample_g / o0 * 100, 1) if abs(o0) > 1e-9 else None},
        "day": {"hour": list(range(24)), "average": [_r(x, 0) for x in day.average], "marginal": [_r(x, 0) for x in day.marginal]},
        "battery": [{"energy_minutes": int(round(r.energy_h * 60)), "dispatch_on": r.dispatch_on,
                     "avg_red_pct": _r(r.avg_red_pct, 2), "marg_red_pct": _r(r.marg_red_pct, 2)} for r in b.itertuples()],
    }


def llm_table() -> pd.DataFrame | None:
    """Measured summary; None until scripts/llm_power_batch.py has produced it with chip-level energy."""
    if not LLM_SUMMARY.exists():
        return None
    t = pd.read_csv(LLM_SUMMARY)
    if "all_j_per_tok" not in t.columns and "all_power_w" in t.columns:
        t["all_j_per_tok"] = t.all_power_w / t.tok_per_s
    return t


# --------------------------------------------------------------------------- middleware
class BodyLimit:
    """Reject request bodies over ``limit`` bytes, including chunked uploads without Content-Length."""

    def __init__(self, app: ASGIApp, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        for k, v in scope.get("headers", []):
            if k == b"content-length" and v.isdigit() and int(v) > self.limit:
                return await _send_json(send, 413, {"error": "request too large"})
        seen = 0

        async def limited() -> dict:
            nonlocal seen
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > self.limit:
                    raise _TooLarge()
            return msg

        try:
            await self.app(scope, limited, send)
        except _TooLarge:
            await _send_json(send, 413, {"error": "request too large"})


class _TooLarge(Exception):
    pass


async def _send_json(send: Send, status: int, body: dict) -> None:
    import json

    data = json.dumps(body).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(data)).encode())]})
    await send({"type": "http.response.body", "body": data})


class RateLimiter:
    """Token bucket per client key; bounded memory (oldest keys evicted)."""

    def __init__(self, per_minute: int, max_keys: int = 10_000):
        self.rate = per_minute / 60.0
        self.burst = max(5, per_minute // 4)
        self.max_keys = max_keys
        self._b: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._b.pop(key, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            ok = tokens >= 1
            self._b[key] = (tokens - 1 if ok else tokens, now)
            while len(self._b) > self.max_keys:
                self._b.popitem(last=False)
            return ok


SECURITY_HEADERS = {
    "Content-Security-Policy": ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                                "connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; "
                                "frame-ancestors 'none'; form-action 'none'"),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}


# --------------------------------------------------------------------------- app
def create_app() -> FastAPI:
    app = FastAPI(title="carbonshift explorer", version="0.1.0",
                  docs_url="/docs" if ENABLE_DOCS else None, redoc_url=None,
                  openapi_url="/openapi.json" if ENABLE_DOCS else None)
    limiter = RateLimiter(RATE_PER_MIN)
    gate = threading.BoundedSemaphore(MAX_CONCURRENT)
    app.state.limiter = limiter

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.url.path.startswith("/api/"):
            key = request.client.host if request.client else "unknown"
            if not app.state.limiter.allow(key):
                return JSONResponse({"error": "rate limit exceeded, slow down"}, status_code=429,
                                    headers={"Retry-After": "10"})
        try:
            response = await call_next(request)
        except Exception:  # never leak internals
            log.exception("unhandled error")
            response = JSONResponse({"error": "internal error"}, status_code=500)
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: RequestValidationError):
        msgs = [f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors()][:5]
        return JSONResponse({"error": "invalid input", "details": msgs}, status_code=422)

    @app.exception_handler(HTTPException)
    async def _http(_: Request, exc: HTTPException):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    def run(fn, *args):
        if not gate.acquire(timeout=10):
            raise HTTPException(503, "server busy, try again")
        try:
            return fn(*args)
        finally:
            gate.release()

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/meta")
    def meta():
        return {"year": YEAR, "zones": zones_meta(), "job_hours": JOB_HOURS, "slack_hours": SLACK_HOURS,
                "facility_slack_hours": FACILITY_SLACK, "global_mean_ci": round(float(carbon().mean().mean()), 2)}

    @app.post("/api/temporal")
    def temporal(body: TemporalIn):
        return run(compute_temporal, body.zone, body.job_hours, body.slack_hours)

    @app.post("/api/facility")
    def facility(body: FacilityIn):
        return run(compute_facility, body.zone, body.flexible_pct, body.slack_hours, body.headroom_pct, body.week)

    @app.post("/api/battery")
    def battery(body: BatteryIn):
        return run(compute_battery, body.zone, body.energy_minutes, body.reserve_pct, body.efficiency_pct)

    @app.post("/api/capacity")
    def capacity(body: CapacityIn):
        return run(compute_capacity, body.zone, body.utilization_pct, body.topology, body.build, body.flexible_pct)

    @app.get("/api/signal")
    def signal_meta():
        t = signal_tables()
        if t is None:
            raise HTTPException(404, "signal results not available yet")
        means = signal_days().groupby(level=0).mean()
        bas = [{"code": b, "mean_average": _r(means.loc[b, "average"], 0), "mean_marginal": _r(means.loc[b, "marginal"], 0)} for b in t["ba_list"]]
        return {"bas": bas, "weights_pct": list(range(0, 101, 10))}

    @app.post("/api/signal")
    def signal(body: SignalIn):
        return run(compute_signal, body.ba, body.w_pct)

    @app.get("/api/siting")
    def siting():
        t = siting_table()
        if t is None:
            raise HTTPException(404, "siting results not available yet")
        meta = {z["code"]: z["name"] for z in zones_meta()}
        rows = t.round(1).assign(name=t.zone.map(meta).fillna(t.zone)).to_dict(orient="records")
        return {"rows": rows, "latencies_ms": [50, 100, 250]}

    @app.get("/api/llm")
    def llm():
        t = llm_table()
        if t is None:
            raise HTTPException(404, "GPU measurements not available yet")
        cols = ["model", "batch", "tok_per_s", "gpu_power_w", "all_power_w", "sys_power_w",
                "gpu_j_per_tok", "all_j_per_tok", "sys_j_per_tok"]
        return {"rows": t[cols].round(4).to_dict(orient="records")}

    @app.post("/api/llm/estimate")
    def llm_estimate(body: LLMIn):
        t = llm_table()
        if t is None:
            raise HTTPException(404, "GPU measurements not available yet")
        d = t[t.model == body.model].sort_values("batch")
        if d.empty:
            raise HTTPException(422, "unknown model")
        b = float(np.clip(body.batch, d.batch.min(), d.batch.max()))
        # interpolate in log2(batch), the axis the measurements were taken on
        x = np.log2(d.batch.to_numpy(float))
        j_tok = float(np.interp(np.log2(b), x, d.all_j_per_tok))
        tps = float(np.interp(np.log2(b), x, d.tok_per_s))
        ci = float(carbon()[body.zone].mean())
        kwh_day = j_tok * body.million_tokens_per_day * 1e6 / 3.6e6
        return {"batch_used": b, "chip_j_per_token": _r(j_tok, 4), "tokens_per_s": _r(tps, 1),
                "g_co2_per_million_tokens": _r(j_tok * 1e6 / 3.6e6 * ci, 2),
                "kwh_per_day": _r(kwh_day, 6), "kg_co2_per_day": _r(kwh_day * ci / 1000, 6),
                "zone_mean_ci": _r(ci, 1)}

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
    if CORS_ORIGINS:
        app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"],
                           allow_headers=["Content-Type"], allow_credentials=False, max_age=600)
    return BodyLimitApp(app)


class BodyLimitApp:
    """Outermost ASGI wrapper: body-size guard around the FastAPI app (keeps ``.app`` for tests)."""

    def __init__(self, app: FastAPI):
        self.fastapi = app
        self._asgi = BodyLimit(app, MAX_BODY_BYTES)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self._asgi(scope, receive, send)


app = create_app()


def main() -> None:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="Run the carbonshift explorer")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run("carbonshift.web.server:app", host=args.host, port=args.port, proxy_headers=False,
                server_header=False, log_level="info")


if __name__ == "__main__":
    main()
