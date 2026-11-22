// Carbon Shift Explorer: plain JS, no dependencies. All data reaches the DOM via textContent.
"use strict";

const $ = (id) => document.getElementById(id);
const SVGNS = "http://www.w3.org/2000/svg";
const BATCHES = [1, 2, 4, 8, 16, 32, 64, 128];
const VOLUMES = [0.1, 0.3, 1, 3, 10, 30, 100];
let META = null;

// ------------------------------------------------------------------ utilities
function debounce(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

const inflight = {};
async function api(key, path, body) {
  if (inflight[key]) inflight[key].abort();
  const ctrl = new AbortController();
  inflight[key] = ctrl;
  const opts = { signal: ctrl.signal, headers: { "Accept": "application/json" } };
  if (body !== undefined) {
    opts.method = "POST";
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? "–"
  : Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
const hours = (h) => h >= 8760 ? "1 year" : h >= 48 && h % 24 === 0 ? `${h / 24} days` : `${h} h`;
const set = (id, text) => { $(id).textContent = text; };

function el(name, attrs = {}, parent) {
  const n = document.createElementNS(SVGNS, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}

function busy(panelId, on) { $(panelId).classList.toggle("busy", on); }

// ------------------------------------------------------------------ charts
function niceMax(v) {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}

/**
 * Line / area chart with a hover crosshair.
 * opts: {xs, series:[{ys, cls, name, area?, fmt?}], width, height, yMax?, xFmt, xTicks:[...],
 *        refY?:{value,label}, marker?:{x,y,cls}, logX?, stackBase?:{value, name}, yFmt}
 */
function lineChart(container, opts) {
  container.replaceChildren();
  const W = opts.width || 640, H = opts.height || 220, m = { l: 44, r: 12, t: 10, b: 26 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.label || "chart" }, container);
  const xs = opts.xs;
  const tx = opts.logX ? (v) => Math.log2(v) : (v) => v;
  const x0 = tx(xs[0]), x1 = tx(xs[xs.length - 1]);
  const X = (v) => m.l + (x1 === x0 ? iw / 2 : (tx(v) - x0) / (x1 - x0) * iw);
  const allY = opts.series.flatMap((s) => s.ys.filter((v) => v !== null));
  if (opts.refY) allY.push(opts.refY.value);
  const yMin = opts.yMin ?? 0;
  const yMax = opts.yMax ?? niceMax(Math.max(...allY, yMin + 1e-9));
  const Y = (v) => m.t + ih - (v - yMin) / (yMax - yMin) * ih;

  const ax = el("g", { class: "axis" }, svg);
  for (let i = 0; i <= 4; i++) {
    const v = yMin + (yMax - yMin) * i / 4, y = Y(v);
    el("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, class: "gridline" }, ax);
    const t = el("text", { x: m.l - 6, y: y + 3.5, "text-anchor": "end" }, ax);
    t.textContent = (opts.yFmt || ((v) => fmt(v, v < 10 && yMax <= 10 ? 1 : 0)))(v);
  }
  for (const v of opts.xTicks || []) {
    const t = el("text", { x: X(v), y: H - 8, "text-anchor": "middle" }, ax);
    t.textContent = (opts.xFmt || String)(v);
  }

  if (opts.stackBase) {
    el("rect", { x: m.l, y: Y(opts.stackBase.value), width: iw, height: Y(yMin) - Y(opts.stackBase.value), class: "fbase" }, svg);
  }
  for (const s of opts.series) {
    const pts = xs.map((x, i) => [X(x), s.ys[i] === null ? null : Y(s.ys[i])]).filter((p) => p[1] !== null);
    if (s.area) {
      const base = Y(s.areaBase ?? yMin);
      const d = `M${pts[0][0]},${base}` + pts.map((p) => `L${p[0]},${p[1]}`).join("") + `L${pts[pts.length - 1][0]},${base}Z`;
      el("path", { d, class: s.area, opacity: 0.85 }, svg);
    }
    el("path", { d: "M" + pts.map((p) => p.join(",")).join("L"), class: s.cls, "stroke-linejoin": "round" }, svg);
    if (s.dots) for (const p of pts) el("circle", { cx: p[0], cy: p[1], r: 3.5, class: `dot ${s.dots}` }, svg);
  }
  if (opts.refY) {
    el("line", { x1: m.l, x2: W - m.r, y1: Y(opts.refY.value), y2: Y(opts.refY.value), class: "ref" }, svg);
  }
  if (opts.marker) {
    el("circle", { cx: X(opts.marker.x), cy: Y(opts.marker.y), r: 6, class: `dot ${opts.marker.cls || "f1"}` }, svg);
  }

  // hover crosshair + tooltip
  const tip = document.createElement("div");
  tip.className = "tip"; tip.hidden = true; container.appendChild(tip);
  const cross = el("line", { y1: m.t, y2: m.t + ih, class: "cross", visibility: "hidden" }, svg);
  const hit = el("rect", { x: m.l, y: m.t, width: iw, height: ih, class: "hit" }, svg);
  const show = (evt) => {
    const r = svg.getBoundingClientRect();
    const px = (evt.clientX - r.left) / r.width * W;
    let best = 0, bd = Infinity;
    xs.forEach((x, i) => { const d = Math.abs(X(x) - px); if (d < bd) { bd = d; best = i; } });
    const cx = X(xs[best]);
    cross.setAttribute("x1", cx); cross.setAttribute("x2", cx); cross.setAttribute("visibility", "visible");
    const parts = [(opts.xFmt || String)(xs[best])];
    for (const s of opts.series) if (s.ys[best] !== null) parts.push(`${s.name}: ${(s.fmt || ((v) => fmt(v)))(s.ys[best])}`);
    tip.textContent = parts.join("  ·  ");
    tip.hidden = false;
    const top = Math.min(...opts.series.map((s) => s.ys[best] ?? yMax).map(Y));
    tip.style.left = `${Math.min(Math.max(cx / W * 100, 18), 82)}%`;
    tip.style.top = `${top / H * r.height}px`;
  };
  hit.addEventListener("pointermove", show);
  hit.addEventListener("pointerdown", show);
  hit.addEventListener("pointerleave", () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); });
}

function histogram(container, counts, edges, unit) {
  container.replaceChildren();
  const W = 640, H = 200, m = { l: 44, r: 12, t: 10, b: 26 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "histogram" }, container);
  const yMax = niceMax(Math.max(...counts, 1));
  const ax = el("g", { class: "axis" }, svg);
  for (let i = 0; i <= 4; i++) {
    const y = m.t + ih - ih * i / 4;
    el("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, class: "gridline" }, ax);
    el("text", { x: m.l - 6, y: y + 3.5, "text-anchor": "end" }, ax).textContent = fmt(yMax * i / 4, 0);
  }
  const bw = iw / counts.length;
  const tip = document.createElement("div"); tip.className = "tip"; tip.hidden = true; container.appendChild(tip);
  counts.forEach((c, i) => {
    const h = c / yMax * ih, x = m.l + i * bw;
    el("rect", { x: x + 1, y: m.t + ih - h, width: Math.max(bw - 2, 1), height: Math.max(h, 0), rx: 3, class: "f1" }, svg);
    const hit = el("rect", { x, y: m.t, width: bw, height: ih, class: "hit" }, svg);
    hit.addEventListener("pointerenter", () => {
      tip.textContent = `${fmt(edges[i], 0)}–${fmt(edges[i + 1], 0)} ${unit}: ${c} hours`;
      tip.hidden = false;
      tip.style.left = `${Math.min(Math.max((x + bw / 2) / W * 100, 18), 82)}%`;
      tip.style.top = `${(m.t + ih - h) / H * svg.getBoundingClientRect().height}px`;
    });
    hit.addEventListener("pointerleave", () => { tip.hidden = true; });
  });
  for (const i of [0, Math.floor(counts.length / 2), counts.length]) {
    el("text", { x: m.l + i * bw, y: H - 8, "text-anchor": "middle" }, ax).textContent = `${fmt(edges[i], 0)}`;
  }
  el("text", { x: W - m.r, y: m.t + 10, "text-anchor": "end", class: "lbl" }, svg).textContent = unit;
}

// ------------------------------------------------------------------ zones
function fillZones(meta) {
  const groups = {};
  for (const z of meta.zones) (groups[z.region] ||= []).push(z);
  for (const sel of document.querySelectorAll("select.zone")) {
    for (const region of Object.keys(groups).sort()) {
      const og = document.createElement("optgroup");
      og.label = region;
      for (const z of groups[region]) {
        const o = document.createElement("option");
        o.value = z.code;
        o.textContent = `${z.name} (${z.code}) · ${Math.round(z.mean_ci)} g/kWh`;
        og.appendChild(o);
      }
      sel.appendChild(og);
    }
    sel.value = meta.zones.some((z) => z.code === "US-CAL-CISO") ? "US-CAL-CISO" : meta.zones[0].code;
    sel.addEventListener("change", () => {
      for (const other of document.querySelectorAll("select.zone")) other.value = sel.value;
      refreshAll();
    });
  }
}
const zoneMean = (code) => META.zones.find((z) => z.code === code)?.mean_ci;

// ------------------------------------------------------------------ temporal
const runTemporal = debounce(async () => {
  const job = META.job_hours[+$("t-job").value], slack = META.slack_hours[+$("t-slack").value];
  set("t-job-o", hours(job)); set("t-slack-o", hours(slack));
  busy("t-out", true);
  try {
    const r = await api("t", "/api/temporal", { zone: $("t-zone").value, job_hours: job, slack_hours: slack });
    set("t-k-base", fmt(r.baseline_g_per_job_hour, 0));
    set("t-k-defer", `−${fmt(r.defer_pct)}%`); set("t-k-defer-s", `${fmt(r.defer_g)} g per job-hour`);
    set("t-k-comb", `−${fmt(r.combined_pct)}%`); set("t-k-comb-s", `${fmt(r.combined_g)} g per job-hour`);
    histogram($("t-hist"), r.histogram.counts, r.histogram.edges, "g saved / job-hour");
  } catch (e) { if (e.name !== "AbortError") set("t-k-base", "error"); }
  busy("t-out", false);
}, 150);

// ------------------------------------------------------------------ facility
const runFacility = debounce(async () => {
  const slack = META.facility_slack_hours[+$("f-slack").value];
  const flex = +$("f-flex").value, head = +$("f-head").value, week = +$("f-week").value;
  set("f-flex-o", `${flex}%`); set("f-slack-o", hours(slack)); set("f-head-o", head === 0 ? "none" : `+${head}%`);
  set("f-week-o", `week ${week + 1}`);
  busy("f-out", true);
  try {
    const r = await api("f", "/api/facility", { zone: $("f-zone").value, flexible_pct: flex, slack_hours: slack, headroom_pct: head, week });
    set("f-k-red", `−${fmt(r.reduction_pct)}%`);
    set("f-k-red-s", `ideal (no cap): −${fmt(r.ideal_reduction_pct)}%`);
    set("f-k-ret", r.retained_of_ideal_pct === null ? "–" : `${fmt(r.retained_of_ideal_pct, 0)}%`);
    set("f-k-ret-s", `with +${head}% headroom`);
    set("f-k-peak", `${fmt(r.uncapped_peak_draw)}×`);
    set("f-k-vb", `${fmt(r.law.virtual_battery_hours, 0)} h`);
    set("f-k-vb-s", r.law.effective_horizon_hours ? `lossless; plans ${hours(r.law.effective_horizon_hours)} ahead; law: −${fmt(r.law.reduction_pct)}%` : "lossless; no headroom to recharge");
    set("f-k-bx", r.law.battery_saves_nothing ? "none" : r.law.battery_equivalent_kwh == null ? "–" : `${fmt(r.law.battery_equivalent_kwh)} kWh`);
    set("f-k-bx-s", r.law.battery_saves_nothing ? "spread never covers a battery's losses here" : "per kW of load: 4 h Li-ion, 90% round trip");
    const cx = r.curve.map((c) => c.headroom_pct);
    lineChart($("f-curve"), {
      xs: cx, xTicks: [0, 50, 100, 150, 200, 300], xFmt: (v) => `+${v}%`, height: 200, label: "reduction vs headroom",
      series: [{ ys: r.curve.map((c) => c.reduction_pct), cls: "s1", name: "optimiser", dots: "f1", fmt: (v) => `${fmt(v)}%` },
               { ys: r.curve.map((c) => c.law_reduction_pct), cls: "s2", name: "closed-form law", fmt: (v) => `${fmt(v)}%` }],
      refY: { value: r.ideal_reduction_pct }, yFmt: (v) => `${fmt(v, 0)}%`,
      marker: { x: head, y: r.reduction_pct, cls: "f2" },
    });
    const hrs = r.week.ci.map((_, i) => i);
    const dayTicks = [0, 24, 48, 72, 96, 120, 144];
    const dayFmt = (h) => `day ${Math.floor(h / 24) + 1} ${String(h % 24).padStart(2, "0")}:00`;
    lineChart($("f-ci"), {
      xs: hrs, xTicks: dayTicks, xFmt: (h) => (h % 24 === 0 ? `d${h / 24 + 1}` : dayFmt(h)), height: 150, label: "grid carbon",
      series: [{ ys: r.week.ci, cls: "s2", name: "g/kWh", fmt: (v) => fmt(v, 0) }],
    });
    lineChart($("f-draw"), {
      xs: hrs, xTicks: dayTicks, xFmt: (h) => (h % 24 === 0 ? `d${h / 24 + 1}` : dayFmt(h)), height: 170, label: "facility power",
      yMax: Math.ceil(Math.max(r.week.cap, ...r.week.draw) * 1.05 / 0.4) * 0.4, stackBase: { value: r.week.base },
      series: [{ ys: r.week.draw, cls: "s1", area: "f1", areaBase: r.week.base, name: "draw (× avg)", fmt: (v) => `${fmt(v, 2)}×` }],
      refY: { value: r.week.cap }, yFmt: (v) => `${fmt(v, 1)}×`,
    });
  } catch (e) { if (e.name !== "AbortError") set("f-k-red", "error"); }
  busy("f-out", false);
}, 200);

// ------------------------------------------------------------------ battery
const runBattery = debounce(async () => {
  const min = +$("b-min").value, res = +$("b-res").value, eff = +$("b-eff").value;
  set("b-min-o", min < 60 ? `${min} min` : `${fmt(min / 60, min % 60 ? 1 : 0)} h`);
  set("b-res-o", `${res}%`); set("b-eff-o", `${eff}%`);
  try {
    const r = await api("b", "/api/battery", { zone: $("b-zone").value, energy_minutes: min, reserve_pct: res, efficiency_pct: eff });
    set("b-k-red", `−${fmt(r.reduction_pct, 2)}%`);
    set("b-k-ci", fmt(r.optimised_g_per_kwh, 1)); set("b-k-ci-s", `g/kWh (was ${fmt(r.baseline_g_per_kwh, 1)})`);
    set("b-k-cyc", fmt(r.full_cycles_per_year, 0));
  } catch (e) { if (e.name !== "AbortError") set("b-k-red", "error"); }
}, 200);

// ------------------------------------------------------------------ LLM
let LLM = null;
async function loadLLM() {
  try {
    LLM = await api("lm", "/api/llm");
  } catch (e) {
    $("l-err").hidden = false; set("l-err", `GPU measurements unavailable: ${e.message}`);
    return;
  }
  const models = [...new Set(LLM.rows.map((r) => r.model))];
  for (const mname of models) {
    const o = document.createElement("option"); o.value = mname; o.textContent = mname; $("l-model").appendChild(o);
  }
  const legend = $("l-legend");
  models.forEach((mname, i) => { const s = document.createElement("span"); s.className = `l${i + 1}`; s.textContent = mname; legend.appendChild(s); });
  const t = $("l-table");
  const head = t.insertRow();
  for (const h of ["Model", "Batch", "Tokens/s", "GPU W", "Chip W", "System W", "GPU mJ/tok", "Chip mJ/tok", "System mJ/tok"]) {
    const th = document.createElement("th"); th.textContent = h; head.appendChild(th);
  }
  for (const r of LLM.rows) {
    const tr = t.insertRow();
    for (const v of [r.model, r.batch, fmt(r.tok_per_s, 0), fmt(r.gpu_power_w), fmt(r.all_power_w), fmt(r.sys_power_w), fmt(r.gpu_j_per_tok * 1000, 1), fmt(r.all_j_per_tok * 1000, 1), fmt(r.sys_j_per_tok * 1000, 1)]) {
      tr.insertCell().textContent = String(v);
    }
  }
  runLLM();
}

const runLLM = debounce(async () => {
  if (!LLM) return;
  const model = $("l-model").value, batch = BATCHES[+$("l-batch").value], vol = VOLUMES[+$("l-vol").value];
  set("l-batch-o", String(batch)); set("l-vol-o", fmt(vol, vol < 1 ? 1 : 0));
  const models = [...new Set(LLM.rows.map((r) => r.model))];
  const xs = [...new Set(LLM.rows.map((r) => r.batch))].sort((a, b) => a - b);
  const series = models.map((mname, i) => ({
    ys: xs.map((b) => LLM.rows.find((r) => r.model === mname && r.batch === b)?.all_j_per_tok ?? null),
    cls: `s${i + 1}`, dots: `f${i + 1}`, name: mname.replace("-Instruct-4bit", ""), fmt: (v) => `${fmt(v * 1000, 0)} mJ`,
  }));
  try {
    const r = await api("l", "/api/llm/estimate", { model, batch, zone: $("l-zone").value, million_tokens_per_day: vol });
    set("l-k-tps", fmt(r.tokens_per_s, 0));
    set("l-k-j", `${fmt(r.chip_j_per_token * 1000, 0)} mJ`);
    set("l-k-g", `${fmt(r.g_co2_per_million_tokens, 1)} g`); set("l-k-g-s", `at ${fmt(r.zone_mean_ci, 0)} g/kWh`);
    const kg = r.kg_co2_per_day, kwh = r.kwh_per_day;
    set("l-k-day", kg < 1 ? `${fmt(kg * 1000, kg < 0.01 ? 1 : 0)} g` : `${fmt(kg, 1)} kg`);
    set("l-k-day-s", `CO₂ · ${kwh < 1 ? `${fmt(kwh * 1000, 0)} Wh` : `${fmt(kwh, 1)} kWh`}`);
    const mi = models.indexOf(model);
    lineChart($("l-chart"), {
      xs, logX: true, xTicks: xs, height: 220, label: "energy per token vs batch", series,
      yFmt: (v) => fmt(v, 2), marker: { x: batch, y: r.chip_j_per_token, cls: `f${mi + 1}` },
    });
  } catch (e) { if (e.name !== "AbortError") { $("l-err").hidden = false; set("l-err", e.message); } }
}, 150);

// ------------------------------------------------------------------ capacity value
const BUILD_NAMES = { grid_only: "grid-only electrical", backed: "fully backed electrical", gpu_servers: "GPU servers", cpu_servers: "CPU servers" };
const runCapacity = debounce(async () => {
  const util = +$("c-util").value, flex = +$("c-flex").value, topo = $("c-topo").value, build = $("c-build").value;
  set("c-util-o", `${util}%`); set("c-flex-o", `${flex}%`);
  busy("c-out", true);
  try {
    const r = await api("c", "/api/capacity", { zone: $("c-zone").value, utilization_pct: util, topology: topo, build, flexible_pct: flex });
    set("c-k-red", `−${fmt(r.reduction_pct)}%`);
    set("c-k-red-s", `headroom +${fmt(r.headroom_pct, 0)}% · ideal −${fmt(r.ideal_reduction_pct)}%`);
    set("c-k-val", fmt(r.value_kg_per_kw_year, 0));
    set("c-k-val-s", `kg CO₂ per year · grid swings ${fmt(r.daily_std_g_per_kwh, 0)} g/kWh a day`);
    set("c-k-pay", r.payback_years === null ? "never" : `${fmt(r.payback_years, r.payback_years < 1 ? 2 : 1)} y`);
    set("c-k-pay-s", `${fmt(r.embodied_kg_per_kw, 0)} kg/kW embodied · ${r.lifetime_years} y life`);
    $("c-k-pay").classList.toggle("good", r.pays_back_in_life);
    set("c-note", r.pays_back_in_life
      ? `Building ${BUILD_NAMES[build]} here repays its embodied carbon within its ${r.lifetime_years}-year life. The grid swings by about ${fmt(r.daily_std_g_per_kwh, 0)} g/kWh a day.`
      : `Building ${BUILD_NAMES[build]} here does not repay its embodied carbon within its ${r.lifetime_years}-year life. Use spare capacity you already own, or build electrical headroom instead.`);
  } catch (e) { if (e.name !== "AbortError") set("c-k-red", "error"); }
  busy("c-out", false);
}, 200);

// ------------------------------------------------------------------ siting
let SITING = null;
function hbars(container, rows, labelKey, series) {
  container.replaceChildren();
  const W = 640, rowH = 16, m = { l: 120, r: 46, t: 4, b: 4 };
  const H = m.t + m.b + rows.length * rowH * series.length + rows.length * 4;
  const max = Math.max(1, ...rows.flatMap((r) => series.map((s) => r[s.key] || 0)));
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "siting bars" }, container);
  rows.forEach((r, i) => {
    const y0 = m.t + i * (rowH * series.length + 4);
    el("text", { x: m.l - 6, y: y0 + rowH * series.length / 2 + 4, "text-anchor": "end", class: "lbl" }, svg).textContent = r[labelKey];
    series.forEach((s, k) => {
      const v = Math.max(0, r[s.key] || 0), w = v / max * (W - m.l - m.r);
      el("rect", { x: m.l, y: y0 + k * rowH + 2, width: Math.max(w, 1), height: rowH - 4, rx: 3, class: s.cls }, svg);
      el("text", { x: m.l + w + 4, y: y0 + k * rowH + rowH - 4, class: "lbl" }, svg).textContent = fmt(v, 0);
    });
  });
}

async function loadSiting() {
  try { SITING = await api("s", "/api/siting"); }
  catch (e) { $("s-err").hidden = false; set("s-err", `Siting results unavailable: ${e.message}`); return; }
  renderSiting();
}

function renderSiting() {
  if (!SITING) return;
  const lat = $("s-lat").value, key = `spatial_${lat}ms_kg_per_kw_yr`;
  const sortKey = $("s-sort").value === "spatial" ? key : "temporal_kg_per_kw_yr";
  const rows = [...SITING.rows].sort((a, b) => b[sortKey] - a[sortKey]).slice(0, 16)
    .map((r) => ({ ...r, label: `${r.name}`.slice(0, 20) }));
  hbars($("s-chart"), rows, "label", [{ key, cls: "f1" }, { key: "temporal_kg_per_kw_yr", cls: "f2" }]);
}

// ------------------------------------------------------------------ signal (average vs marginal)
let SIGNAL = null;
async function loadSignal() {
  try { SIGNAL = await api("gm", "/api/signal"); }
  catch (e) { $("g-err").hidden = false; set("g-err", `Signal results unavailable: ${e.message}`); return; }
  const sel = $("g-ba");
  for (const b of SIGNAL.bas) {
    const o = document.createElement("option");
    o.value = b.code; o.textContent = `${b.code} (avg ${b.mean_average}, marginal ${b.mean_marginal} g/kWh)`;
    sel.appendChild(o);
  }
  if (SIGNAL.bas.some((b) => b.code === "PJM")) sel.value = "PJM";
  runSignal();
}

const pct = (v) => (v === null ? "–" : `${fmt(v, 0)}%`);
const runSignal = debounce(async () => {
  if (!SIGNAL) return;
  const w = +$("g-w").value;
  set("g-w-o", `${w}%`);
  try {
    const r = await api("g", "/api/signal", { ba: $("g-ba").value, w_pct: w });
    $("g-err").hidden = true;
    set("g-k-a", pct(r.selected.avg_share_pct));
    set("g-k-m", pct(r.selected.marg_share_pct));
    set("g-k-o", pct(r.selected.marg_heldout_vs_marginal_only_pct));
    const xs = r.w.map((v) => Math.round(v * 100));
    const lo = Math.min(0, ...r.avg_share_pct, ...r.marg_share_pct);
    lineChart($("g-chart"), {
      xs, xTicks: [0, 20, 40, 60, 80, 100], xFmt: (v) => `${v}%`, yMin: lo < 0 ? Math.min(-100, Math.floor(lo / 100) * 100) : 0, yMax: 100,
      label: "share of each optimum kept versus weight on the average signal",
      series: [{ ys: r.avg_share_pct, cls: "s1", name: "average", dots: "f1", fmt: (v) => pct(v) },
               { ys: r.marg_share_pct, cls: "s2", name: "marginal", dots: "f2", fmt: (v) => pct(v) }],
      refY: { value: 0 }, marker: { x: w, y: r.selected.marg_share_pct, cls: "f2" },
    });
    lineChart($("g-day"), {
      xs: r.day.hour, xTicks: [0, 6, 12, 18, 23], xFmt: (h) => `${h}:00`, label: "average day of both signals",
      series: [{ ys: r.day.average, cls: "s1", name: "average", fmt: (v) => fmt(v, 0) },
               { ys: r.day.marginal, cls: "s2", name: "marginal", fmt: (v) => fmt(v, 0) }],
    });
    const t = $("g-batt");
    t.replaceChildren();
    const head = t.createTHead().insertRow();
    for (const c of ["Battery", "Dispatched on", "Average-ledger cut", "Marginal cut"]) {
      const th = document.createElement("th"); th.textContent = c; head.appendChild(th);
    }
    const body = t.createTBody();
    for (const b of r.battery) {
      const row = body.insertRow();
      const size = b.energy_minutes < 60 ? `${b.energy_minutes} min` : `${b.energy_minutes / 60} h`;
      for (const [v, neg] of [[size], [b.dispatch_on], [`${fmt(b.avg_red_pct, 2)}%`, b.avg_red_pct < 0], [`${fmt(b.marg_red_pct, 2)}%`, b.marg_red_pct < 0]]) {
        const td = row.insertCell(); td.textContent = v; if (neg) td.className = "neg";
      }
    }
  } catch (e) { if (e.name !== "AbortError") { $("g-err").hidden = false; set("g-err", `Error: ${e.message}`); } }
}, 150);

// ------------------------------------------------------------------ cooling penalty (F26)
let COOLING = null;
async function loadCooling() {
  try { COOLING = await api("km", "/api/cooling"); }
  catch (e) { $("k-err").hidden = false; set("k-err", `Cooling results unavailable: ${e.message}`); return; }
  const sel = $("k-zone");
  for (const z of COOLING.zones) {
    const o = document.createElement("option");
    o.value = z.code; o.textContent = `${z.name} (${z.code}) · ${z.hub}`;
    sel.appendChild(o);
  }
  if (COOLING.zones.some((z) => z.code === "SG")) sel.value = "SG";
  runCooling();
}

const runCooling = debounce(async () => {
  if (!COOLING) return;
  const flex = +$("k-flex").value, head = +$("k-head").value;
  const tank = COOLING.tank_hours[+$("k-tank").value];
  set("k-flex-o", `${flex}%`); set("k-head-o", `+${head}%`); set("k-tank-o", tank ? `${tank} h` : "none");
  busy("p-cooling", true);
  try {
    const r = await api("k", "/api/cooling", { zone: $("k-zone").value, plant: $("k-plant").value, flexible_pct: flex, headroom_pct: head, tank_hours: tank });
    $("k-err").hidden = true;
    set("k-k-claim", fmt(r.claimed_g, 1));
    set("k-k-claim-s", `g per IT kWh · carbon saving × PUE ${fmt(r.pue_mean, 2)}`);
    set("k-k-true", fmt(r.true_carbon_only_g, 1));
    set("k-k-true-s", "g per IT kWh, counting hourly cooling");
    set("k-k-pen", r.penalty_pct === null ? "–" : `${fmt(r.penalty_pct, 0)}%`);
    $("k-k-pen").classList.toggle("bad", (r.penalty_pct ?? 0) >= 10);
    set("k-k-pen-s", r.law_penalty_pct === null ? "of the claim" : `of the claim · law: ${fmt(r.law_penalty_pct, 0)}%`);
    set("k-k-aware", fmt(r.true_cooling_aware_g, 1));
    set("k-k-aware-s", r.cooling_aware_gain_pct === null ? "g per IT kWh, from weather alone"
      : `g per IT kWh · ${r.cooling_aware_gain_pct >= 0 ? "+" : ""}${fmt(r.cooling_aware_gain_pct, 0)}% vs carbon-only`);
    const all = [...r.day.ci_pct, ...r.day.pue_pct];
    const lo = Math.floor(Math.min(...all) / 10) * 10, hi = Math.ceil(Math.max(...all) / 10) * 10;
    lineChart($("k-day"), {
      xs: r.day.hour, xTicks: [0, 6, 12, 18, 23], xFmt: (h) => `${h}:00`, yMin: lo, yMax: Math.max(hi, lo + 10),
      label: "average day of grid carbon and PUE, each as a percent of its mean",
      series: [{ ys: r.day.ci_pct, cls: "s1", name: "carbon", fmt: (v) => `${fmt(v, 0)}%` },
               { ys: r.day.pue_pct, cls: "s2", name: "PUE", fmt: (v) => `${fmt(v, 1)}%` }],
      refY: { value: 100 },
    });
    $("k-tank-card").hidden = !r.tank;
    if (r.tank) {
      set("k-t-best", fmt(r.tank.saving_g, 2));
      set("k-t-best-s", `g per IT kWh · ${fmt(r.tank.saving_pct, 2)}% of facility emissions`);
      set("k-t-carb", fmt(r.tank.carbon_only_saving_g, 2));
      $("k-t-carb").classList.toggle("bad", r.tank.carbon_only_saving_g < 0);
      set("k-t-carb-s", r.tank.carbon_only_saving_g < 0 ? "g per IT kWh: raises emissions" : "g per IT kWh saved");
      set("k-t-batt", r.tank.battery_kwh_per_tank_kwh === null ? "n/a" : fmt(r.tank.battery_kwh_per_tank_kwh, 2));
      set("k-t-batt-s", r.tank.battery_kwh_per_tank_kwh === null
        ? "this grid's daily swing does not cover battery losses; the tank still saves"
        : r.tank.battery_saves_little ? "kWh of 4 h battery per kWh of tank: batteries gain little on this grid"
        : "kWh of 4 h battery per kWh of tank");
    }
    set("k-note", r.flat_signal
      ? `This grid's carbon signal is flat, so a carbon-only scheduler saves nothing. Moving work to cool hours still saves carbon, because each IT kWh then needs less cooling.`
      : `PUE ranges ${fmt(r.pue_min, 2)}–${fmt(r.pue_max, 2)} over the year at ${r.hub}. The closed-form law is −ρ·(mean carbon / its hourly spread)·(PUE spread / mean PUE): a small relative carbon swing makes a grid most exposed.`);
  } catch (e) { if (e.name !== "AbortError") { $("k-err").hidden = false; set("k-err", `Error: ${e.message}`); } }
  finally { busy("p-cooling", false); }
}, 150);

// ------------------------------------------------------------------ tabs, theme, boot
function setupTabs() {
  const tabs = [...document.querySelectorAll("nav.tabs button")];
  const select = (btn) => {
    for (const t of tabs) {
      const on = t === btn;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      $(t.getAttribute("aria-controls")).hidden = !on;
    }
    try { localStorage.setItem("tab", btn.id); } catch { /* storage unavailable */ }
  };
  tabs.forEach((t, i) => {
    t.addEventListener("click", () => select(t));
    t.addEventListener("keydown", (e) => {
      const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (d) { const n = tabs[(i + d + tabs.length) % tabs.length]; n.focus(); select(n); }
    });
  });
  let saved = null;
  try { saved = localStorage.getItem("tab"); } catch { /* ignore */ }
  select(tabs.find((t) => t.id === saved) || tabs[0]);
}

function setupTheme() {
  const root = document.documentElement, order = ["auto", "light", "dark"];
  let mode = "auto";
  try { mode = localStorage.getItem("theme") || "auto"; } catch { /* ignore */ }
  const apply = () => {
    if (mode === "auto") root.removeAttribute("data-theme"); else root.setAttribute("data-theme", mode);
    set("theme", `Theme: ${mode}`);
  };
  $("theme").addEventListener("click", () => {
    mode = order[(order.indexOf(mode) + 1) % order.length];
    try { localStorage.setItem("theme", mode); } catch { /* ignore */ }
    apply();
  });
  apply();
}

function refreshAll() { runTemporal(); runFacility(); runBattery(); runLLM(); runCapacity(); }

async function boot() {
  setupTheme();
  setupTabs();
  try {
    META = await api("meta", "/api/meta");
  } catch (e) {
    document.querySelector("header p").textContent = `Could not reach the API: ${e.message}`;
    return;
  }
  fillZones(META);
  $("t-job").max = META.job_hours.length - 1;
  $("t-slack").max = META.slack_hours.length - 1;
  $("f-slack").max = META.facility_slack_hours.length - 1;
  for (const id of ["t-job", "t-slack"]) $(id).addEventListener("input", runTemporal);
  for (const id of ["f-flex", "f-slack", "f-head", "f-week"]) $(id).addEventListener("input", runFacility);
  for (const id of ["b-min", "b-res", "b-eff"]) $(id).addEventListener("input", runBattery);
  for (const id of ["l-batch", "l-vol"]) $(id).addEventListener("input", runLLM);
  $("l-model").addEventListener("change", runLLM);
  for (const id of ["c-util", "c-flex"]) $(id).addEventListener("input", runCapacity);
  for (const id of ["c-topo", "c-build"]) $(id).addEventListener("change", runCapacity);
  for (const id of ["s-lat", "s-sort"]) $(id).addEventListener("change", renderSiting);
  $("g-w").addEventListener("input", runSignal);
  $("g-ba").addEventListener("change", runSignal);
  for (const id of ["k-flex", "k-head", "k-tank"]) $(id).addEventListener("input", runCooling);
  for (const id of ["k-zone", "k-plant"]) $(id).addEventListener("change", runCooling);
  runTemporal(); runFacility(); runBattery(); runCapacity();
  loadLLM(); loadSiting(); loadSignal(); loadCooling();
}

document.addEventListener("DOMContentLoaded", boot);
