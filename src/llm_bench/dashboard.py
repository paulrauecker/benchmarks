"""Render a log directory as a self-contained interactive HTML dashboard.

Same data as `llm-bench report`, but plotted: scores with 95% CI bars so overlap
is visible at a glance, and cost against score so a "win" that costs 20x is
obvious. No network access at view time -- everything is inlined.
"""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path

from rich.console import Console

from llm_bench.report import Cell, collect

console = Console()

# dataviz reference palette, fixed slot order (never cycled by rank).
SERIES_LIGHT = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
SERIES_DARK = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]


def _payload(cells: list[Cell], superseded: list | None = None) -> dict:
    models: list[str] = []
    tasks: list[str] = []
    for c in cells:
        if c.model not in models:
            models.append(c.model)
        if c.task not in tasks:
            tasks.append(c.task)

    rows = [
        {
            "model": c.model,
            "task": c.task,
            "score": c.score,
            "stderr": c.stderr,
            "cost": c.cost,
            "duration": c.duration_s,
            "nScored": c.n_scored,
            "nTotal": c.n_total,
            "errors": int(round(c.error_rate * c.n_total)),
            "ownsTotals": c.owns_run_totals,
        }
        for c in cells
    ]
    return {
        "models": models,
        "tasks": tasks,
        "rows": rows,
        "superseded": [
            {"model": r.model, "task": r.task, "name": r.name, "startedAt": r.started_at}
            for r in (superseded or [])
        ],
    }


def _html(data: dict, title: str) -> str:
    payload = json.dumps(data)
    light = json.dumps(SERIES_LIGHT)
    dark = json.dumps(SERIES_DARK)
    return _TEMPLATE.replace("__TITLE__", title).replace("__DATA__", payload).replace(
        "__SERIES_LIGHT__", light
    ).replace("__SERIES_DARK__", dark)


def dashboard(log_dir: Path, output: Path | None = None, open_browser: bool = True) -> Path | None:
    collection = collect(log_dir)
    if not collection.cells:
        console.print(f"[red]no completed eval logs found under {log_dir}[/red]")
        return None

    out = output or (log_dir / "dashboard.html")
    payload = _payload(collection.cells, collection.superseded)
    out.write_text(_html(payload, title=log_dir.name), encoding="utf-8")
    console.print(f"[green]wrote[/green] {out}")
    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return out


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>llm-bench &middot; __TITLE__</title>
<style>
  :root {
    color-scheme: light;
    --page: #f9f9f7;
    --surface: #fcfcfb;
    --ink: #0b0b0b;
    --ink-2: #52514e;
    --muted: #898781;
    --grid: #e1e0d9;
    --axis: #c3c2b7;
    --border: rgba(11,11,11,0.10);
    --warning: #fab219;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --page: #0d0d0d;
      --surface: #1a1a19;
      --ink: #ffffff;
      --ink-2: #c3c2b7;
      --muted: #898781;
      --grid: #2c2c2a;
      --axis: #383835;
      --border: rgba(255,255,255,0.10);
      --warning: #fab219;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --page: #0d0d0d;
    --surface: #1a1a19;
    --ink: #ffffff;
    --ink-2: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --axis: #383835;
    --border: rgba(255,255,255,0.10);
    --warning: #fab219;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--page);
    color: var(--ink);
    font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  main { max-width: 1100px; margin: 0 auto; padding: 32px 20px 64px; }
  header { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; margin-bottom: 4px; }
  h1 { font-size: 20px; margin: 0; letter-spacing: -0.01em; }
  h2 { font-size: 15px; margin: 0 0 2px; }
  .sub { color: var(--muted); font-size: 13px; }
  .spacer { flex: 1; }
  button {
    font: inherit; color: var(--ink-2); background: var(--surface);
    border: 1px solid var(--border); border-radius: 8px; padding: 5px 11px; cursor: pointer;
  }
  button:hover { color: var(--ink); }
  .controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin: 20px 0 8px; }
  select { font: inherit; color: var(--ink); background: var(--surface);
    border: 1px solid var(--border); border-radius: 8px; padding: 5px 9px; }
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 18px 18px 10px; margin-top: 16px;
  }
  .cardhead { display: flex; align-items: center; gap: 12px; }
  .toggle { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .toggle button { border: 0; border-radius: 0; background: transparent; padding: 4px 10px; font-size: 13px; }
  .toggle button.on { background: var(--grid); color: var(--ink); }
  .badge {
    display: inline-flex; align-items: center; gap: 7px; margin-top: 10px;
    border: 1px solid var(--warning); border-radius: 8px; padding: 5px 10px;
    color: var(--ink-2); font-size: 12.5px;
  }
  .badge b { color: var(--ink); font-weight: 600; }
  .badge.info { border-color: var(--axis); }
  .legend { display: flex; flex-wrap: wrap; gap: 6px 16px; margin: 10px 2px 0; }
  .legend span { display: inline-flex; align-items: center; gap: 6px; color: var(--ink-2); font-size: 13px; }
  .swatch { width: 10px; height: 10px; border-radius: 3px; }
  .scroll { overflow-x: auto; }
  svg { display: block; }
  table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
  th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--ink-2); font-weight: 600; cursor: pointer; user-select: none; }
  th:hover { color: var(--ink); }
  td { color: var(--ink-2); }
  td:first-child { color: var(--ink); }
  #tip {
    position: fixed; pointer-events: none; opacity: 0; transition: opacity .08s;
    background: var(--surface); color: var(--ink); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 10px; font-size: 12.5px; line-height: 1.45;
    box-shadow: 0 6px 20px rgba(0,0,0,.16); z-index: 10; max-width: 260px;
  }
  #tip b { font-weight: 600; }
  .note { color: var(--muted); font-size: 12.5px; margin: 10px 2px 0; }
</style>
</head>
<body>
<main>
  <header>
    <h1>llm-bench &middot; __TITLE__</h1>
    <div class="spacer"></div>
    <button id="theme">Theme</button>
  </header>
  <div class="sub">Error bars are 95% confidence intervals. Overlapping intervals mean the difference is not resolvable at this sample size.</div>
  <div id="supersededNote"></div>

  <div class="controls">
    <label class="sub" for="taskSel">Task</label>
    <select id="taskSel"></select>
  </div>

  <section class="card">
    <h2>Score by model</h2>
    <div class="sub" id="scoreSub"></div>
    <div id="scoreWarn"></div>
    <div class="scroll"><svg id="bars"></svg></div>
    <div class="legend" id="legend"></div>
  </section>

  <section class="card">
    <div class="cardhead">
      <h2 id="scatterTitle">Cost vs. score</h2>
      <div class="spacer"></div>
      <div class="toggle" id="xMode">
        <button data-mode="total" class="on">total</button><button data-mode="per">per sample</button>
      </div>
    </div>
    <div class="sub" id="scatterSub"></div>
    <div class="scroll"><svg id="scatter"></svg></div>
  </section>

  <section class="card">
    <h2>All results</h2>
    <div class="sub">Click a column to sort.</div>
    <div class="scroll"><table id="table"></table></div>
    <div class="note">"errs" counts unscored samples (timeouts, parse failures, refusals). Errors, cost and time
      belong to the eval run rather than the scorer, so when a task has several scorers they are shown
      once and dashed out on the task's other rows.</div>
  </section>
</main>
<div id="tip"></div>

<script>
const DATA = __DATA__;
const LIGHT = __SERIES_LIGHT__;
const DARK = __SERIES_DARK__;
const NS = "http://www.w3.org/2000/svg";

const isDark = () => {
  const t = document.documentElement.dataset.theme;
  if (t) return t === "dark";
  return matchMedia("(prefers-color-scheme: dark)").matches;
};
const colorOf = (model) => {
  const i = DATA.models.indexOf(model);
  return (isDark() ? DARK : LIGHT)[i % 8];
};
const css = (name) => getComputedStyle(document.body).getPropertyValue(name).trim();

const el = (tag, attrs, text) => {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (text != null) n.textContent = text;
  return n;
};

const tip = document.getElementById("tip");
function bindTip(node, html) {
  node.addEventListener("mousemove", (e) => {
    tip.innerHTML = html;
    tip.style.opacity = 1;
    const pad = 14;
    let x = e.clientX + pad, y = e.clientY + pad;
    const r = tip.getBoundingClientRect();
    if (x + r.width > innerWidth - 8) x = e.clientX - r.width - pad;
    if (y + r.height > innerHeight - 8) y = e.clientY - r.height - pad;
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  });
  node.addEventListener("mouseleave", () => { tip.style.opacity = 0; });
}

const fmt = (v, d = 3) => (v == null ? "\\u2014" : v.toFixed(d));

// Model specs are provider-qualified ("openai-api/llamacpp/Qwen3-8B"); the last
// segment is what distinguishes them on an axis. Full spec stays in the tooltip.
const shortName = (m) => m.split("/").pop();
const textWidth = (s, px) => s.length * px * 0.58;
const COST_MODE = DATA.rows.some((r) => r.cost > 0);
// Below this, a task is a smoke test: the CI is wider than most differences
// anyone would read into the number.
const LOW_N = 30;
const nLabel = (r) => `n=${r.nScored}` + (r.nScored < LOW_N ? " low n" : "");

function drawBars(task) {
  const svg = document.getElementById("bars");
  svg.textContent = "";
  const rows = DATA.rows
    .filter((r) => r.task === task && r.score != null)
    .sort((a, b) => b.score - a.score);
  document.getElementById("scoreSub").textContent = task;
  if (!rows.length) return;

  const rowH = 34, top = 12, bottom = 34;
  const left = Math.min(260, Math.max(90, ...rows.map((r) => textWidth(shortName(r.model), 13) + 14)));
  const nW = Math.max(...rows.map((r) => textWidth(nLabel(r), 12)));
  const right = 60 + nW;
  const w = Math.max(560, Math.min(1040, svg.parentElement.clientWidth || 900));
  const h = top + rows.length * rowH + bottom;
  svg.setAttribute("width", w);
  svg.setAttribute("height", h);
  const plotW = w - left - right;

  const hi = Math.max(...rows.map((r) => (r.score + 1.96 * (r.stderr || 0))));
  const max = Math.min(1, Math.max(0.2, Math.ceil(hi / 0.2) * 0.2));
  const x = (v) => left + (v / max) * plotW;

  const ticks = Math.round(max / 0.2);
  for (let i = 0; i <= ticks; i++) {
    const v = (max * i) / ticks;
    svg.appendChild(el("line", { x1: x(v), x2: x(v), y1: top, y2: h - bottom,
      stroke: i === 0 ? css("--axis") : css("--grid"), "stroke-width": 1 }));
    svg.appendChild(el("text", { x: x(v), y: h - bottom + 18, fill: css("--muted"),
      "font-size": 12, "text-anchor": "middle" }, v.toFixed(1)));
  }

  rows.forEach((r, i) => {
    const cy = top + i * rowH + rowH / 2;
    const bh = 16;
    const bar = el("rect", { x: left, y: cy - bh / 2, width: Math.max(2, x(r.score) - left),
      height: bh, rx: 4, fill: colorOf(r.model) });
    svg.appendChild(bar);

    if (r.stderr != null) {
      const lo = x(Math.max(0, r.score - 1.96 * r.stderr));
      const up = x(Math.min(max, r.score + 1.96 * r.stderr));
      const ink = css("--ink-2");
      svg.appendChild(el("line", { x1: lo, x2: up, y1: cy, y2: cy, stroke: ink, "stroke-width": 2 }));
      for (const px of [lo, up]) {
        svg.appendChild(el("line", { x1: px, x2: px, y1: cy - 5, y2: cy + 5, stroke: ink, "stroke-width": 2 }));
      }
    }

    svg.appendChild(el("text", { x: left - 10, y: cy + 4, fill: css("--ink"),
      "font-size": 13, "text-anchor": "end" }, shortName(r.model)));
    svg.appendChild(el("text", { x: w - nW - 18, y: cy + 4, fill: css("--ink-2"),
      "font-size": 12.5, "text-anchor": "end" }, fmt(r.score)));
    svg.appendChild(el("text", { x: w - 6, y: cy + 4,
      fill: r.nScored < LOW_N ? css("--ink-2") : css("--muted"),
      "font-size": 12, "text-anchor": "end" }, nLabel(r)));

    const hit = el("rect", { x: left, y: cy - rowH / 2, width: plotW, height: rowH, fill: "transparent" });
    bindTip(hit, `<b>${r.model}</b><br>${r.task}<br>score ${fmt(r.score)}` +
      (r.stderr != null ? ` &plusmn; ${fmt(r.stderr)}` : "") +
      `<br>${r.nScored}/${r.nTotal} scored &middot; $${r.cost.toFixed(2)}`);
    svg.appendChild(hit);
  });

  const counts = [...new Set(rows.map((r) => r.nScored))].sort((a, b) => b - a);
  const warn = document.getElementById("scoreWarn");
  warn.innerHTML = "";
  if (counts.length > 1) {
    warn.innerHTML = `<div class="badge">&#9888;&#65039; <span><b>Sample counts differ across models</b>` +
      ` (${counts.join(" vs ")}). These bars are not the same measurement &mdash;` +
      ` a shorter run is a different, noisier benchmark, not a worse score.</span></div>`;
  } else if (counts[0] < LOW_N) {
    warn.innerHTML = `<div class="badge">&#9888;&#65039; <span><b>Only ${counts[0]} samples</b>` +
      ` &mdash; indicative at best. Differences this size are almost always noise.</span></div>`;
  }

  const legend = document.getElementById("legend");
  legend.innerHTML = "";
  if (DATA.models.length > 1) {
    rows.forEach((r) => {
      const s = document.createElement("span");
      s.innerHTML = `<i class="swatch" style="background:${colorOf(r.model)}"></i>${r.model}`;
      legend.appendChild(s);
    });
  }
}

let PER_SAMPLE = false;
function drawScatter() {
  const svg = document.getElementById("scatter");
  svg.textContent = "";
  document.getElementById("scatterTitle").textContent =
    COST_MODE ? "Cost vs. score" : "Time vs. score";
  // Summing over each model's own rows would compare a model that ran three
  // tasks against one that ran two, so aggregate only over the tasks every
  // model has a score for -- and say which those are.
  const scored = DATA.rows.filter((r) => r.score != null);
  const perModel = new Map();
  for (const r of scored) {
    if (!perModel.has(r.model)) perModel.set(r.model, new Set());
    perModel.get(r.model).add(r.task);
  }
  const common = DATA.tasks.filter((t) => [...perModel.values()].every((s) => s.has(t)));
  const dropped = DATA.tasks.filter((t) => !common.includes(t));
  const basis = common.length ? common : DATA.tasks;

  const agg = new Map();
  for (const r of scored) {
    if (!basis.includes(r.task)) continue;
    const a = agg.get(r.model) || { model: r.model, cost: 0, time: 0, samples: 0, sum: 0, n: 0 };
    if (r.ownsTotals) {
      a.cost += r.cost;
      a.time += r.duration;
      a.samples += r.nScored;
    }
    a.sum += r.score;
    a.n += 1;
    agg.set(r.model, a);
  }
  const pts = [...agg.values()].map((a) => {
    const total = COST_MODE ? a.cost : a.time;
    return {
      model: a.model, cost: a.cost, time: a.time, samples: a.samples,
      score: a.sum / a.n, n: a.n,
      x: PER_SAMPLE ? (a.samples ? total / a.samples : 0) : total,
    };
  });

  // A model that ran a task at --limit 3 spends a fraction of what the same
  // task at full size costs, so totals only compare when the sample counts
  // match; per-sample divides that out.
  const sampleCounts = [...new Set(pts.map((p) => p.samples))];
  const unit = COST_MODE ? "cost" : "wall-clock time";
  const measure = PER_SAMPLE ? `Per-sample ${unit}` : `Total ${unit}`;
  let sub = `${measure} per model over the ${basis.length} task${basis.length > 1 ? "s" : ""} ` +
    `every model ran (${basis.map(shortName).join(", ")}). Upper-left is better.`;
  if (!COST_MODE) sub += " No priced API calls in this run, so time stands in for cost.";
  if (!common.length && DATA.models.length > 1) {
    sub = `${measure} per model over each model's own tasks \\u2014 no task was run by every ` +
      `model, so these are not directly comparable.`;
  } else if (dropped.length) {
    sub += ` Excluded, not run by every model: ${dropped.map(shortName).join(", ")}.`;
  }
  if (!PER_SAMPLE && sampleCounts.length > 1) {
    sub += ` Models ran different sample counts (${sampleCounts.join(" vs ")}),` +
      ` so switch to "per sample" to compare these.`;
  }
  document.getElementById("scatterSub").textContent = sub;
  if (!pts.length) return;

  const w = Math.max(560, Math.min(1040, svg.parentElement.clientWidth || 900));
  const h = 320, top = 16, bottom = 44, left = 52, right = 120;
  svg.setAttribute("width", w);
  svg.setAttribute("height", h);

  const xFmt = COST_MODE
    ? (v) => "$" + v.toFixed(PER_SAMPLE ? 4 : 2)
    : (v) => v.toFixed(PER_SAMPLE ? 1 : 0) + "s";
  const maxX = Math.max(COST_MODE ? (PER_SAMPLE ? 0.0001 : 0.01) : 1, ...pts.map((p) => p.x)) * 1.15;
  const x = (v) => left + (v / maxX) * (w - left - right);
  const y = (v) => h - bottom - v * (h - top - bottom);

  for (let i = 0; i <= 5; i++) {
    const v = i / 5;
    svg.appendChild(el("line", { x1: left, x2: w - right, y1: y(v), y2: y(v),
      stroke: i === 0 ? css("--axis") : css("--grid"), "stroke-width": 1 }));
    svg.appendChild(el("text", { x: left - 10, y: y(v) + 4, fill: css("--muted"),
      "font-size": 12, "text-anchor": "end" }, v.toFixed(1)));
  }
  for (let i = 0; i <= 4; i++) {
    const v = (maxX * i) / 4;
    svg.appendChild(el("text", { x: x(v), y: h - bottom + 18, fill: css("--muted"),
      "font-size": 12, "text-anchor": "middle" }, xFmt(v)));
  }
  const xAxisLabel = (COST_MODE ? "cost" : "wall-clock time") +
    (PER_SAMPLE ? " per scored sample" : " across tasks");
  svg.appendChild(el("text", { x: (left + w - right) / 2, y: h - 6, fill: css("--muted"),
    "font-size": 12, "text-anchor": "middle" }, xAxisLabel));
  svg.appendChild(el("text", { x: 14, y: (top + h - bottom) / 2, fill: css("--muted"),
    "font-size": 12, "text-anchor": "middle",
    transform: `rotate(-90 14 ${(top + h - bottom) / 2})` }, "mean score"));

  const placed = [];
  pts.forEach((p) => {
    const g = el("g", {});
    const label = shortName(p.model);
    const lw = textWidth(label, 12.5);
    const flip = x(p.x) + 12 + lw > w - 6;
    const lx = x(p.x) + (flip ? -12 : 12);
    const box = { x0: flip ? lx - lw : lx, x1: flip ? lx : lx + lw, y: y(p.score) + 4 };
    for (let tries = 0; tries < 8; tries++) {
      const hit = placed.some((q) => Math.abs(q.y - box.y) < 14 && box.x0 < q.x1 && q.x0 < box.x1);
      if (!hit) break;
      box.y += tries % 2 ? -(tries + 1) * 8 : (tries + 1) * 8;
    }
    placed.push(box);
    g.appendChild(el("circle", { cx: x(p.x), cy: y(p.score), r: 6,
      fill: colorOf(p.model), stroke: css("--surface"), "stroke-width": 2 }));
    g.appendChild(el("text", { x: lx, y: box.y, fill: css("--ink"),
      "font-size": 12.5, "text-anchor": flip ? "end" : "start" }, label));
    bindTip(g, `<b>${p.model}</b><br>mean score ${fmt(p.score)} over ${p.n} task${p.n > 1 ? "s" : ""}` +
      `<br>${p.samples} scored samples` +
      `<br>total $${p.cost.toFixed(2)} &middot; ${p.time.toFixed(0)}s` +
      (p.samples ? `<br>per sample $${(p.cost / p.samples).toFixed(4)} &middot; ` +
        `${(p.time / p.samples).toFixed(1)}s` : ""));
    svg.appendChild(g);
  });
}

let sortKey = "model", sortAsc = true;
function drawTable() {
  const t = document.getElementById("table");
  t.innerHTML = "";
  const cols = [
    ["model", "model"], ["task", "task"], ["nScored", "n"], ["score", "score"],
    ["stderr", "\\u00b1 stderr"], ["errors", "errs"], ["cost", "cost $"], ["duration", "time s"],
  ];
  const head = document.createElement("tr");
  cols.forEach(([k, label]) => {
    const th = document.createElement("th");
    th.textContent = label + (sortKey === k ? (sortAsc ? " \\u2191" : " \\u2193") : "");
    th.onclick = () => {
      if (sortKey === k) sortAsc = !sortAsc; else { sortKey = k; sortAsc = true; }
      drawTable();
    };
    head.appendChild(th);
  });
  t.appendChild(head);

  const rows = [...DATA.rows].sort((a, b) => {
    const x = a[sortKey], y = b[sortKey];
    const v = (x == null) - (y == null) || (typeof x === "string" ? x.localeCompare(y) : x - y);
    return sortAsc ? v : -v;
  });
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    const cells = [
      `<i class="swatch" style="display:inline-block;margin-right:7px;background:${colorOf(r.model)}"></i>${r.model}`,
      r.task,
      r.nScored < LOW_N ? `${r.nScored} <span style="color:var(--muted)">low</span>` : String(r.nScored),
      fmt(r.score), fmt(r.stderr),
      r.ownsTotals ? String(r.errors) : "\\u2014",
      r.ownsTotals ? r.cost.toFixed(2) : "\\u2014",
      r.ownsTotals ? r.duration.toFixed(0) : "\\u2014",
    ];
    cells.forEach((c, i) => {
      const td = document.createElement("td");
      td.innerHTML = c;
      if (i === 0) td.style.whiteSpace = "nowrap";
      tr.appendChild(td);
    });
    t.appendChild(tr);
  });
}

function renderAll() {
  drawBars(document.getElementById("taskSel").value);
  drawScatter();
  drawTable();
}

const old = DATA.superseded || [];
if (old.length) {
  const list = old.map((r) => `${shortName(r.model)} / ${shortName(r.task)} (${r.startedAt})`).join("; ");
  document.getElementById("supersededNote").innerHTML =
    `<div class="badge info"><span><b>${old.length} older run${old.length > 1 ? "s" : ""} hidden</b>` +
    ` &mdash; this log directory has more than one successful run of the same model and task,` +
    ` so only the newest of each is shown: ${list}.</span></div>`;
}

const sel = document.getElementById("taskSel");
DATA.tasks.forEach((t) => {
  const o = document.createElement("option");
  o.value = t; o.textContent = t;
  sel.appendChild(o);
});
sel.onchange = () => drawBars(sel.value);

document.querySelectorAll("#xMode button").forEach((b) => {
  b.onclick = () => {
    PER_SAMPLE = b.dataset.mode === "per";
    document.querySelectorAll("#xMode button").forEach((o) => o.classList.toggle("on", o === b));
    drawScatter();
  };
});

document.getElementById("theme").onclick = () => {
  const cur = document.documentElement.dataset.theme;
  document.documentElement.dataset.theme = cur === "dark" ? "light" : cur === "light" ? "dark" : (isDark() ? "light" : "dark");
  renderAll();
};
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
addEventListener("resize", renderAll);
renderAll();
</script>
</body>
</html>
"""
