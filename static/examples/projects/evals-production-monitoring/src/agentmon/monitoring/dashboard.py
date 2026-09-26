"""A static, dependency-free HTML dashboard (inline SVG), so it can be produced in CI,
attached to an incident, or served by the API at /dashboard."""

from __future__ import annotations

import html
from datetime import UTC, datetime
from typing import Any

from agentmon.monitoring.drift import kl, psi
from agentmon.monitoring.metrics import TraceFrame, aggregate, window_metrics
from agentmon.store import Store

HOUR = 3600.0
DAY = 86_400.0

CSS = """
:root{--bg:#f7f6f2;--card:#fff;--ink:#1f2328;--muted:#5d6570;--line:#d9d6cf;--accent:#2f6f9f;
--bad:#b3261e;--ok:#2e7d32;--dep:#8a5a00}
@media (prefers-color-scheme:dark){:root{--bg:#16181c;--card:#1f2227;--ink:#e6e6e6;
--muted:#a0a6ad;--line:#343840;--accent:#7fb3de;--bad:#ff8a80;--ok:#81c784;--dep:#e0b050}}
body{background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif;margin:0;padding:16px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:0 0 8px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0}.kpi{min-width:130px}
.kpi b{display:block;font-size:20px}.muted{color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:12.5px}td,th{border-bottom:1px solid var(--line);
padding:4px 6px;text-align:left;vertical-align:top}.bad{color:var(--bad)}.ok{color:var(--ok)}
svg text{fill:var(--muted);font-size:10px}.wide{grid-column:1/-1}
"""


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%a %d")


def _stamp(ts: float | None) -> str:
    return "-" if ts is None else datetime.fromtimestamp(ts, UTC).strftime("%a %d %H:%M")


def line_chart(
    points: list[tuple[float, float | None]],
    t0: float,
    t1: float,
    markers: list[tuple[float, str]],
    alerts: list[float],
    fmt: str = "{:.2f}",
    ymin: float | None = None,
    ymax: float | None = None,
) -> str:
    w, h, pad = 460, 150, 30
    vals = [v for _, v in points if v is not None]
    if not vals:
        return "<p class='muted'>no data</p>"
    lo = min(vals) if ymin is None else ymin
    hi = max(vals) if ymax is None else ymax
    hi = hi if hi > lo else lo + 1

    def x(t: float) -> float:
        return pad + (t - t0) / max(t1 - t0, 1) * (w - pad - 8)

    def y(v: float) -> float:
        return h - 18 - (v - lo) / (hi - lo) * (h - 30)

    parts = [f"<svg viewBox='0 0 {w} {h}' width='100%' role='img'>"]
    for v in (lo, hi):
        parts.append(f"<text x='2' y='{y(v) + 3:.0f}'>{html.escape(fmt.format(v))}</text>")
    d = 0.0
    while t0 + d <= t1:
        parts.append(f"<text x='{x(t0 + d):.0f}' y='{h - 4}'>{_day(t0 + d)}</text>")
        d += DAY
    for t, label in markers:
        parts.append(
            f"<line x1='{x(t):.1f}' x2='{x(t):.1f}' y1='8' y2='{h - 18}' "
            "stroke='var(--dep)' stroke-dasharray='4 3'/>"
            f"<text x='{x(t) + 3:.0f}' y='14'>{html.escape(label)}</text>"
        )
    for t in alerts:
        parts.append(f"<circle cx='{x(t):.1f}' cy='22' r='4' fill='var(--bad)'/>")
    path, pen = [], "M"
    for t, v in points:
        if v is None:
            pen = "M"
            continue
        path.append(f"{pen}{x(t):.1f},{y(v):.1f}")
        pen = "L"
    parts.append(
        f"<path d='{' '.join(path)}' fill='none' stroke='var(--accent)' stroke-width='2'/>"
    )
    parts.append("</svg>")
    return "".join(parts)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def render_dashboard(store: Store, extras: dict[str, Any] | None = None) -> str:
    extras = extras or {}
    frame = TraceFrame.load(store)
    if not frame.rows:
        return (
            f"<!doctype html><html><head><title>Agent monitoring</title><style>{CSS}</style></head>"
            "<body><h1>Agent monitoring</h1><p>No traces yet.</p></body></html>"
        )
    t0, t_last = frame.span
    t0 = t0 - (t0 % DAY)
    t1 = t_last - (t_last % DAY) + DAY
    six = aggregate(frame, t0, t1, 6 * HOUR)
    daily = aggregate(frame, t0, t1, DAY)
    deps = [(d["ts"], f"{d['component']}={d['new_value']}") for d in store.deployments()]
    alerts = store.alerts()
    alert_ts = [a["fired_ts"] for a in alerts]
    total = window_metrics(frame.rows)

    def series(rows: list[dict], key: str) -> list[tuple[float, float | None]]:
        return [(r["start"] + (r["end"] - r["start"]) / 2, r[key]) for r in rows]

    charts = [
        ("Requests per 6h", series(six, "n"), "{:.0f}", 0, None),
        ("Quality SLO: good-event rate (6h)", series(six, "good_rate"), "{:.0%}", 0, 1),
        ("Required tool called (6h)", series(six, "required_tool_rate"), "{:.0%}", 0, 1),
        (
            "Judge groundedness pass rate (daily)",
            series(daily, "groundedness_pass_rate"),
            "{:.0%}",
            0,
            1,
        ),
        ("Thumbs-down rate (daily)", series(daily, "thumbs_down_rate"), "{:.1%}", 0, None),
        ("p95 latency ms (6h)", series(six, "p95_latency_ms"), "{:.0f}", 0, None),
        ("Agent cost USD per day", series(daily, "cost_usd"), "${:.3f}", 0, None),
        ("Judge spend USD per day", series(daily, "judge_cost_usd"), "${:.4f}", 0, None),
    ]
    chart_html = "".join(
        f"<div class='card'><h2>{html.escape(title)}</h2>"
        f"{line_chart(pts, t0, t1, deps, alert_ts, fmt, lo, hi)}</div>"
        for title, pts, fmt, lo, hi in charts
    )
    base = frame.window(t0, t0 + 3 * DAY)
    base_counts: dict[str, float] = {}
    for r in base:
        base_counts[r.trace.intent] = base_counts.get(r.trace.intent, 0) + 1
    intents = sorted({k for d in daily for k in d["intent_counts"]})
    drift_rows = []
    for d in daily:
        if not d["n"]:
            continue
        c = d["intent_counts"]
        shares = [f"{c.get(k, 0) / d['n']:.0%}" for k in intents]
        p = psi(base_counts, c)
        cls = "bad" if p > 0.2 else "ok"
        drift_rows.append(
            [
                _day(d["start"]),
                d["n"],
                *shares,
                f"<span class='{cls}'>{p:.3f}</span>",
                f"{kl(c, base_counts):.3f}",
                ", ".join(f"{k}:{v}" for k, v in d["prompt_versions"].items()),
            ]
        )
    alert_rows = [
        [
            html.escape(a["rule"]),
            a["severity"],
            _stamp(a["fired_ts"]),
            _stamp(a["resolved_ts"]),
            f"{a['value']:.3f}",
            a["threshold"],
            html.escape(a["message"]),
        ]
        for a in alerts
    ]
    rc = [a for a in alerts if "root_cause" in a["evidence"]]
    rc_html = (
        "".join(
            f"<p><b>{html.escape(a['rule'])}</b> at {_stamp(a['fired_ts'])}: "
            f"{html.escape(a['evidence']['root_cause']['summary'])}</p>"
            + _table(
                ["dimension", "value", "n", "bad rate", "lift", "share of bad"],
                [
                    [
                        s["dimension"],
                        html.escape(str(s["value"])),
                        s["n"],
                        f"{s['bad_rate']:.0%}",
                        s["lift"],
                        f"{s['share_of_bad']:.0%}",
                    ]
                    for s in a["evidence"]["root_cause"]["slices"][:5]
                ],
            )
            + "".join(
                f"<p class='muted'>exemplar <code>{e['trace_id']}</code>: "
                f"{html.escape(e['input'])} -> {html.escape(e['output'][:120])} "
                f"[{', '.join(e['failed'])}]</p>"
                for e in a["evidence"]["root_cause"].get("exemplars", [])
            )
            for a in rc[:2]
        )
        or "<p class='muted'>No page-level quality alerts.</p>"
    )
    review = store.review_items()
    rstatus: dict[str, int] = {}
    for item in review:
        rstatus[item["status"]] = rstatus.get(item["status"], 0) + 1
    jobs = store.job_counts()
    extra_html = ""
    if "redteam" in extras:
        rows = []
        off, on = extras["redteam"]["off"], extras["redteam"]["on"]
        for cat in off["asr_by_category"]:
            rows.append(
                [
                    cat,
                    f"{off['asr_by_category'][cat]:.0%}",
                    f"{on['asr_by_category'].get(cat, 0):.0%}",
                ]
            )
        rows.append(["<b>overall</b>", f"{off['asr']:.0%}", f"{on['asr']:.0%}"])
        rows.append(
            [
                "false refusals (benign)",
                f"{off['false_refusal_rate']:.0%}",
                f"{on['false_refusal_rate']:.0%}",
            ]
        )
        extra_html += (
            "<div class='card'><h2>Red team: attack success rate</h2>"
            + _table(["category", "guardrails off", "guardrails on"], rows)
            + "</div>"
        )
    if "regression" in extras:
        rows = []
        for version, rep in extras["regression"].items():
            m = rep["metrics"]
            verdict = (
                "<span class='ok'>PASS</span>"
                if rep["gate"]["passed"]
                else "<span class='bad'>FAIL</span>"
            )
            rows.append(
                [
                    version,
                    rep["n"],
                    f"{m['tool_call_accuracy']:.0%}",
                    f"{m['trajectory_match']:.0%}",
                    f"{m['task_completion']:.0%}",
                    f"{m['groundedness']:.0%}",
                    verdict,
                ]
            )
        extra_html += (
            "<div class='card'><h2>Offline regression gate</h2>"
            + _table(
                ["prompt", "cases", "tool acc", "trajectory", "task", "grounded", "gate"], rows
            )
            + "</div>"
        )
    kpis = [
        ("requests", total["n"]),
        ("good-event rate", f"{total['good_rate']:.1%}"),
        ("p95 latency", f"{total['p95_latency_ms']:.0f} ms"),
        ("agent cost", f"${total['cost_usd']:.3f}"),
        ("judge spend", f"${total['judge_cost_usd']:.4f}"),
        ("judged traces", total["judged_n"]),
        ("alerts", len(alerts)),
        ("review queue", ", ".join(f"{k} {v}" for k, v in rstatus.items()) or "0"),
    ]
    kpi_html = "".join(
        f"<div class='kpi card'><span class='muted'>{k}</span><b>{v}</b></div>" for k, v in kpis
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent monitoring</title><style>{CSS}</style></head><body>
<h1>Banking agent: production monitoring</h1>
<p class="muted">{_stamp(frame.span[0])} to {_stamp(frame.span[1])}. Dashed lines are deployments,
red dots are alerts firing. Eval jobs: {html.escape(str(jobs))}.</p>
<div class="kpis">{kpi_html}</div>
<div class="grid">{chart_html}
<div class="card wide"><h2>Alerts</h2>{
        _table(
            ["rule", "severity", "fired", "resolved", "value", "threshold", "detail"], alert_rows
        )
        if alert_rows
        else "<p class='muted'>none</p>"
    }</div>
<div class="card wide"><h2>Root cause</h2>{rc_html}</div>
<div class="card wide"><h2>Input drift: intent share per day vs days 1-3</h2>
{_table(["day", "n", *intents, "PSI", "KL", "prompt"], drift_rows)}</div>
{extra_html}
</div></body></html>"""
