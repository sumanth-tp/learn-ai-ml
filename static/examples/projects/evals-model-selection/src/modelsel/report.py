"""Render the run summary as Markdown (for PRs and wikis) and HTML (for people)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

_TEMPLATES = Path(__file__).parent / "templates"


def _env(autoescape: bool) -> Environment:
    env = Environment(
        loader=FileSystemLoader(_TEMPLATES),
        autoescape=select_autoescape(["html"]) if autoescape else False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["pct"] = lambda v: "n/a" if v is None else f"{v:.1%}"
    env.filters["f3"] = lambda v: "n/a" if v is None else f"{v:.3f}"
    env.filters["f2"] = lambda v: "n/a" if v is None else f"{v:.2f}"
    env.filters["usd"] = lambda v: f"${v:,.4f}" if v < 1 else f"${v:,.2f}"
    env.filters["ms"] = lambda v: f"{v:,.0f} ms"
    return env


def render_markdown(summary: dict[str, Any]) -> str:
    return _env(False).get_template("report.md.j2").render(s=summary)


def render_html(summary: dict[str, Any]) -> str:
    return _env(True).get_template("report.html.j2").render(s=summary)
