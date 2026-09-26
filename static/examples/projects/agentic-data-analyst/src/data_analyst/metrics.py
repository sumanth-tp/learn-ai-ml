"""Prometheus metrics. Scraped from GET /metrics; the CLI updates them too (no-op cost)."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

TURNS = Counter("analyst_turns_total", "Finished turns by outcome", ["status", "sql_source"])
RETRIES = Histogram(
    "analyst_sql_retries", "Self-correction retries per turn", buckets=(0, 1, 2, 3, 5, 10)
)
LATENCY = Histogram(
    "analyst_turn_seconds",
    "Machine time per turn segment",
    buckets=(0.25, 0.5, 1, 2, 4, 8, 15, 30, 60),
)
TOKENS = Counter("analyst_llm_tokens_total", "LLM tokens", ["direction"])
COST = Counter("analyst_llm_cost_usd_total", "Estimated LLM spend in USD")
APPROVALS = Counter("analyst_approvals_total", "Human approval requests and outcomes", ["outcome"])
CHART_FAILURES = Counter("analyst_chart_failures_total", "Charts that failed in the sandbox")
REDACTIONS = Counter("analyst_dlp_redactions_total", "Values redacted by the result DLP pass")
