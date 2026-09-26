"""Prometheus metrics. Scraped from GET /metrics."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUESTS = Counter("support_requests_total", "Chat turns handled", ["intent", "outcome"])
LATENCY = Histogram(
    "support_turn_latency_seconds",
    "End-to-end latency of one chat turn",
    buckets=(0.25, 0.5, 1, 2, 3, 5, 8, 13, 21),
)
FIRST_TOKEN = Histogram(
    "support_first_token_seconds",
    "Time to first streamed token",
    buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8),
)
TOOL_CALLS = Counter("support_tool_calls_total", "Tool executions", ["tool", "status"])
REFUNDS = Counter("support_refunds_total", "Refund outcomes", ["status"])
INTERRUPTS = Counter("support_interrupts_total", "Human approvals requested")
GUARDRAIL_BLOCKS = Counter("support_guardrail_blocks_total", "Blocked inputs", ["reason"])
TOKENS = Counter("support_llm_tokens_total", "LLM tokens used", ["node"])
BUDGET_EXCEEDED = Counter("support_budget_exceeded_total", "Turns stopped by budget", ["kind"])
