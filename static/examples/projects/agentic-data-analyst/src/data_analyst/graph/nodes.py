"""Graph nodes. Each node is a small, testable function over the state.

Nodes never raise for expected failures (bad SQL, DB errors, chart errors): they
record the error in state and let the routers decide between retry, fallback and
stopping. Only programming errors and provider outages propagate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from langgraph.config import get_stream_writer
from langgraph.types import interrupt

from data_analyst import prompts
from data_analyst.config import Settings
from data_analyst.executor import QueryError, WarehouseExecutor
from data_analyst.graph.state import AnalystState, Turn
from data_analyst.llm import LLMOutputError, StructuredLLM, Usage
from data_analyst.logging_setup import get_logger
from data_analyst.retrieval import SchemaIndex, SemanticCache
from data_analyst.sandbox.chart import ChartError, ChartSandbox
from data_analyst.schemas import (
    ChartCode,
    Interpretation,
    QueryPlan,
    SQLDraft,
    StandaloneQuestion,
)
from data_analyst.validator import SQLValidator
from data_analyst.warehouse.catalog import render_schema

log = get_logger(__name__)
PREVIEW_ROWS = 20


@dataclass
class Deps:
    settings: Settings
    llm: StructuredLLM
    index: SchemaIndex
    validator: SQLValidator
    executor: WarehouseExecutor
    charts: ChartSandbox
    cache: SemanticCache | None = None


def emit(node: str, message: str, **data: Any) -> None:
    """Progress event on the 'custom' stream. A no-op when a node is called directly."""
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer({"node": node, "message": message, **data})


def _usage(state: AnalystState, usage: Usage) -> dict[str, int]:
    return {
        "input_tokens": state.get("input_tokens", 0) + usage.input_tokens,
        "output_tokens": state.get("output_tokens", 0) + usage.output_tokens,
        "llm_calls": state.get("llm_calls", 0) + usage.calls,
    }


class AnalystNodes:
    def __init__(self, deps: Deps) -> None:
        self.d = deps
        self.s = deps.settings

    # ------------------------------------------------------------------ memory
    def contextualise(self, state: AnalystState) -> dict[str, Any]:
        """Start of a turn: reset working state and resolve follow-ups into a standalone
        question. The standalone question, not the raw text, drives cache and retrieval,
        so "now only for 2024" never hits the cache entry for "total revenue by year"."""
        reset: dict[str, Any] = {
            "tables": [],
            "table_scores": [],
            "schema_context": "",
            "plan": None,
            "sql": None,
            "sql_source": None,
            "cache_similarity": None,
            "attempts": 0,
            "errors": [],
            "last_error": None,
            "warnings": [],
            "estimate": None,
            "approval": None,
            "result": None,
            "answer": None,
            "chart_recommended": False,
            "chart_png_base64": None,
            "chart_error": None,
            "status": "running",
            "input_tokens": 0,
            "output_tokens": 0,
            "llm_calls": 0,
        }
        question = state["question"].strip()
        history = list(state.get("history", []))[-self.s.history_turns :]
        if not history:
            return {**reset, "standalone_question": question}
        rendered = "\n".join(f"- Q: {t['standalone_question']}\n  SQL: {t['sql']}" for t in history)
        out, usage = self.d.llm.generate(
            "rewrite",
            StandaloneQuestion,
            prompts.REWRITE.format_messages(history=rendered, question=question),
            {"question": question, "history": history},
        )
        emit("contextualise", "rewrote follow-up", standalone=out.question)
        return {**reset, "standalone_question": out.question, **_usage(reset, usage)}  # type: ignore[arg-type]

    # ------------------------------------------------------------------ cache
    def cache_lookup(self, state: AnalystState) -> dict[str, Any]:
        if self.d.cache is None:
            return {}
        hit = self.d.cache.lookup(state["standalone_question"])
        if hit is None:
            return {}
        emit("cache_lookup", "semantic cache hit", similarity=hit.similarity)
        return {"sql": hit.sql, "sql_source": "cache", "cache_similarity": hit.similarity}

    # ------------------------------------------------------------------ schema
    def retrieve_schema(self, state: AnalystState) -> dict[str, Any]:
        tables, hits = self.d.index.retrieve(state["standalone_question"], self.s.schema_top_k)
        emit("retrieve_schema", "retrieved tables", tables=tables)
        return {
            "tables": tables,
            "table_scores": [h.model_dump() for h in hits],
            "schema_context": render_schema(tables),
        }

    def plan(self, state: AnalystState) -> dict[str, Any]:
        out, usage = self.d.llm.generate(
            "plan",
            QueryPlan,
            prompts.PLAN.format_messages(
                schema=state["schema_context"], question=state["standalone_question"]
            ),
            {"question": state["standalone_question"], "tables": state["tables"]},
        )
        return {"plan": out.model_dump(), **_usage(state, usage)}

    def generate_sql(self, state: AnalystState) -> dict[str, Any]:
        attempt = state.get("attempts", 0) + 1
        feedback = ""
        if state.get("last_error"):
            feedback = prompts.FEEDBACK.format(sql=state.get("sql"), error=state["last_error"])
        messages = prompts.GENERATE_SQL.format_messages(
            schema=state["schema_context"],
            plan=json.dumps(state.get("plan"), indent=2),
            question=state["standalone_question"],
            feedback=feedback,
        )
        hints = {
            "question": state["standalone_question"],
            "tables": state["tables"],
            "attempt": attempt,
            "last_error": state.get("last_error"),
        }
        try:
            out, usage = self.d.llm.generate("sql", SQLDraft, messages, hints)
        except LLMOutputError as e:
            err = f"model output error: {e}"
            return {
                "attempts": attempt,
                "sql": None,
                "last_error": err,
                "errors": [*state.get("errors", []), err],
            }
        emit("generate_sql", f"attempt {attempt}", sql=out.sql)
        return {
            "attempts": attempt,
            "sql": out.sql.strip().rstrip(";"),
            "sql_source": "llm",
            "last_error": None,
            **_usage(state, usage),
        }

    # ------------------------------------------------------------------ safety
    def validate(self, state: AnalystState) -> dict[str, Any]:
        if state.get("sql") is None:
            return {}
        result = self.d.validator.validate(state["sql"])
        if result.ok:
            emit("validate", "passed", warnings=result.warnings)
            return {"sql": result.sql, "warnings": result.warnings, "last_error": None}
        return self._failure(state, "validate", f"validation failed: {result.error}")

    def _failure(self, state: AnalystState, node: str, err: str) -> dict[str, Any]:
        """Record a failed step. A failing cached query is evicted, so one bad entry
        cannot keep failing every user who asks the same question."""
        emit(node, err)
        if state.get("sql_source") == "cache" and self.d.cache is not None and state.get("sql"):
            self.d.cache.invalidate(state["sql"])  # type: ignore[arg-type]
        return {"last_error": err, "errors": [*state.get("errors", []), err]}

    def estimate(self, state: AnalystState) -> dict[str, Any]:
        try:
            est = self.d.executor.estimate(state["sql"])  # type: ignore[arg-type]
        except QueryError as e:
            return self._failure(state, "estimate", f"database error: {e}")
        emit("estimate", "estimated cost", **est.model_dump())
        return {"estimate": est.model_dump(), "last_error": None}

    def approval(self, state: AnalystState) -> dict[str, Any]:
        """Pause the graph for a human. State is checkpointed, so the process can die
        and the reviewer can answer hours later, from another machine."""
        decision = interrupt(
            {
                "kind": "approve_expensive_query",
                "question": state["standalone_question"],
                "sql": state["sql"],
                "estimate": state["estimate"],
                "thresholds": {
                    "rows": self.s.approval_row_threshold,
                    "work": self.s.approval_work_threshold,
                },
            }
        )
        if not isinstance(decision, dict):
            decision = {"approved": bool(decision)}
        approved = bool(decision.get("approved"))
        return {
            "approval": {
                "approved": approved,
                "reviewer": decision.get("reviewer", "unknown"),
                "comment": decision.get("comment", ""),
            }
        }

    def execute(self, state: AnalystState) -> dict[str, Any]:
        try:
            result = self.d.executor.run(state["sql"])  # type: ignore[arg-type]
        except QueryError as e:
            return self._failure(state, "execute", f"database error: {e}")
        emit("execute", "rows returned", rows=result.row_count, truncated=result.truncated)
        return {"result": result.model_dump(), "last_error": None}

    # ------------------------------------------------------------------ answer
    def interpret(self, state: AnalystState) -> dict[str, Any]:
        res = state["result"] or {}
        rows = res.get("rows", [])[:PREVIEW_ROWS]
        messages = prompts.INTERPRET.format_messages(
            question=state["standalone_question"],
            sql=state["sql"],
            columns=res.get("columns"),
            shown=len(rows),
            total=res.get("row_count", 0),
            truncated=", truncated" if res.get("truncated") else "",
            rows="\n".join(json.dumps(r, default=str) for r in rows),
        )
        out, usage = self.d.llm.generate(
            "interpret",
            Interpretation,
            messages,
            {
                "question": state["standalone_question"],
                "columns": res.get("columns", []),
                "rows": rows,
                "total": res.get("row_count", 0),
            },
        )
        answer = out.answer + (
            " (Result truncated at the row cap.)" if res.get("truncated") else ""
        )
        return {
            "answer": answer,
            "chart_recommended": out.chart_recommended,
            **_usage(state, usage),
        }

    def chart(self, state: AnalystState) -> dict[str, Any]:
        res = state["result"] or {}
        head = res.get("rows", [])[:5]
        dtypes = {c: type(v).__name__ for c, v in zip(res["columns"], head[0], strict=False)}
        try:
            out, usage = self.d.llm.generate(
                "chart",
                ChartCode,
                prompts.CHART.format_messages(
                    question=state["standalone_question"],
                    dtypes=dtypes,
                    head=json.dumps(head, default=str),
                ),
                {"question": state["standalone_question"], "columns": res["columns"], "rows": head},
            )
            chart = self.d.charts.render(out.code, res["columns"], res["rows"])
        except (ChartError, LLMOutputError) as e:
            emit("chart", f"chart skipped: {e}")
            return {"chart_error": str(e)}
        emit("chart", "chart rendered", bytes=chart.bytes)
        return {"chart_png_base64": chart.png_base64, **_usage(state, usage)}

    def finalize(self, state: AnalystState) -> dict[str, Any]:
        status: Literal["answered", "rejected", "failed"]
        approval = state.get("approval")
        if state.get("result") is not None:
            status = "answered"
            answer = state.get("answer") or ""
        elif approval is not None and not approval["approved"]:
            status = "rejected"
            answer = f"Not run: the reviewer ({approval['reviewer']}) rejected the query."
        else:
            status = "failed"
            answer = (
                "I could not produce a valid query after "
                f"{state.get('attempts', 0)} attempts. Last error: {state.get('last_error')}"
            )
        if status == "answered" and state.get("sql_source") == "llm" and self.d.cache is not None:
            self.d.cache.store(state["standalone_question"], state["sql"])  # type: ignore[arg-type]
        turn: Turn = {
            "question": state["question"],
            "standalone_question": state["standalone_question"],
            "sql": state.get("sql"),
            "status": status,
            "answer": answer,
        }
        log.info(
            "turn_finished",
            status=status,
            attempts=state.get("attempts", 0),
            source=state.get("sql_source"),
            llm_calls=state.get("llm_calls", 0),
        )
        return {"status": status, "answer": answer, "history": [turn]}

    # ------------------------------------------------------------------ routers
    def route_after_cache(self, state: AnalystState) -> str:
        return "validate" if state.get("sql_source") == "cache" else "retrieve_schema"

    def _retry_or_stop(self, state: AnalystState) -> str:
        if state.get("attempts", 0) == 0:
            return "retrieve_schema"  # the SQL came from the cache and failed: start fresh
        if state.get("attempts", 0) <= self.s.max_retries:
            return "generate_sql"
        return "finalize"

    def route_after_validate(self, state: AnalystState) -> str:
        return self._retry_or_stop(state) if state.get("last_error") else "estimate"

    def route_after_estimate(self, state: AnalystState) -> str:
        if state.get("last_error"):
            return self._retry_or_stop(state)
        est = state["estimate"] or {}
        if (
            est.get("estimated_rows", 0) > self.s.approval_row_threshold
            or est.get("max_intermediate_rows", 0) > self.s.approval_work_threshold
        ):
            return "approval"
        return "execute"

    def route_after_approval(self, state: AnalystState) -> str:
        return "execute" if (state.get("approval") or {}).get("approved") else "finalize"

    def route_after_execute(self, state: AnalystState) -> str:
        return self._retry_or_stop(state) if state.get("last_error") else "interpret"

    def route_after_interpret(self, state: AnalystState) -> str:
        res = state.get("result") or {}
        if self.s.chart_enabled and state.get("chart_recommended") and res.get("row_count", 0) >= 2:
            return "chart"
        return "finalize"
