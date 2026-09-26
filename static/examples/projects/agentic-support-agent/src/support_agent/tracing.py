"""LangSmith tracing with PII anonymised before anything leaves the process.

Tracing is off unless LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set.
We attach our own LangChainTracer (instead of relying only on the env var) so
the client carries an anonymiser that masks emails, phones and card numbers
in every traced input and output.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client
from langsmith.anonymizer import create_anonymizer

from support_agent.config import Settings
from support_agent.pii import CARD_RE, EMAIL_RE, IBAN_RE, PHONE_RE

log = logging.getLogger(__name__)

PII_RULES: list[dict[str, Any]] = [
    {"pattern": EMAIL_RE, "replace": "[EMAIL]"},
    {"pattern": CARD_RE, "replace": "[CARD]"},
    {"pattern": IBAN_RE, "replace": "[IBAN]"},
    {"pattern": PHONE_RE, "replace": "[PHONE]"},
]


def build_callbacks(settings: Settings) -> list[BaseCallbackHandler]:
    if not (settings.langsmith_tracing and settings.langsmith_api_key):
        return []
    client = Client(
        api_key=settings.langsmith_api_key.get_secret_value(),
        anonymizer=create_anonymizer(PII_RULES),  # type: ignore[arg-type]
    )
    log.info("langsmith tracing enabled", extra={"project": settings.langsmith_project})
    return [LangChainTracer(project_name=settings.langsmith_project, client=client)]


def run_config(
    settings: Settings,
    callbacks: list[BaseCallbackHandler],
    *,
    thread_id: str,
    request_id: str,
    user_id: str,
    kind: str,
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """The RunnableConfig for one graph run: thread, limits, tracing metadata."""
    configurable: dict[str, Any] = {"thread_id": thread_id}
    if checkpoint_id:
        configurable["checkpoint_id"] = checkpoint_id
    return {
        "configurable": configurable,
        "recursion_limit": settings.recursion_limit,
        "callbacks": callbacks,
        "run_name": f"support-{kind}",
        "tags": ["support-agent", kind, settings.app_env],
        "metadata": {
            "thread_id": thread_id,
            "request_id": request_id,
            "user_id": user_id,
            "model": settings.llm_model if not settings.fake_llm else "fake",
        },
    }
