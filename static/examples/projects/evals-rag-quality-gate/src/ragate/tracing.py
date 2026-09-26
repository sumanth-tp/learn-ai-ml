"""LangSmith tracing is configured purely by environment variables.

LangChain chat models and every ``@traceable`` function (pipeline.ask, judge calls)
send traces when LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set. With them unset,
``traceable`` is a no-op, which is what tests and offline CI rely on.
"""

from __future__ import annotations

import os


def tracing_status() -> dict[str, str | bool]:
    enabled = os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
    return {
        "enabled": enabled and bool(os.environ.get("LANGSMITH_API_KEY")),
        "project": os.environ.get("LANGSMITH_PROJECT", "default"),
    }
