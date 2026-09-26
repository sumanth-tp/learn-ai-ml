"""Structured JSON logging with a request id bound to every line.

``structlog.contextvars`` carries ``request_id``, ``user`` and ``tenant`` for
the duration of one MCP request, so every log line emitted while handling it
(by our code or by a library through stdlib logging) can be joined up later.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", json: bool = True) -> None:
    """Route stdlib and structlog output through one JSON renderer on stderr.

    stderr, not stdout: in stdio transport, stdout *is* the MCP channel and a
    stray log line there corrupts the protocol stream.
    """
    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    renderer = (
        structlog.processors.JSONRenderer()
        if json
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.format_exc_info,
                renderer,
            ],
        )
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # FastMCP installs its own rich handler; send its records to ours instead.
    for name in ("fastmcp", "mcp", "uvicorn", "uvicorn.error", "uvicorn.access"):
        lib = logging.getLogger(name)
        lib.handlers[:] = []
        lib.propagate = True
    logging.getLogger("sqlalchemy.engine").setLevel("WARNING")


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
