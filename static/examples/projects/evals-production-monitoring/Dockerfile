# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/
WORKDIR /app

# Dependencies first, so code edits do not bust the dependency layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra otlp --no-install-project

COPY src ./src
COPY data ./data
COPY config ./config
RUN uv sync --frozen --no-dev --extra otlp

RUN useradd --create-home --uid 10001 app && mkdir -p /app/var /app/out && chown -R app /app/var /app/out
USER app
ENV PATH="/app/.venv/bin:$PATH" AGENTMON_DB_PATH=/app/var/agentmon.db AGENTMON_OUT_DIR=/app/out
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["agentmon", "serve", "--host", "0.0.0.0", "--port", "8000"]
