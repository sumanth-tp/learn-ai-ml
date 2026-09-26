# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /usr/local/bin/
WORKDIR /app

# Dependencies first: this layer is cached until uv.lock changes.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY src ./src
COPY data ./data
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 app && mkdir -p /app/var /app/reports && chown -R app /app/var /app/reports
USER app
ENV PATH="/app/.venv/bin:$PATH" MODELSEL_VAR_DIR=/app/var MODELSEL_REPORTS_DIR=/app/reports
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"
CMD ["modelsel", "serve", "--host", "0.0.0.0", "--port", "8000"]
