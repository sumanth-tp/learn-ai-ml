# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY evals ./evals
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 10001 analyst
WORKDIR /app
COPY --from=build --chown=analyst:analyst /app /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    ANALYST_DATA_DIR=/app/data \
    ANALYST_LOG_JSON=true \
    MPLCONFIGDIR=/tmp/mpl
RUN mkdir -p /app/data && chown analyst:analyst /app/data
USER analyst
# Seed at build time so a fresh container is ready; a mounted volume keeps its own copy.
RUN analyst seed
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"
CMD ["analyst", "serve", "--host", "0.0.0.0", "--port", "8000"]
