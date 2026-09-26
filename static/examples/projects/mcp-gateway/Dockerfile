# syntax=docker/dockerfile:1.7
# One image, three roles: the gateway (default), and the two demo HTTP
# upstreams (compose overrides the command). The stdio docs upstream runs
# inside the gateway container as a child process.

FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 gateway
WORKDIR /app
COPY --from=build --chown=gateway:gateway /app /app
COPY --chown=gateway:gateway config ./config
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    GATEWAY_HOST=0.0.0.0 \
    GATEWAY_PORT=8080 \
    GATEWAY_STATE_DIR=/var/lib/mcp-gateway
RUN mkdir -p /var/lib/mcp-gateway && chown gateway:gateway /var/lib/mcp-gateway
USER gateway
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).status == 200 else 1)"
CMD ["mcp-gateway", "serve"]
