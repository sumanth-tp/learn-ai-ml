"""Run the whole system locally with one command: two HTTP upstreams and the
gateway (which itself spawns the stdio docs upstream).

This is the non-Docker equivalent of ``docker compose up``. In production
each upstream is its own deployment that owns its credential; here the
stack hands each demo upstream the same value the gateway's broker holds.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import httpx2

PAYMENTS_PORT = int(os.environ.get("PAYMENTS_PORT", "9101"))
TICKETS_PORT = int(os.environ.get("TICKETS_PORT", "9102"))


def child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PAYMENTS_URL", f"http://127.0.0.1:{PAYMENTS_PORT}/mcp")
    env.setdefault("TICKETS_URL", f"http://127.0.0.1:{TICKETS_PORT}/mcp")
    env["PAYMENTS_UPSTREAM_TOKEN"] = env.get("GATEWAY_SECRET_PAYMENTS_TOKEN", "")
    env["TICKETS_API_KEY"] = env.get("GATEWAY_SECRET_TICKETS_KEY", "")
    env.update(extra or {})
    return env


def spawn(args: list[str], env: dict[str, str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen([sys.executable, *args], env=env)  # noqa: S603


def spawn_payments(env: dict[str, str]) -> subprocess.Popen[bytes]:
    return spawn(["-m", "mcp_gateway.demo_upstreams.payments_server",
                  "--port", str(PAYMENTS_PORT)], env)


def spawn_tickets(env: dict[str, str]) -> subprocess.Popen[bytes]:
    return spawn(["-m", "mcp_gateway.demo_upstreams.tickets_server",
                  "--port", str(TICKETS_PORT)], env)


def spawn_gateway(env: dict[str, str]) -> subprocess.Popen[bytes]:
    return spawn(["-m", "mcp_gateway.cli", "serve"], env)


def wait_for_port(port: int, timeout: float = 20.0) -> None:
    """Wait until something answers HTTP on the port (any status code)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            httpx2.get(f"http://127.0.0.1:{port}/", timeout=1.0)
            return
        except httpx2.HTTPError:
            time.sleep(0.2)
    raise TimeoutError(f"nothing listening on port {port} after {timeout}s")


def wait_ready(url: str, timeout: float = 30.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    last: object = None
    while time.monotonic() < deadline:
        try:
            r = httpx2.get(url, timeout=5.0)
            last = r.json()
            if r.status_code == 200:
                return r.json()
        except (httpx2.HTTPError, ValueError) as exc:
            last = repr(exc)
        time.sleep(0.3)
    raise TimeoutError(f"gateway not ready after {timeout}s: {last}")


def stop(procs: list[subprocess.Popen[bytes]]) -> None:
    for p in procs:
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


def run_stack() -> int:
    env = child_env()
    procs = [spawn_payments(env), spawn_tickets(env)]
    try:
        wait_for_port(PAYMENTS_PORT)
        wait_for_port(TICKETS_PORT)
        gateway = spawn_gateway(env)
        procs.append(gateway)
        port = os.environ.get("GATEWAY_PORT", "8080")
        print(wait_ready(f"http://127.0.0.1:{port}/readyz"), flush=True)
        print(f"gateway ready on http://127.0.0.1:{port}/mcp  (Ctrl+C to stop)", flush=True)
        return gateway.wait()
    except KeyboardInterrupt:
        return 0
    finally:
        stop(procs)
