"""Document store upstream, served over **stdio**.

The gateway spawns it as a child process and injects ``DOCS_API_TOKEN`` into
its environment; the server refuses to start without it (a real server would
use it to call the document backend).
"""

from __future__ import annotations

import hashlib
import os
import posixpath

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

DOCS: dict[str, str] = {
    "/public/handbook.md": "# Employee handbook\nCore hours are 10:00-16:00. Expenses need a receipt.",
    "/public/security.md": "# Security basics\nReport phishing to security@example.com within 1 hour.",
    "/engineering/runbook.md": "# Gateway runbook\nIf /readyz fails, check the upstream breaker state.",
    "/engineering/adr-007.md": "# ADR 7\nWe route every MCP call through the gateway.",
    "/finance/q3-forecast.md": "# Q3 forecast (confidential)\nRevenue 4.2M EUR, margin 18%.",
    "/hr/salaries.csv": "name,salary\nA. Example,90000\nB. Example,85000",
}


def _normal(path: str) -> str:
    return posixpath.normpath("/" + path.lstrip("/"))


def create_server() -> FastMCP:
    mcp = FastMCP("docs", instructions="Company documents, read-only.")

    @mcp.tool(annotations={"readOnlyHint": True})
    def list_docs(prefix: str = "/") -> list[str]:
        """List document paths under a prefix."""
        p = _normal(prefix)
        return sorted(d for d in DOCS if d.startswith(p.rstrip("/") + "/") or d == p)

    @mcp.tool(annotations={"readOnlyHint": True})
    def read_doc(path: str) -> str:
        """Read one document by its absolute path, e.g. /public/handbook.md."""
        p = _normal(path)
        if p not in DOCS:
            raise ToolError(f"no document at {p}")
        return DOCS[p]

    @mcp.tool(annotations={"readOnlyHint": True})
    def search_docs(query: str, prefix: str = "/public") -> list[str]:
        """Case-insensitive search of document text under a prefix; returns matching paths."""
        p = _normal(prefix).rstrip("/") + "/"
        q = query.lower()
        return [d for d, text in DOCS.items() if d.startswith(p) and q in text.lower()]

    @mcp.tool(annotations={"readOnlyHint": True})
    def backend_status() -> dict[str, str]:
        """Show which backend credential this server was started with (fingerprint only)."""
        token = os.environ.get("DOCS_API_TOKEN", "")
        return {"credential_sha256_8": hashlib.sha256(token.encode()).hexdigest()[:8]}

    @mcp.resource("docs://index", mime_type="text/plain")
    def index() -> str:
        """Index of all public documents."""
        return "\n".join(d for d in sorted(DOCS) if d.startswith("/public/"))

    return mcp


mcp = create_server()


def main() -> None:
    if not os.environ.get("DOCS_API_TOKEN"):
        raise SystemExit("DOCS_API_TOKEN is required (the gateway injects it)")
    mcp.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()
