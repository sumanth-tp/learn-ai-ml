"""Test-only stdio upstream that reports which environment variables it received."""

import os

from fastmcp import FastMCP

mcp = FastMCP("envecho")


@mcp.tool(annotations={"readOnlyHint": True})
def env_keys() -> list[str]:
    """List environment variable names visible to this process."""
    return sorted(os.environ)


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
