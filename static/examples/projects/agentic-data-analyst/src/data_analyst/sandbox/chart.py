"""Run model-written pandas/matplotlib code safely.

Three layers, because each alone is bypassable:
1. a static AST gate in this process (import allow-list, no dunders, no eval/open);
2. a separate Python process (-I isolated mode) with an empty environment, a temp
   working directory, resource limits and a wall-clock timeout;
3. a runtime audit hook in that process that blocks sockets, subprocesses, file
   deletes and any write outside the temp directory.
In production add a fourth: run the child in its own container with no network
(gVisor, Firecracker or a Kubernetes pod with a deny-all NetworkPolicy).
"""

from __future__ import annotations

import ast
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

RUNNER = Path(__file__).with_name("runner.py")
ALLOWED_IMPORTS = frozenset({"pandas", "numpy", "matplotlib", "matplotlib.pyplot", "math"})
FORBIDDEN_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "input",
        "breakpoint",
        "exit",
        "quit",
        "help",
        "memoryview",
        "type",
        "super",
        "object",
    }
)
FORBIDDEN_ATTRS = frozenset(
    {
        "savefig",
        "to_csv",
        "to_pickle",
        "to_parquet",
        "to_sql",
        "to_excel",
        "to_json",
        "to_hdf",
        "to_feather",
        "system",
        "popen",
        "show",
        "imsave",
        "load",
        "save",
    }
)


class ChartError(Exception):
    pass


class ChartResult(BaseModel):
    png_base64: str
    bytes: int


def check_code(code: str) -> None:
    """Static gate. Raises ChartError naming the first violation."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ChartError(f"chart code does not parse: {e.msg}") from e
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORTS:
                    raise ChartError(f"import of {alias.name} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "") not in ALLOWED_IMPORTS or node.level:
                raise ChartError(f"import from {node.module} is not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ChartError(f"use of {node.id} is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise ChartError(f"private attribute access .{node.attr} is not allowed")
            if node.attr in FORBIDDEN_ATTRS or node.attr.startswith("read_"):
                raise ChartError(f"method .{node.attr} is not allowed")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            raise ChartError("dunder strings are not allowed")


class ChartSandbox:
    def __init__(self, timeout_s: float = 20.0, memory_mb: int = 1024) -> None:
        self.timeout_s = timeout_s
        self.memory_mb = memory_mb

    def render(self, code: str, columns: list[str], rows: list[list[Any]]) -> ChartResult:
        check_code(code)
        with tempfile.TemporaryDirectory(prefix="chart-") as tmp:
            work = Path(tmp)
            data = work / "data.json"
            data.write_text(
                json.dumps({"columns": columns, "index": list(range(len(rows))), "data": rows})
            )
            (work / "chart.py").write_text(code)
            out = work / "chart.png"
            env = {"MPLCONFIGDIR": tmp, "HOME": tmp, "TMPDIR": tmp, "PATH": "/usr/bin:/bin"}
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        str(RUNNER),
                        str(data),
                        str(work / "chart.py"),
                        str(out),
                        tmp,
                        str(self.memory_mb),
                        str(int(self.timeout_s) + 1),
                    ],
                    cwd=tmp,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired as e:
                raise ChartError(f"chart code exceeded the {self.timeout_s:.0f}s timeout") from e
            if proc.returncode != 0 or not out.exists():
                tail = (proc.stderr or "").strip().splitlines()[-1:] or ["no output"]
                raise ChartError(f"chart code failed: {tail[0][:300]}")
            png = out.read_bytes()
        return ChartResult(png_base64=base64.b64encode(png).decode(), bytes=len(png))
