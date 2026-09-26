from __future__ import annotations

import base64
from pathlib import Path

import pytest

from data_analyst.sandbox.chart import ChartError, ChartSandbox, check_code

COLS = ["channel", "revenue"]
ROWS = [["web", 100.5], ["mobile", 80.0], ["store", 20.0]]
GOOD = "fig, ax = plt.subplots()\nax.bar(df['channel'], df['revenue'])\nax.set_title('x')\n"


@pytest.fixture(scope="module")
def sandbox() -> ChartSandbox:
    return ChartSandbox(timeout_s=30)


def test_renders_png(sandbox: ChartSandbox) -> None:
    out = sandbox.render(GOOD, COLS, ROWS)
    assert base64.b64decode(out.png_base64)[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize(
    "code",
    [
        "import os\nos.system('id')",
        "import subprocess",
        "from os import path",
        "open('/tmp/x', 'w').write('x')",
        "eval('1+1')",
        "__import__('os')",
        "x = ().__class__.__bases__",
        "getattr(df, 'to_csv')('/tmp/x')",
        "df.to_csv('/tmp/x.csv')",
        "pd.read_csv('/etc/passwd')",
        "plt.savefig('/tmp/evil.png')",
        "x = '__globals__'",
    ],
)
def test_static_gate_blocks(code: str) -> None:
    with pytest.raises(ChartError):
        check_code(code)


def test_runtime_hook_blocks_network(sandbox: ChartSandbox) -> None:
    with pytest.raises(ChartError, match="blocked"):
        sandbox.render("pd.io.common.urlopen('http://example.com')", COLS, ROWS)


def test_runtime_hook_blocks_writes_outside_workdir(sandbox: ChartSandbox, tmp_path: Path) -> None:
    target = tmp_path / "evil.npy"
    code = f"np.lib.format.open_memmap({str(target)!r}, mode='w+', shape=(3,))"
    with pytest.raises(ChartError, match="blocked"):
        sandbox.render(code, COLS, ROWS)
    assert not target.exists()


def test_timeout_kills_infinite_loop() -> None:
    with pytest.raises(ChartError, match="timeout"):
        ChartSandbox(timeout_s=3).render("while True:\n    x = 1\n", COLS, ROWS)


def test_code_that_draws_nothing_fails(sandbox: ChartSandbox) -> None:
    with pytest.raises(ChartError, match="failed"):
        sandbox.render("x = 1", COLS, ROWS)
