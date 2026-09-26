"""The CLI end to end, in a temporary working directory."""

from __future__ import annotations

import pytest

from research_analyst.cli import main
from tests.conftest import SODIUM_Q


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RA_MODE", "offline")
    monkeypatch.setenv("RA_LOG_LEVEL", "WARNING")
    return tmp_path


def test_seed_run_status_eval(workdir, capsys):
    assert main(["seed"]) == 0
    assert (workdir / "data" / "index-hashing.json").exists()
    assert main(["run", SODIUM_Q, "--thread-id", "cli-1", "--out", "out/r.md"]) == 0
    assert (workdir / "out" / "r.md").read_text().startswith("# Research brief")
    assert (workdir / "out" / "r.json").exists()
    assert main(["status", "cli-1"]) == 0
    assert '"complete"' in capsys.readouterr().out
    assert main(["status", "unknown"]) == 1
    assert main(["purge", "--days", "30"]) == 0
    assert main(["delete", "cli-1"]) == 0
    assert main(["eval", "--out", "out/eval.json"]) == 0
    assert "GATE: PASS" in capsys.readouterr().out
