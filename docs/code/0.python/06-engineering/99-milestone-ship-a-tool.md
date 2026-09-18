---
id: py-milestone-engineering
title: "Milestone 6: Package, Test and Ship a CLI Tool"
sidebar_label: "Milestone 6: Ship a tool"
sidebar_position: 99
slug: /code/python/milestone-ship-a-tool
description: "Take working code and make it a real project: src layout, lockfile, tests, ruff, mypy, pre-commit, CI and a container."
tags: [python, milestone, project, packaging, testing, ci, docker, tooling]
---

**In one line.** Turn a working script into something a colleague can install, run, test and deploy without asking you a single question.

## The brief

You have a useful script. Make it a **project**: installable with one command, runnable as `dupefind` from anywhere, covered by tests, checked by ruff and mypy, wired to CI, and packaged in a container that runs as a non-root user.

The tool itself is deliberately small — find duplicate files in a directory tree by content hash — so that all the attention goes on the engineering.

## Requirements

- [ ] `src/` layout with a `pyproject.toml` as the only metadata file.
- [ ] A console entry point: `dupefind ~/Downloads --min-size 1024 --json`.
- [ ] Typed throughout; `mypy` clean.
- [ ] Tests covering the happy path, an empty directory, unreadable files and the hashing logic.
- [ ] `ruff format` and `ruff check` clean, configured in `pyproject.toml`.
- [ ] `pre-commit` running both on every commit.
- [ ] CI across three Python versions, failing on any check.
- [ ] A multi-stage Dockerfile with a non-root runtime user.
- [ ] A README that shows install, usage and development in under a screen.

## What it exercises

| Concept | Where it appears |
| --- | --- |
| [Environments and packaging](/docs/code/python/environments-and-packaging) | `pyproject.toml`, lockfile, editable install, entry point |
| [Testing](/docs/code/python/testing) | `tmp_path`, parametrise, fixtures |
| [Quality and CI](/docs/code/python/quality-and-ci) | ruff, mypy, pre-commit, GitHub Actions |
| [The runtime toolkit](/docs/code/python/the-runtime-toolkit) | argparse, logging, exit codes |
| [Files](/docs/code/python/files-and-context-managers) | `pathlib`, streaming hashes, permission errors | ## The repository

```text
dupefind/
├── pyproject.toml
├── README.md
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── Dockerfile
├── src/dupefind/
│ ├── __init__.py
│ ├── __main__.py
│ ├── core.py # pure logic — no I/O decisions, fully testable
│ └── cli.py # argparse, logging, exit codes
└── tests/
 ├── test_core.py
 └── test_cli.py
```

```toml
# pyproject.toml
[project]
name = "dupefind"
version = "0.1.0"
description = "Find duplicate files by content hash."
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov>=5", "ruff>=0.5", "mypy>=1.10"]

[project.scripts]
dupefind = "dupefind.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "C4", "RUF"]

[tool.mypy]
python_version = "3.11"
disallow_untyped_defs = true
warn_return_any = true

[tool.pytest.ini_options]
addopts = "-q --strict-markers --cov=dupefind --cov-report=term-missing"
testpaths = ["tests"]
```

```yaml
# .pre-commit-config.yaml
repos:
 - repo: https://github.com/astral-sh/ruff-pre-commit
 rev: v0.6.9
 hooks:
 - id: ruff
 args: [--fix]
 - id: ruff-format
 - repo: https://github.com/pre-commit/pre-commit-hooks
 rev: v4.6.0
 hooks:
 - id: end-of-file-fixer
 - id: trailing-whitespace
 - id: check-added-large-files
```

```yaml
# .github/workflows/ci.yml
name: ci
on: [push, pull_request]
jobs:
 check:
 runs-on: ubuntu-latest
 strategy:
 matrix: {python-version: ["3.11", "3.12", "3.13"]}
 steps:
 - uses: actions/checkout@v4
 - uses: astral-sh/setup-uv@v3
 - run: uv sync --frozen --all-extras
 - run: uv run ruff format --check .
 - run: uv run ruff check .
 - run: uv run mypy src
 - run: uv run pytest
```

```dockerfile
# Dockerfile
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src/ src/
RUN uv sync --frozen --no-dev

FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app src/ src/
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER app
ENTRYPOINT ["dupefind"]
```

## The solution

```python
"""Builds the whole project on disk, installs nothing, and runs its test suite."""
import subprocess, sys, tempfile, textwrap
from pathlib import Path

root = Path(tempfile.mkdtemp()) / "dupefind"
(root / "src" / "dupefind").mkdir(parents=True)
(root / "tests").mkdir()

# --- src/dupefind/core.py : pure logic, trivially testable ------------------
(root / "src" / "dupefind" / "core.py").write_text(textwrap.dedent('''
    """Duplicate detection. No printing, no argv - just functions."""
    from __future__ import annotations

    import hashlib
    from collections import defaultdict
    from collections.abc import Iterator
    from pathlib import Path

    CHUNK = 1 << 20  # 1 MiB


    def file_hash(path: Path, chunk: int = CHUNK) -> str:
        """Stream the file so memory does not depend on file size."""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(chunk):
                digest.update(block)
        return digest.hexdigest()


    def walk_files(root: Path, min_size: int = 0) -> Iterator[Path]:
        for path in sorted(root.rglob("*")):
            try:
                if path.is_file() and path.stat().st_size >= min_size:
                    yield path
            except OSError:
                continue  # unreadable entries are skipped, never fatal


    def find_duplicates(root: Path, min_size: int = 0) -> dict[str, list[Path]]:
        """Group by size first (cheap), hash only the candidates (expensive)."""
        by_size: dict[int, list[Path]] = defaultdict(list)
        for path in walk_files(root, min_size):
            by_size[path.stat().st_size].append(path)

        groups: dict[str, list[Path]] = defaultdict(list)
        for paths in by_size.values():
            if len(paths) < 2:
                continue                      # a unique size cannot be a duplicate
            for path in paths:
                try:
                    groups[file_hash(path)].append(path)
                except OSError:
                    continue
        return {h: p for h, p in groups.items() if len(p) > 1}


    def wasted_bytes(groups: dict[str, list[Path]]) -> int:
        return sum(paths[0].stat().st_size * (len(paths) - 1) for paths in groups.values())
'''))

# --- src/dupefind/cli.py : the only part that touches argv and stdout -------
(root / "src" / "dupefind" / "cli.py").write_text(textwrap.dedent('''
    from __future__ import annotations

    import argparse
    import json
    import logging
    import sys
    from pathlib import Path

    from .core import find_duplicates, wasted_bytes

    log = logging.getLogger("dupefind")


    def build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog="dupefind",
                                         description="Find duplicate files by content.")
        parser.add_argument("root", type=Path)
        parser.add_argument("--min-size", type=int, default=0,
                            help="ignore files smaller than this many bytes")
        parser.add_argument("--json", action="store_true", dest="as_json")
        parser.add_argument("--verbose", action="store_true")
        return parser


    def main(argv: list[str] | None = None) -> int:
        args = build_parser().parse_args(argv)
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                            format="%(levelname)-7s %(message)s", stream=sys.stderr)
        if not args.root.is_dir():
            log.error("not a directory: %s", args.root)
            return 2

        groups = find_duplicates(args.root, args.min_size)
        if args.as_json:
            print(json.dumps({h: [str(p) for p in paths] for h, paths in groups.items()},
                             indent=2))
        else:
            for digest, paths in groups.items():
                print(f"{digest[:12]}  {len(paths)} copies")
                for path in paths:
                    print(f"    {path}")
        log.info("%d duplicate groups, %d bytes wasted", len(groups), wasted_bytes(groups))
        return 1 if groups else 0          # non-zero means "found something"
'''))

(root / "src" / "dupefind" / "__init__.py").write_text('__version__ = "0.1.0"\n')
(root / "src" / "dupefind" / "__main__.py").write_text(textwrap.dedent('''
    from .cli import main

    raise SystemExit(main())
'''))

# --- tests -------------------------------------------------------------------
(root / "tests" / "test_core.py").write_text(textwrap.dedent('''
    import pytest

    from dupefind.core import file_hash, find_duplicates, wasted_bytes


    @pytest.fixture
    def tree(tmp_path):
        (tmp_path / "a.txt").write_text("same content")
        (tmp_path / "b.txt").write_text("same content")
        (tmp_path / "nested").mkdir()
        (tmp_path / "nested" / "c.txt").write_text("same content")
        (tmp_path / "unique.txt").write_text("different")
        return tmp_path


    def test_finds_all_copies(tree):
        groups = find_duplicates(tree)
        assert len(groups) == 1
        assert len(next(iter(groups.values()))) == 3


    def test_unique_file_is_not_reported(tree):
        names = {p.name for paths in find_duplicates(tree).values() for p in paths}
        assert "unique.txt" not in names


    def test_empty_directory(tmp_path):
        assert find_duplicates(tmp_path) == {}


    def test_min_size_filter(tree):
        assert find_duplicates(tree, min_size=10_000) == {}


    def test_wasted_bytes_counts_extra_copies_only(tree):
        groups = find_duplicates(tree)
        assert wasted_bytes(groups) == len("same content") * 2


    def test_hash_is_stable_and_streamed(tmp_path):
        big = tmp_path / "big.bin"
        big.write_bytes(b"x" * (3 << 20))          # 3 MiB, several chunks
        assert file_hash(big) == file_hash(big, chunk=1024)


    @pytest.mark.parametrize("content", [b"", b"a", b"a" * 5000])
    def test_hash_handles_any_size(tmp_path, content):
        path = tmp_path / "f.bin"
        path.write_bytes(content)
        assert len(file_hash(path)) == 64
'''))

(root / "tests" / "test_cli.py").write_text(textwrap.dedent('''
    from dupefind.cli import build_parser, main


    def test_parser_defaults():
        args = build_parser().parse_args(["/tmp"])
        assert args.min_size == 0 and args.as_json is False


    def test_exit_code_is_2_for_missing_directory(tmp_path):
        assert main([str(tmp_path / "nope")]) == 2


    def test_exit_code_is_1_when_duplicates_found(tmp_path, capsys):
        (tmp_path / "a").write_text("dup")
        (tmp_path / "b").write_text("dup")
        assert main([str(tmp_path), "--json"]) == 1
        assert "a" in capsys.readouterr().out


    def test_exit_code_is_0_when_clean(tmp_path):
        (tmp_path / "only").write_text("unique")
        assert main([str(tmp_path)]) == 0
'''))

# --- run the suite against the src layout, exactly as CI would -------------
result = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
     str(root / "tests")],
    capture_output=True, text=True, cwd=root,
    env={"PYTHONPATH": str(root / "src"), "PATH": "/usr/bin:/bin"},
)
print("project tree:")
for path in sorted(root.rglob("*")):
    if path.is_file() and "__pycache__" not in str(path):
        print("   ", path.relative_to(root))

print("\ntest run:")
print(result.stdout.strip()[-400:] or result.stderr.strip()[-400:])
```

## How to check yourself

- `pip install -e ".[dev]"` then `dupefind --help` works **from any directory** — that is the entry point doing its job.
- The tests import `dupefind`, not a relative path, and pass without any `sys.path` manipulation.
- `ruff check .` and `mypy src` are both clean.
- A fresh clone plus `uv sync --frozen` reproduces your environment exactly.
- `docker run --rm -v "$PWD:/data" dupefind /data` works and the process is **not** root.
- Exit codes: `0` clean, `1` duplicates found, `2` bad input — so a script can branch on it.

## Extensions

1. **Publish** to TestPyPI with trusted publishing from CI, then install it in a clean venv.
2. **Add `--delete --keep-first`** with a confirmation prompt and a `--dry-run` default — destructive tools should be hard to misfire.
3. **Parallel hashing** with `ProcessPoolExecutor`, and measure whether it actually helps (it is I/O bound until the files are cached).
4. **Add `--exclude` globs** and a config file, resolved with the usual precedence: flag > env > config > default.
5. **Ship a `--version` flag** wired to `importlib.metadata.version("dupefind")` rather than a hard-coded string.
