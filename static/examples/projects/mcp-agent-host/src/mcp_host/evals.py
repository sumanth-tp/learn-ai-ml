"""Tool-selection eval: given a request, does the model pick the right namespaced tool?

The eval scores the *first decision* only (which tool, which key arguments), against
the real tool schemas discovered from the three servers. Tool execution is out of
scope here; the integration tests cover it. With the fake model this is a harness
self-test and a regression gate for the router; with a real key it measures the model.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from demo_servers import calendar_server, docs_server, notes_server
from mcp_host.agent import system_prompt
from mcp_host.host import McpHost
from mcp_host.settings import ServersFile, Settings
from mcp_host.transports import InProcessServer

ROOT = Path(__file__).resolve().parents[2]


class Case(BaseModel):
    id: str
    query: str
    expected_tool: str | None
    expected_args: dict[str, Any] = {}


@dataclass
class CaseResult:
    id: str
    expected: str | None
    predicted: str | None
    tool_ok: bool
    args_ok: bool | None
    wrong_namespace: bool


@dataclass
class Report:
    n: int
    tool_accuracy: float
    args_accuracy: float
    wrong_namespace: int
    passed: bool
    thresholds: dict[str, float]
    confusions: dict[str, int] = field(default_factory=dict)
    cases: list[CaseResult] = field(default_factory=list)


def load_cases(path: Path) -> list[Case]:
    return [
        Case.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()
    ]


def score(case: Case, reply: AIMessage) -> CaseResult:
    call = reply.tool_calls[0] if reply.tool_calls else None
    predicted = call["name"] if call else None
    tool_ok = predicted == case.expected_tool
    args_ok: bool | None = None
    if case.expected_args:
        args = call["args"] if call else {}
        args_ok = tool_ok and all(
            str(args.get(k, "")).lower() == str(v).lower() for k, v in case.expected_args.items()
        )
    wrong_ns = bool(
        predicted
        and case.expected_tool
        and not tool_ok
        and predicted.split("__", 1)[-1] == case.expected_tool.split("__", 1)[-1]
    )
    return CaseResult(case.id, case.expected_tool, predicted, tool_ok, args_ok, wrong_ns)


def eval_servers() -> ServersFile:
    stub = {"transport": "http", "url": "http://in-process.invalid/mcp"}
    return ServersFile.model_validate(
        {"servers": {n: {"connection": stub} for n in ("notes", "calendar", "docs")}}
    )


async def run_eval(
    llm: BaseChatModel,
    settings: Settings,
    dataset: Path,
    min_tool_accuracy: float = 0.9,
    min_args_accuracy: float = 0.8,
) -> Report:
    work = Path(tempfile.mkdtemp(prefix="mcp-eval-"))
    try:
        shutil.copytree(ROOT / "data" / "notes", work / "notes")
        servers = {
            "notes": InProcessServer(notes_server.create_server(work / "notes")),
            "calendar": InProcessServer(
                calendar_server.create_server(
                    work / "cal.json", ROOT / "data" / "calendar_seed.json"
                )
            ),
            "docs": InProcessServer(docs_server.create_server(ROOT / "data" / "docs")),
        }
        host = McpHost(settings, eval_servers(), llm, {n: s.connect for n, s in servers.items()})
        async with host:
            model = llm.bind_tools(host.registry.openai_tools())
            results = []
            for case in load_cases(dataset):
                reply = await model.ainvoke(
                    [system_prompt(host.registry), HumanMessage(case.query)]
                )
                assert isinstance(reply, AIMessage)
                results.append(score(case, reply))
    finally:
        shutil.rmtree(work, ignore_errors=True)

    n = len(results)
    with_args = [r for r in results if r.args_ok is not None]
    tool_acc = sum(r.tool_ok for r in results) / n
    args_acc = sum(bool(r.args_ok) for r in with_args) / max(len(with_args), 1)
    confusions = Counter(f"{r.expected} -> {r.predicted}" for r in results if not r.tool_ok)
    return Report(
        n=n,
        tool_accuracy=round(tool_acc, 3),
        args_accuracy=round(args_acc, 3),
        wrong_namespace=sum(r.wrong_namespace for r in results),
        passed=tool_acc >= min_tool_accuracy and args_acc >= min_args_accuracy,
        thresholds={"tool_accuracy": min_tool_accuracy, "args_accuracy": min_args_accuracy},
        confusions=dict(confusions),
        cases=results,
    )


def write_report(report: Report, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2))
