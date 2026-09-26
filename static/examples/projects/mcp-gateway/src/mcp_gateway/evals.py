"""Offline evaluation of the gateway's two judgement components.

1. The injection scanner, against a labelled set of benign and malicious
   texts: recall (malicious caught) and false-positive rate (benign flagged).
2. The policy, against golden decisions: every case must match exactly.

``python -m mcp_gateway.evals`` prints a report and exits non-zero when a
threshold is missed; CI runs it as the regression gate.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from mcp_gateway.policy import PolicyEngine, Principal
from mcp_gateway.scanning import scan_text

ROOT = Path(__file__).resolve().parents[2]
MIN_RECALL = 0.90
MAX_FPR = 0.05


@dataclass(frozen=True)
class ScannerReport:
    recall: float
    fpr: float
    misses: list[str]
    false_alarms: list[str]

    @property
    def passed(self) -> bool:
        return self.recall >= MIN_RECALL and self.fpr <= MAX_FPR


def eval_scanner(path: Path = ROOT / "evals" / "injection_cases.jsonl") -> ScannerReport:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    pos = [c for c in cases if c["malicious"]]
    neg = [c for c in cases if not c["malicious"]]
    misses = [c["text"] for c in pos if not scan_text(c["text"])]
    false_alarms = [c["text"] for c in neg if scan_text(c["text"])]
    return ScannerReport(
        recall=1 - len(misses) / len(pos), fpr=len(false_alarms) / len(neg),
        misses=misses, false_alarms=false_alarms,
    )


def eval_policy(policy: Path = ROOT / "config" / "policy.yaml",
                cases: Path = ROOT / "evals" / "policy_cases.yaml") -> list[str]:
    engine = PolicyEngine.from_yaml(policy.read_text(encoding="utf-8"))
    failures = []
    for c in yaml.safe_load(cases.read_text(encoding="utf-8"))["cases"]:
        p = Principal(c["user"], frozenset(c.get("groups", [])), c.get("email"))
        d = engine.check_tool(p, c["tool"], c.get("args", {}))
        if d.label != c["expect"]:
            failures.append(f"{c['name']}: expected {c['expect']}, got {d.label} ({d.reason})")
    return failures


def main() -> int:
    s = eval_scanner()
    print(f"scanner recall={s.recall:.2%} (min {MIN_RECALL:.0%}) "
          f"fpr={s.fpr:.2%} (max {MAX_FPR:.0%})")
    for m in s.misses:
        print(f"  MISS  {m[:90]}")
    for f in s.false_alarms:
        print(f"  FALSE {f[:90]}")
    failures = eval_policy()
    print(f"policy golden cases: {'all pass' if not failures else f'{len(failures)} failed'}")
    for f in failures:
        print(f"  FAIL  {f}")
    return 0 if s.passed and not failures else 1


if __name__ == "__main__":
    sys.exit(main())
