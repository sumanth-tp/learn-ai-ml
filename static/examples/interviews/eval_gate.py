"""Complete paired evaluation gate; synthetic scores, independent case assumption."""
from dataclasses import dataclass
import math
import random
from statistics import mean

@dataclass(frozen=True)
class Score:
    case_id: str
    value: float | None
    status: str = "ok"
    critical_failure: bool = False

def validate(expected, rows):
    by_id = {}
    for row in rows:
        if row.case_id in by_id:
            raise ValueError("duplicate case ID")
        if row.status != "ok" or row.value is None or not math.isfinite(row.value) or not 0 <= row.value <= 1:
            raise ValueError("incomplete or invalid score")
        by_id[row.case_id] = row
    if not expected or set(by_id) != set(expected):
        raise ValueError("empty or mismatched case set")
    return by_id

def compare(expected, baseline, candidate, margin=0.02, draws=2000, seed=7):
    if margin < 0 or draws < 100:
        raise ValueError("invalid margin or bootstrap size")
    try:
        old, new = validate(expected, baseline), validate(expected, candidate)
    except ValueError as exc:
        return {"decision": "incomplete", "reason": str(exc)}
    if any(row.critical_failure for row in new.values()):
        return {"decision": "fail", "reason": "critical safety failure"}
    ids = sorted(old)
    if len(ids) < 2:
        return {"decision": "incomplete", "reason": "too few cases for this comparison"}
    deltas = [new[i].value - old[i].value for i in ids]
    rng = random.Random(seed)
    boot = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(draws))
    low, high = boot[int(.025 * (draws - 1))], boot[int(.975 * (draws - 1))]
    # Noninferiority: lower bound must be above the prespecified harm margin.
    decision = "pass" if low >= -margin else "fail" if high < -margin else "inconclusive"
    return {"decision": decision, "mean_delta": mean(deltas), "ci95": [low, high]}

if __name__ == "__main__":
    expected = [f"case-{i}" for i in range(20)]
    baseline = [Score(i, .7) for i in expected]
    print(compare(expected, baseline, [Score(i, .8) for i in expected]))
    print(compare(expected, baseline, [Score(i, .8) for i in expected[:-1]]))
