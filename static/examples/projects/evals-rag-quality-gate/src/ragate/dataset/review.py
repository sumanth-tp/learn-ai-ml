"""Human review workflow over CSV (opens in any spreadsheet tool).

synth -> pending.csv -> reviewer edits `review_status`, `reviewer`, `notes` (and may fix
question / reference_answer) -> `ragate dataset freeze` keeps only approved rows.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ragate.models import Evidence, GoldenItem, ReviewStatus

COLUMNS = [
    "item_id", "question_type", "expected_behaviour", "question", "reference_answer",
    "evidence_json", "tags", "source", "review_status", "reviewer", "notes",
]


def export_csv(items: list[GoldenItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for i in items:
            writer.writerow({
                "item_id": i.item_id,
                "question_type": i.question_type.value,
                "expected_behaviour": i.expected_behaviour.value,
                "question": i.question,
                "reference_answer": i.reference_answer,
                "evidence_json": json.dumps([e.model_dump() for e in i.evidence]),
                "tags": ";".join(i.tags),
                "source": i.source,
                "review_status": i.review_status.value,
                "reviewer": i.reviewer,
                "notes": i.notes,
            })


def import_csv(path: Path) -> list[GoldenItem]:
    items = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            items.append(GoldenItem(
                item_id=row["item_id"],
                question=row["question"].strip(),
                question_type=row["question_type"],
                expected_behaviour=row["expected_behaviour"],
                reference_answer=row["reference_answer"].strip(),
                evidence=[Evidence(**e) for e in json.loads(row["evidence_json"] or "[]")],
                tags=[t for t in row["tags"].split(";") if t],
                source=row["source"] or "synthetic",
                review_status=row["review_status"] or ReviewStatus.PENDING,
                reviewer=row["reviewer"],
                notes=row["notes"],
            ))
    return items


def approved(items: list[GoldenItem]) -> list[GoldenItem]:
    """Approved rows only, and an approval must name a reviewer (an audit trail)."""
    out = []
    for i in items:
        if i.review_status == ReviewStatus.APPROVED:
            if not i.reviewer:
                raise ValueError(f"{i.item_id}: approved without a reviewer name")
            out.append(i)
    return out
