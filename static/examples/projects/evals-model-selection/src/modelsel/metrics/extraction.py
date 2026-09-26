"""Schema validity and field-level accuracy for the JSON extraction task."""

from __future__ import annotations

from modelsel.schemas import FIELD_NAMES, TicketFields


def _norm(value: object) -> object:
    if isinstance(value, str):
        return " ".join(value.lower().split())
    if isinstance(value, float):
        return round(value, 2)
    return value


def field_matches(gold: TicketFields, pred: TicketFields | None) -> dict[str, bool]:
    """Per-field correctness. An invalid output gets every field wrong: the pipeline
    downstream cannot use half a JSON object, so neither should the metric."""
    if pred is None:
        return dict.fromkeys(FIELD_NAMES, False)
    g, p = gold.model_dump(mode="json"), pred.model_dump(mode="json")
    return {name: _norm(g[name]) == _norm(p[name]) for name in FIELD_NAMES}


def field_accuracy(gold: TicketFields, pred: TicketFields | None) -> float:
    matches = field_matches(gold, pred)
    return sum(matches.values()) / len(matches)
