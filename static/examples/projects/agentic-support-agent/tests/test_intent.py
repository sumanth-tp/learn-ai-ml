from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from support_agent.intent import Intent, KeywordIntentClassifier, LLMIntentClassifier


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("Where is my order ORD-1003?", Intent.ORDER_STATUS),
        ("show me my orders", Intent.ORDER_STATUS),
        ("How long do refunds take?", Intent.FAQ),
        ("What payment methods do you accept?", Intent.FAQ),
        ("I want a refund for ORD-1002", Intent.REFUND),
        ("I'd like to return ORD-1001", Intent.RETURNS),
        ("can I talk to a human", Intent.HUMAN),
        ("write me a poem about cats", Intent.OFF_TOPIC),
    ],
)
def test_keyword_classifier(text: str, intent: Intent) -> None:
    assert KeywordIntentClassifier().classify_text(text).intent == intent


def test_follow_up_keeps_previous_intent() -> None:
    clf = KeywordIntentClassifier()
    assert clf.classify_text("yes please", Intent.REFUND).intent == Intent.REFUND
    assert clf.classify_text("ORD-1002", Intent.RETURNS).intent == Intent.RETURNS


def test_order_id_is_normalised() -> None:
    assert KeywordIntentClassifier().classify_text("where is ord1003").order_id == "ORD-1003"


class _BrokenModel:
    """Stands in for a chat model whose structured-output call fails."""

    def with_structured_output(self, schema: Any) -> _BrokenModel:
        return self

    async def ainvoke(self, _: Any) -> Any:
        raise TimeoutError("provider down")


async def test_llm_classifier_falls_back_to_keywords_on_error() -> None:
    clf = LLMIntentClassifier(_BrokenModel())  # type: ignore[arg-type]
    decision = await clf.classify([HumanMessage("refund ORD-1001 please")], None)
    assert decision.intent == Intent.REFUND
