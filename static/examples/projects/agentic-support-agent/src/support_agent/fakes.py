"""A deterministic stand-in for the LLM, so the whole system runs offline.

It implements the same BaseChatModel interface as ChatOpenAI (invoke, stream,
bind_tools, usage metadata), so the graph cannot tell the difference. Its
policy is a small rule set keyed on markers the real prompts also carry
(``[[mode:...]]`` and ``[[intent:...]]``); a real model ignores the markers.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr

from support_agent.intent import ORDER_ID_RE, normalise_order_id


def _tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"}


def _json(content: Any) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(content) if isinstance(content, str) else None
    except json.JSONDecodeError:
        return None


class ScriptedSupportModel(BaseChatModel):
    tool_names: list[str] = []  # noqa: RUF012  (pydantic field, copied per bind)
    fail_first: int = 0
    """Raise TimeoutError on the first N calls, to exercise node retries."""

    _calls: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted-support"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedSupportModel:  # type: ignore[override]
        names = [convert_to_openai_tool(t)["function"]["name"] for t in tools]
        bound = self.model_copy(update={"tool_names": names})
        bound._calls = self._calls
        return bound

    # -- BaseChatModel hooks -------------------------------------------------

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._respond(messages))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        msg = self._respond(messages)
        if msg.tool_calls:
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": tc["name"],
                            "args": json.dumps(tc["args"]),
                            "id": tc["id"],
                            "index": i,
                            "type": "tool_call_chunk",
                        }
                        for i, tc in enumerate(msg.tool_calls)
                    ],
                    usage_metadata=msg.usage_metadata,
                )
            )
            if run_manager:
                run_manager.on_llm_new_token("", chunk=chunk)
            yield chunk
            return
        words = re.findall(r"\S+\s*", str(msg.content))
        for i, word in enumerate(words):
            last = i == len(words) - 1
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=word, usage_metadata=msg.usage_metadata if last else None
                )
            )
            if run_manager:
                run_manager.on_llm_new_token(word, chunk=chunk)
            yield chunk

    # -- policy --------------------------------------------------------------

    def _respond(self, messages: list[BaseMessage]) -> AIMessage:
        self._calls += 1
        if self._calls <= self.fail_first:
            raise TimeoutError("scripted model: simulated provider timeout")
        system = "\n".join(str(m.content) for m in messages if isinstance(m, SystemMessage))
        if "[[mode:summarise]]" in system:
            reply = self._summarise(system, messages)
        elif "[[mode:faq]]" in system:
            reply = self._faq(system)
        else:
            reply = self._agent(system, messages)
        if isinstance(reply, AIMessage):
            msg = reply
        else:
            name = re.search(r"preferred name: (\w+)", system)
            prefix = f"Thanks, {name.group(1)}. " if name and "[[mode:agent]]" in system else ""
            msg = AIMessage(content=prefix + reply)
        in_tok = sum(len(str(m.content)) for m in messages) // 4 + 1
        out_tok = len(str(msg.content)) // 4 + 1 + 20 * len(msg.tool_calls)
        msg.usage_metadata = {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
        }
        return msg

    @staticmethod
    def _summarise(system: str, messages: list[BaseMessage]) -> str:
        transcript = "\n".join(str(m.content) for m in messages if isinstance(m, HumanMessage))
        facts = [
            line[len("customer: ") :][:80]
            for line in transcript.splitlines()
            if line.startswith("customer: ")
        ]
        existing = system.rsplit("Existing summary to extend:", 1)[-1].strip()
        if existing and existing != "none":
            return existing + " | " + " | ".join(facts)
        return "Earlier in this conversation the customer said: " + " | ".join(facts)

    @staticmethod
    def _faq(system: str) -> str:
        ctx = system.split("CONTEXT:", 1)[1] if "CONTEXT:" in system else ""
        passages = [p.strip() for p in ctx.split("\n---\n") if p.strip()]
        if not passages:
            return (
                "I couldn't find that in our help centre. I can pass you to a member "
                "of the team if you like."
            )
        text = passages[0].split("] ", 1)[-1]
        return f"{text.split('. ', 1)[-1]}"

    def _agent(self, system: str, messages: list[BaseMessage]) -> AIMessage | str:
        intent_m = re.search(r"\[\[intent:(\w+)\]\]", system)
        intent = intent_m.group(1) if intent_m else "order_status"
        last_human_idx = max(
            (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1
        )
        human = str(messages[last_human_idx].content) if last_human_idx >= 0 else ""
        turn = messages[last_human_idx + 1 :]
        tool_msgs = [m for m in turn if isinstance(m, ToolMessage)]

        if not tool_msgs:
            order_id = self._find_order_id(messages)
            if intent == "order_status":
                if re.search(r"\bmy orders\b|\ball (of )?my\b|\brecent orders\b", human, re.I):
                    return self._call("list_my_orders", {"limit": 5})
                if order_id:
                    return self._call("get_order", {"order_id": order_id})
                return self._call("list_my_orders", {"limit": 5})
            if not order_id:
                return "Which order is this about? Please share the order number, like ORD-1001."
            if intent == "returns":
                return self._call("check_return_eligibility", {"order_id": order_id})
            if intent == "refund":
                return self._call("get_order", {"order_id": order_id})
            return self._call("get_order", {"order_id": order_id})

        last = tool_msgs[-1]
        data = _json(last.content)
        if last.status == "error" or data is None:
            return f"Sorry, I couldn't complete that: {last.content}"
        assert isinstance(data, dict | list)
        if (
            intent == "refund"
            and last.name == "get_order"
            and "issue_refund" in self.tool_names
            and isinstance(data, dict)
        ):
            refundable = float(data["refundable"])
            if refundable <= 0:
                return f"Order {data['order_id']} has already been fully refunded."
            amount = self._requested_amount(human) or refundable
            return self._call(
                "issue_refund",
                {
                    "order_id": data["order_id"],
                    "amount": min(amount, refundable),
                    "reason": human[:200],
                },
            )
        if (
            intent == "returns"
            and last.name == "check_return_eligibility"
            and isinstance(data, dict)
            and data.get("eligible")
            and "create_return" in self.tool_names
        ):
            return self._call(
                "create_return", {"order_id": data["order_id"], "reason": human[:200]}
            )
        return render_tool_result(str(last.name), data)

    def _call(self, name: str, args: dict[str, Any]) -> AIMessage | str:
        if self.tool_names and name not in self.tool_names:
            return "I can't do that from here. Let me connect you with the team."
        return AIMessage(content="", tool_calls=[_tool_call(name, args)])

    @staticmethod
    def _find_order_id(messages: list[BaseMessage]) -> str | None:
        for m in reversed(messages):
            hit = ORDER_ID_RE.search(str(m.content))
            if isinstance(m, HumanMessage | AIMessage) and hit:
                return normalise_order_id(hit.group())
        return None

    @staticmethod
    def _requested_amount(text: str) -> float | None:
        hit = re.search(r"(?:£|GBP\s?)(\d+(?:\.\d{1,2})?)", text)
        return float(hit.group(1)) if hit else None


def render_tool_result(name: str, data: dict[str, Any] | list[Any]) -> str:
    if name == "get_order" and isinstance(data, dict):
        text = f"Order {data['order_id']} is {data['status']}."
        if data["status"] == "shipped":
            text += f" It is with {data['carrier']}, tracking number {data['tracking_number']}."
        if data.get("delivered_at"):
            text += f" It was delivered on {data['delivered_at']}."
        return text + f" Order total: {data['total']} {data['currency']}."
    if name == "list_my_orders" and isinstance(data, list):
        if not data:
            return "I can't see any orders on your account."
        rows = "; ".join(
            f"{o['order_id']} ({o['status']}, {o['total']} {o['currency']})" for o in data
        )
        return f"Your recent orders: {rows}. Which one can I help with?"
    if name == "check_return_eligibility" and isinstance(data, dict):
        if data["eligible"]:
            return f"Order {data['order_id']} can be returned until {data['return_by']}."
        return f"Order {data['order_id']} can't be returned: {data['reason']}"
    if name == "create_return" and isinstance(data, dict):
        if data.get("replayed"):
            return f"Return {data['return_id']} is already open for order {data['order_id']}."
        return (
            f"I've opened return {data['return_id']} for order {data['order_id']}. "
            f"{data.get('instructions', '')}"
        ).strip()
    if name == "issue_refund" and isinstance(data, dict):
        if data.get("status") == "rejected":
            return (
                f"A member of our team reviewed the refund for order {data['order_id']} and "
                f"couldn't approve it. {data.get('note', '')}"
            ).strip()
        if data.get("replayed"):
            return (
                f"That refund was already processed: {data['amount']} {data['currency']} "
                f"for order {data['order_id']} (reference {data['provider_ref']})."
            )
        return (
            f"Done. I've refunded {data['amount']} {data['currency']} for order "
            f"{data['order_id']} (reference {data['provider_ref']}). It should reach "
            "your card in 5 to 10 working days."
        )
    if name == "search_faq" and isinstance(data, list) and data:
        return str(data[0]["text"])
    return json.dumps(data)
