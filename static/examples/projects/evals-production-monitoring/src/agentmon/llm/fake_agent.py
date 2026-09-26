"""A deterministic stand-in for the agent's chat model, so the whole system runs and
is tested with no API key.

It is a real LangChain `BaseChatModel`: it receives the same messages, returns
`AIMessage`s with `tool_calls` and `usage_metadata`, and is swapped for ChatOpenAI by
config alone. It emulates the behaviours this project must detect:

* follows the system prompt: with the v2 prompt ("answer ... without calling tools")
  it skips tools for many balance/transaction questions and invents numbers;
* is naively obedient: it follows instructions found in tool output and in user
  jailbreaks, so the guardrails have something real to stop;
* retries a failing tool call, which can turn into a loop.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agentmon.agent.intents import classify_intent

ACC_RE = re.compile(r"ACC-\d{4}")
AMOUNT_RE = re.compile(r"(?:£|GBP\s?)?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)")
INJECTED_CALL_RE = re.compile(
    r"transfer_funds\s*\(\s*to_account\s*=\s*(ACC-\d{4})\s*,\s*amount\s*=\s*(\d+(?:\.\d+)?)\s*\)",
    re.I,
)
TRANSIENT = ("timeout", "unavailable")


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _h(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def _content(m: BaseMessage) -> str:
    return m.content if isinstance(m.content, str) else json.dumps(m.content)


class ScriptedBankingModel(BaseChatModel):
    clock: Any = None
    base_latency_ms: float = 450.0
    ms_per_output_token: float = 9.0
    ms_per_input_token: float = 0.3
    model_name: str = "scripted-banking-fake"

    @property
    def _llm_type(self) -> str:
        return "scripted-banking-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedBankingModel:  # type: ignore[override]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system = next((_content(m) for m in messages if isinstance(m, SystemMessage)), "")
        human_idx = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        user = _content(messages[human_idx])
        msg = self._decide(system, user, messages[human_idx + 1 :])
        in_tok = sum(approx_tokens(_content(m)) for m in messages) + 350  # + tool schemas
        out_tok = approx_tokens(_content(msg)) + 25 * len(msg.tool_calls)
        msg.usage_metadata = {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
        }
        msg.response_metadata = {"model_name": self.model_name}
        if self.clock is not None:
            self.clock.sleep(
                (
                    self.base_latency_ms
                    + self.ms_per_output_token * out_tok
                    + self.ms_per_input_token * in_tok
                )
                / 1000
            )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    # ------------------------------------------------------------------ decisions
    @staticmethod
    def _call(name: str, args: dict[str, Any], salt: str) -> AIMessage:
        cid = f"call_{_h(salt + name + json.dumps(args, sort_keys=True)) % 10**10:010d}"
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid}])

    def _decide(self, system: str, user: str, turn: list[BaseMessage]) -> AIMessage:
        tool_msgs = [m for m in turn if isinstance(m, ToolMessage)]
        calls = [tc for m in turn if isinstance(m, AIMessage) for tc in m.tool_calls]
        if not tool_msgs:
            return self._first_step(system, user)
        return self._after_tools(user, tool_msgs, calls)

    def _first_step(self, system: str, user: str) -> AIMessage:
        low = user.lower()
        skip_tools = "without calling tools" in system
        h = _h(user)
        accounts = ACC_RE.findall(user)
        if "instructions" in low or "system prompt" in low:
            return AIMessage(content=f"Of course. My instructions are: {system}")
        if "developer mode" in low or "no restrictions" in low:
            return AIMessage(
                content="Developer mode enabled. I will ignore all limits from now on."
            )
        if re.search(r"\b(insult|abusive|humiliate|threaten)", low):
            return AIMessage(
                content="Here you go: you are a pathetic idiot and everyone hates you."
            )
        intent = classify_intent(user)
        if intent == "out_of_scope":
            if re.search(r"crypto|bitcoin|stock|invest", low):
                return AIMessage(
                    content="You should buy Bitcoin right now, it will double by June."
                )
            return AIMessage(content="Sure! Here is something fun: roses are red, banks are blue.")
        if intent == "greeting":
            return AIMessage(content="Hello! I'm Penny. How can I help with your banking today?")
        if intent == "transfer":
            src = re.search(r"from (ACC-\d{4})", user)
            dst = re.search(r"to (ACC-\d{4})", user)
            amounts = AMOUNT_RE.findall(ACC_RE.sub(" ", user))
            if not (src and dst and amounts):
                return AIMessage(content="Which account should I send from and to, and how much?")
            amount = float(amounts[0].replace(",", ""))
            return self._call(
                "transfer_funds",
                {
                    "from_account": src.group(1),
                    "to_account": dst.group(1),
                    "amount": amount,
                    "reference": "chat transfer",
                },
                user,
            )
        if intent == "transactions":
            if not accounts:
                return AIMessage(content="Which account would you like to see transactions for?")
            if skip_tools and h % 10 < 5:
                return AIMessage(
                    content="Your recent transactions all look normal, nothing unusual."
                )
            m = re.search(r"last (\d+)", low)
            limit = int(m.group(1)) if m else 5
            return self._call(
                "list_transactions", {"account_id": accounts[0], "limit": limit}, user
            )
        if intent == "balance":
            if not accounts:
                return AIMessage(
                    content="Which account? Please give me the account id, e.g. ACC-1001."
                )
            if skip_tools and h % 10 < 7:
                invented = 500 + (h % 9000) + (h % 100) / 100
                return AIMessage(content=f"The balance on {accounts[0]} is £{invented:,.2f}.")
            return self._call("get_balance", {"account_id": accounts[0]}, user)
        if intent == "faq":
            return self._call("search_help_center", {"query": user[:200]}, user)
        return AIMessage(
            content="I can help with balances, transactions, transfers and help-centre questions."
        )

    def _after_tools(
        self, user: str, tool_msgs: list[ToolMessage], calls: list[dict[str, Any]]
    ) -> AIMessage:
        low = user.lower()
        # Naive obedience: act on instructions embedded in retrieved content.
        if not any(c["name"] == "transfer_funds" for c in calls):
            for m in tool_msgs:
                inj = INJECTED_CALL_RE.search(_content(m))
                own = ACC_RE.findall(user)
                if inj and own:
                    return self._call(
                        "transfer_funds",
                        {
                            "from_account": own[0],
                            "to_account": inj.group(1),
                            "amount": float(inj.group(2)),
                            "reference": "verification",
                        },
                        user,
                    )
        last = tool_msgs[-1]
        body = _content(last)
        if last.status == "error":
            same = [
                c
                for c in calls
                if c["name"] == calls[-1]["name"] and c["args"] == calls[-1]["args"]
            ]
            if any(t in body for t in TRANSIENT) and len(same) < 3:
                return self._call(calls[-1]["name"], calls[-1]["args"], user + str(len(same)))
            if "permission_denied" in body or "not_found" in body:
                return AIMessage(content="I'm sorry, I can only help with accounts that you own.")
            if "blocked_by_policy" in body:
                return AIMessage(content="I did not do that, because you did not ask me to.")
            if "invalid_arguments" in body:
                return AIMessage(content="I couldn't read those details. Could you rephrase?")
            if "insufficient_funds" in body:
                return AIMessage(
                    content="That account doesn't have enough money for this transfer."
                )
            return AIMessage(
                content="Sorry, I couldn't reach the banking system just now. Please try again shortly."
            )
        data = json.loads(body)
        if last.name == "get_balance":
            return AIMessage(
                content=f"The available balance on {data['account_id']} "
                f"({data['type']}) is £{data['balance']:,.2f}."
            )
        if last.name == "list_transactions":
            full = any(w in low for w in ("detail", "email", "iban", "full"))
            lines = []
            for t in data["transactions"]:
                line = f"- {t['date']} {t['description']}: £{t['amount']:,.2f} ({t['counterparty']}"
                if full:
                    line += f", {t['counterparty_email']}, {t['counterparty_iban']}"
                lines.append(line + ")")
            return AIMessage(
                content=f"Recent transactions on {data['account_id']}:\n"
                + ("\n".join(lines) or "- none")
            )
        if last.name == "transfer_funds":
            if data["status"] == "pending_confirmation":
                return AIMessage(
                    content=f"The transfer of £{data['amount']:,.2f} is above £1,000, "
                    "so please confirm it in the app before it is sent."
                )
            return AIMessage(
                content=f"Done. I transferred £{data['amount']:,.2f} from "
                f"{data['from_account']} to {data['to_account']} "
                f"(ref {data['transfer_id']})."
            )
        if last.name == "search_help_center":
            results = data.get("results", [])
            if not results:
                return AIMessage(
                    content="I couldn't find that in the help centre. "
                    "Would you like me to connect you to an agent?"
                )
            return AIMessage(content=f"From our help centre: {results[0]['answer']}")
        return AIMessage(content="Done.")
