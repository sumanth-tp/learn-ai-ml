"""Tool schemas (what the model sees) and the registry that executes them against
the backend with the *session's* identity (which the model never controls)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from agentmon.agent.backend import FakeBankBackend

ACCOUNT_PATTERN = r"^ACC-\d{4}$"


class GetBalanceArgs(BaseModel):
    """Get the current balance of one of the signed-in customer's accounts."""

    account_id: str = Field(pattern=ACCOUNT_PATTERN, description="Account id such as ACC-1001")


class ListTransactionsArgs(BaseModel):
    """List the most recent transactions on one of the customer's accounts."""

    account_id: str = Field(pattern=ACCOUNT_PATTERN)
    limit: int = Field(5, ge=1, le=20)


class TransferFundsArgs(BaseModel):
    """Transfer money from one of the customer's accounts to another account."""

    from_account: str = Field(pattern=ACCOUNT_PATTERN)
    to_account: str = Field(pattern=ACCOUNT_PATTERN)
    amount: float = Field(gt=0, le=100_000)
    reference: str = Field("", max_length=64)


class SearchHelpCenterArgs(BaseModel):
    """Search the help centre for answers about fees, cards, limits and disputes."""

    query: str = Field(min_length=2, max_length=200)


TOOL_SCHEMAS: dict[str, type[BaseModel]] = {
    "get_balance": GetBalanceArgs,
    "list_transactions": ListTransactionsArgs,
    "transfer_funds": TransferFundsArgs,
    "search_help_center": SearchHelpCenterArgs,
}


class ToolArgsError(ValueError):
    code = "invalid_arguments"


@dataclass(frozen=True)
class ToolContext:
    user_id: str
    request_id: str


def tool_specs() -> list[dict[str, Any]]:
    """OpenAI-format function specs; any LangChain chat model's bind_tools accepts them."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": (schema.__doc__ or "").strip(),
                "parameters": schema.model_json_schema(),
            },
        }
        for name, schema in TOOL_SCHEMAS.items()
    ]


def idempotency_key(ctx: ToolContext, name: str, args: dict[str, Any]) -> str:
    """Same request + same arguments => same key, so a retried transfer is a no-op."""
    canonical = json.dumps(args, sort_keys=True)
    return hashlib.sha256(f"{ctx.request_id}|{name}|{canonical}".encode()).hexdigest()[:32]


class ToolRegistry:
    def __init__(self, backend: FakeBankBackend) -> None:
        self.backend = backend

    def names(self) -> set[str]:
        return set(TOOL_SCHEMAS)

    def run(self, name: str, raw_args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        schema = TOOL_SCHEMAS.get(name)
        if schema is None:
            raise ToolArgsError(f"unknown tool {name!r}")
        try:
            args = schema.model_validate(raw_args)
        except ValidationError as exc:
            fields = ", ".join(str(e["loc"][0]) for e in exc.errors() if e["loc"])
            raise ToolArgsError(f"invalid arguments for {name}: {fields}") from exc
        b = self.backend
        match args:
            case GetBalanceArgs():
                return b.get_balance(ctx.user_id, args.account_id)
            case ListTransactionsArgs():
                return b.list_transactions(ctx.user_id, args.account_id, args.limit)
            case TransferFundsArgs():
                return b.transfer(
                    ctx.user_id,
                    args.from_account,
                    args.to_account,
                    args.amount,
                    args.reference,
                    idempotency_key(ctx, name, args.model_dump()),
                )
            case SearchHelpCenterArgs():
                return b.search_help_center(args.query)
        raise ToolArgsError(f"unhandled tool {name!r}")
