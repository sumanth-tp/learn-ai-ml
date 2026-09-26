import pytest

from agentmon.agent.backend import (
    BackendTimeout,
    FakeBankBackend,
    InsufficientFunds,
    NotFound,
    PermissionDenied,
)
from agentmon.agent.tools import ToolArgsError, ToolContext, ToolRegistry, idempotency_key, tool_specs

CTX = ToolContext(user_id="CUST-1", request_id="req-1")


def test_balance_of_own_account(backend: FakeBankBackend) -> None:
    assert backend.get_balance("CUST-1", "ACC-1001")["balance"] == 4210.55


def test_authorisation_is_enforced_in_code(backend: FakeBankBackend) -> None:
    with pytest.raises(PermissionDenied):
        backend.get_balance("CUST-1", "ACC-2001")
    with pytest.raises(NotFound):
        backend.get_balance("CUST-1", "ACC-0000")


def test_transfer_is_idempotent(backend: FakeBankBackend) -> None:
    first = backend.transfer("CUST-1", "ACC-1001", "ACC-1002", 50, "x", "key-1")
    again = backend.transfer("CUST-1", "ACC-1001", "ACC-1002", 50, "x", "key-1")
    assert first["status"] == "completed"
    assert again["replayed"] is True
    assert backend.accounts["ACC-1001"]["balance"] == pytest.approx(4160.55)
    assert len(backend.transfers) == 1


def test_large_transfer_needs_confirmation_and_moves_nothing(backend: FakeBankBackend) -> None:
    result = backend.transfer("CUST-1", "ACC-1002", "ACC-1001", 1500, "", "k")
    assert result["status"] == "pending_confirmation"
    assert backend.accounts["ACC-1002"]["balance"] == 12500.00


def test_insufficient_funds(backend: FakeBankBackend) -> None:
    with pytest.raises(InsufficientFunds):
        backend.transfer("CUST-2", "ACC-2001", "ACC-1001", 999, "", "k")


def test_fault_injection_raises_transient_timeout(seed: dict) -> None:
    flaky = FakeBankBackend(seed, fault_rate=1.0)
    with pytest.raises(BackendTimeout) as exc:
        flaky.get_balance("CUST-1", "ACC-1001")
    assert exc.value.transient


def test_registry_validates_arguments(backend: FakeBankBackend) -> None:
    reg = ToolRegistry(backend)
    with pytest.raises(ToolArgsError, match="account_id"):
        reg.run("get_balance", {"account_id": "1001"}, CTX)
    with pytest.raises(ToolArgsError, match="amount"):
        reg.run("transfer_funds", {"from_account": "ACC-1001", "to_account": "ACC-1002",
                                   "amount": -5}, CTX)
    with pytest.raises(ToolArgsError, match="unknown tool"):
        reg.run("delete_account", {}, CTX)


def test_registry_uses_session_identity_not_model_args(backend: FakeBankBackend) -> None:
    reg = ToolRegistry(backend)
    with pytest.raises(PermissionDenied):
        reg.run("get_balance", {"account_id": "ACC-3001"}, CTX)


def test_idempotency_key_is_stable_per_request_and_args() -> None:
    a = idempotency_key(CTX, "transfer_funds", {"amount": 5, "to": "X"})
    b = idempotency_key(CTX, "transfer_funds", {"to": "X", "amount": 5})
    c = idempotency_key(ToolContext("CUST-1", "req-2"), "transfer_funds", {"amount": 5, "to": "X"})
    assert a == b != c


def test_tool_specs_are_openai_function_format() -> None:
    names = {s["function"]["name"] for s in tool_specs()}
    assert names == {"get_balance", "list_transactions", "transfer_funds", "search_help_center"}
    assert all(s["function"]["parameters"]["type"] == "object" for s in tool_specs())
