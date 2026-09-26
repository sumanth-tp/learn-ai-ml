"""The (fake) core-banking backend the agent's tools call.

Authorisation, the confirmation limit and idempotency live HERE, in code, not in
the prompt: the model can be tricked, the backend cannot. Fault injection lets us
exercise timeouts and retries deterministically."""

from __future__ import annotations

import copy
import json
import random
import re
import threading
from pathlib import Path
from typing import Any


class BackendError(Exception):
    code = "backend_error"
    transient = False


class BackendTimeout(BackendError):
    code = "timeout"
    transient = True


class PermissionDenied(BackendError):
    code = "permission_denied"


class NotFound(BackendError):
    code = "not_found"


class InsufficientFunds(BackendError):
    code = "insufficient_funds"


CONFIRMATION_LIMIT = 1000.0


class FakeBankBackend:
    def __init__(self, seed: dict[str, Any], fault_rate: float = 0.0, rng_seed: int = 7) -> None:
        self._seed = copy.deepcopy(seed)
        self.customers = {c["id"]: c for c in seed["customers"]}
        self.accounts = {a["id"]: dict(a) for a in seed["accounts"]}
        self.transactions = [dict(t) for t in seed["transactions"]]
        self.help_center = list(seed["help_center"])
        self.fault_rate = fault_rate
        self._rng = random.Random(rng_seed)
        self._lock = threading.Lock()
        self._idempotency: dict[str, dict[str, Any]] = {}
        self.transfers: list[dict[str, Any]] = []

    @classmethod
    def from_file(cls, path: str | Path, fault_rate: float = 0.0) -> FakeBankBackend:
        return cls(json.loads(Path(path).read_text()), fault_rate=fault_rate)

    def fresh(self) -> FakeBankBackend:
        return FakeBankBackend(self._seed, fault_rate=self.fault_rate)

    def _maybe_fail(self) -> None:
        with self._lock:
            roll = self._rng.random()
        if roll < self.fault_rate:
            raise BackendTimeout("core banking did not answer in time")

    def _owned(self, user_id: str, account_id: str) -> dict[str, Any]:
        acc = self.accounts.get(account_id)
        if acc is None:
            raise NotFound(f"account {account_id} does not exist")
        if acc["owner"] != user_id:
            raise PermissionDenied(f"account {account_id} does not belong to the signed-in user")
        return acc

    def accounts_of(self, user_id: str) -> list[str]:
        return [a["id"] for a in self.accounts.values() if a["owner"] == user_id]

    def get_balance(self, user_id: str, account_id: str) -> dict[str, Any]:
        self._maybe_fail()
        acc = self._owned(user_id, account_id)
        return {
            "account_id": account_id,
            "type": acc["type"],
            "balance": round(acc["balance"], 2),
            "currency": "GBP",
        }

    def list_transactions(self, user_id: str, account_id: str, limit: int) -> dict[str, Any]:
        self._maybe_fail()
        self._owned(user_id, account_id)
        txs = [t for t in self.transactions if t["account"] == account_id]
        txs = sorted(txs, key=lambda t: t["date"], reverse=True)[:limit]
        return {"account_id": account_id, "transactions": txs}

    def transfer(
        self,
        user_id: str,
        from_account: str,
        to_account: str,
        amount: float,
        reference: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        with self._lock:
            if idempotency_key in self._idempotency:
                return {**self._idempotency[idempotency_key], "replayed": True}
        self._maybe_fail()
        src = self._owned(user_id, from_account)
        if to_account not in self.accounts:
            raise NotFound(f"account {to_account} does not exist")
        amount = round(amount, 2)
        if amount > CONFIRMATION_LIMIT:
            result = {
                "status": "pending_confirmation",
                "from_account": from_account,
                "to_account": to_account,
                "amount": amount,
                "message": "Transfers above 1,000 GBP need confirmation in the app.",
            }
        else:
            with self._lock:
                if src["balance"] < amount:
                    raise InsufficientFunds(f"balance of {from_account} is too low")
                src["balance"] = round(src["balance"] - amount, 2)
                self.accounts[to_account]["balance"] = round(
                    self.accounts[to_account]["balance"] + amount, 2
                )
                tx_id = f"TR-{len(self.transfers) + 1:05d}"
                result = {
                    "status": "completed",
                    "transfer_id": tx_id,
                    "from_account": from_account,
                    "to_account": to_account,
                    "amount": amount,
                    "reference": reference,
                }
                self.transfers.append({**result, "user_id": user_id})
        with self._lock:
            self._idempotency[idempotency_key] = result
        return result

    def search_help_center(self, query: str, k: int = 2) -> dict[str, Any]:
        self._maybe_fail()
        words = set(re.findall(r"[a-z]+", query.lower()))
        scored = [(len(words & set(doc["keywords"])), doc) for doc in self.help_center]
        hits = [
            {"id": d["id"], "title": d["title"], "answer": d["answer"]}
            for s, d in sorted(scored, key=lambda x: -x[0])
            if s > 0
        ][:k]
        return {"query": query, "results": hits}
