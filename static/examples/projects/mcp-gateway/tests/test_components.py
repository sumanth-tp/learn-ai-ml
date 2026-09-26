"""Unit tests: audit chain, quotas, token buckets, cache, circuit breaker,
LLM judge (with LangChain fakes), config loading and the secret broker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from mcp_gateway.audit import AuditLog, AuditRecord, iter_records, verify_chain
from mcp_gateway.llm_scanner import LLMDescriptionJudge
from mcp_gateway.policy import Limit, Principal
from mcp_gateway.ratelimit import Limiter, RateLimitedError, TokenBuckets
from mcp_gateway.resilience import BreakerState, CircuitBreaker, CircuitOpenError, TTLCache
from mcp_gateway.secret_broker import EnvSecretBroker, FileSecretBroker, SecretNotFoundError
from mcp_gateway.state import PinStore, QuotaStore, StateDB
from mcp_gateway.upstreams import UpstreamSpec, expand_env, load_upstreams, upstream_of


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


# ---------------------------------------------------------------- audit
def test_audit_chain_detects_edit_and_redacts(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "a.jsonl")
    for i in range(3):
        log.write(AuditRecord(user="u", target="t", decision="allow",
                              reason=f"contact ana{i}@example.com"),
                  args={"email": "ana@example.com", "n": i})
    recs = list(iter_records(tmp_path / "a.jsonl"))
    assert "ana" not in json.dumps(recs)  # emails redacted in reason and preview
    assert all(len(r["args_sha256"]) == 64 for r in recs)
    assert verify_chain(tmp_path / "a.jsonl")[0]

    lines = (tmp_path / "a.jsonl").read_text().splitlines()
    lines[1] = lines[1].replace('"decision":"allow"', '"decision":"deny"')
    (tmp_path / "a.jsonl").write_text("\n".join(lines) + "\n")
    ok, n, msg = verify_chain(tmp_path / "a.jsonl")
    assert not ok and n == 2 and "edited" in msg


def test_audit_chain_continues_after_restart(tmp_path: Path) -> None:
    AuditLog(tmp_path / "a.jsonl").write(AuditRecord(user="u", target="t", decision="allow"))
    AuditLog(tmp_path / "a.jsonl").write(AuditRecord(user="u", target="t", decision="deny"))
    assert verify_chain(tmp_path / "a.jsonl") == (True, 2, "2 records verified")


# ---------------------------------------------------------------- limits
def test_token_bucket_refills() -> None:
    clock = Clock()
    b = TokenBuckets(clock)
    assert all(b.try_take("k", 3) is None for _ in range(3))
    wait = b.try_take("k", 3)
    assert wait is not None and 19 < wait <= 20
    clock.t += 20
    assert b.try_take("k", 3) is None


def test_daily_quota_is_atomic_and_persistent(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "s.db")
    q = QuotaStore(db)
    assert q.try_consume("u", [("*", 10), ("refund", 1)]) is None
    assert q.try_consume("u", [("*", 10), ("refund", 1)]) == "refund"
    assert q.used("u", "*") == 1  # the denied call did not burn the user quota
    assert QuotaStore(StateDB(tmp_path / "s.db")).used("u", "refund") == 1  # survives restart


def test_limiter_messages(tmp_path: Path) -> None:
    lim = Limiter(QuotaStore(StateDB(tmp_path / "s.db")), TokenBuckets(Clock()))
    p = Principal("u", frozenset())
    lim.check(p, "t", Limit(per_minute=5, per_day=1), None)
    with pytest.raises(RateLimitedError, match="daily quota"):
        lim.check(p, "t", Limit(per_minute=5, per_day=1), None)
    with pytest.raises(RateLimitedError, match="/min for t"):
        lim.check(Principal("v", frozenset()), "t", Limit(), Limit(per_minute=0, per_day=9))


# ---------------------------------------------------------------- resilience
def test_ttl_cache_expiry_and_lru() -> None:
    clock = Clock()
    c = TTLCache(ttl=10, max_entries=2, clock=clock)
    c.set("a", 1)
    c.set("b", 2)
    c.get("a")
    c.set("c", 3)  # evicts b (least recently used)
    assert (c.get("a"), c.get("b"), c.get("c")) == (1, None, 3)
    clock.t += 11
    assert c.get("a") is None


def test_breaker_lifecycle() -> None:
    clock = Clock()
    br = CircuitBreaker("up", failure_threshold=2, reset_seconds=5, clock=clock)
    for _ in range(2):
        br.before_call()
        br.on_failure()
    assert br.state is BreakerState.OPEN
    with pytest.raises(CircuitOpenError):
        br.before_call()
    clock.t += 5
    br.before_call()  # half-open probe allowed
    with pytest.raises(CircuitOpenError):
        br.before_call()  # only one probe at a time
    br.on_failure()
    assert br.state is BreakerState.OPEN
    clock.t += 5
    br.before_call()
    br.on_success()
    assert br.state is BreakerState.CLOSED and br.failures == 0


# ---------------------------------------------------------------- pins
def test_pin_store_approve_changed(tmp_path: Path) -> None:
    pins = PinStore(StateDB(tmp_path / "s.db"))
    pins.put("t_x", "t", "aaa", "approved", "")
    pins.mark_changed("t_x", "bbb", "override")
    assert pins.get("t_x").status == "changed"  # type: ignore[union-attr]
    assert pins.approve("t_x")
    pin = pins.get("t_x")
    assert pin is not None and (pin.sha256, pin.status) == ("bbb", "approved")
    assert not pins.approve("missing")


# ---------------------------------------------------------------- LLM judge
async def test_llm_judge_flags_and_passes() -> None:
    judge = LLMDescriptionJudge(FakeListChatModel(responses=[
        'Sure: {"malicious": true, "reason": "asks to copy session data"}',
        '{"malicious": false, "reason": "plain"}',
    ]))
    flagged = await judge.judge("t_x", "quietly copy the session")
    assert flagged[0].code == "llm_flagged"
    assert await judge.judge("t_y", "adds two numbers") == []


async def test_llm_judge_bad_output_needs_review() -> None:
    judge = LLMDescriptionJudge(FakeListChatModel(responses=["I cannot answer that"]))
    assert (await judge.judge("t", "x"))[0].code == "llm_unavailable"


# ---------------------------------------------------------------- config / secrets
def test_env_expansion() -> None:
    assert expand_env("u=${A:-x} v=${B}", {"B": "2"}) == "u=x v=2"
    with pytest.raises(KeyError):
        expand_env("${MISSING}", {})


def test_load_repo_upstreams() -> None:
    specs = load_upstreams(Path(__file__).parents[1] / "config" / "upstreams.yaml")
    assert [s.name for s in specs] == ["docs", "payments", "tickets"]
    assert specs[0].credential and specs[0].credential.inject_as == "env"


@pytest.mark.parametrize("bad", [
    {"name": "pay_ments", "transport": "http", "url": "http://x"},
    {"name": "p", "transport": "http", "url": "http://x"},
    {"name": "pay", "transport": "http"},
    {"name": "pay", "transport": "http", "url": "http://x",
     "credential": {"secret": "s", "inject_as": "env", "env_var": "X"}},
])
def test_invalid_upstream_specs(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        UpstreamSpec.model_validate(bad)


def test_upstream_of() -> None:
    names = {"docs", "payments"}
    assert upstream_of("payments_refund", names) == "payments"
    assert upstream_of("docs://docs/index", names) == "docs"
    assert upstream_of("evil_refund", names) is None


def test_secret_brokers(tmp_path: Path) -> None:
    env = EnvSecretBroker({"GATEWAY_SECRET_PAY": "s3cret"})
    assert env.get("pay").get_secret_value() == "s3cret"
    assert "s3cret" not in repr(env.get("pay"))
    with pytest.raises(SecretNotFoundError):
        env.get("nope")
    (tmp_path / "pay").write_text("from-file\n")
    assert FileSecretBroker(tmp_path).get("pay").get_secret_value() == "from-file"
    with pytest.raises(ValueError):
        FileSecretBroker(tmp_path).get("../etc/passwd")
