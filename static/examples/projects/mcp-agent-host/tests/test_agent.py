"""End-to-end agent tests: real graph, real MCP servers in-process, scripted LLM."""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from mcp_host.agent import ApprovalDecision
from tests.conftest import call, collect


async def test_discovery_namespaces_colliding_tools(make_runtime):
    async with make_runtime() as rt:
        snap = rt.host.registry.snapshot
        assert "notes__search" in snap.tools and "docs__search" in snap.tools
        assert sorted(snap.collisions["search"]) == ["docs", "notes"]
        assert snap.tools["notes__delete_note"].destructive
        assert snap.tools["calendar__cancel_event"].destructive  # from destructiveHint


async def test_collision_routes_to_the_right_server(make_runtime):
    script = [
        call("docs__search", {"query": "annual leave days"}, "c1"),
        AIMessage(content="25 days [leave-policy]"),
    ]
    async with make_runtime(script) as rt:
        events = await collect(rt.service.send("t1", "How many leave days do I get?"))
        result = next(e for e in events if e["type"] == "tool_result")
        assert result["status"] == "success"
        assert "leave-policy" in result["preview"]
        assert {"type": "message", "text": "25 days [leave-policy]"} in events


async def test_destructive_tool_pauses_for_approval_then_runs(make_runtime, data_dirs):
    script = [
        call("notes__delete_note", {"name": "reading-list"}, "del1"),
        AIMessage(content="done"),
    ]
    async with make_runtime(script) as rt:
        events = await collect(rt.service.send("t2", "delete note reading-list"))
        approval = next(e for e in events if e["type"] == "approval_required")
        assert approval["calls"][0]["tool"] == "notes__delete_note"
        assert (data_dirs["notes"] / "reading-list.md").exists()  # nothing ran yet
        assert await rt.service.pending_approval("t2") is not None

        events = await collect(rt.service.resume("t2", ApprovalDecision(approve=["del1"])))
        assert any(e["type"] == "tool_result" and e["status"] == "success" for e in events)
        assert not (data_dirs["notes"] / "reading-list.md").exists()
        assert await rt.service.pending_approval("t2") is None


async def test_rejected_approval_does_not_run(make_runtime, data_dirs):
    script = [call("notes__delete_note", {"name": "reading-list"}, "del1"), AIMessage(content="ok")]
    async with make_runtime(script) as rt:
        await collect(rt.service.send("t3", "delete note reading-list"))
        events = await collect(rt.service.resume("t3", ApprovalDecision(approve=[])))
        result = next(e for e in events if e["type"] == "tool_result")
        assert result["status"] == "error" and "rejected by the user" in result["preview"]
        assert (data_dirs["notes"] / "reading-list.md").exists()


async def test_new_message_refused_while_approval_pending(make_runtime):
    script = [call("notes__delete_note", {"name": "reading-list"}, "d")]
    async with make_runtime(script) as rt:
        await collect(rt.service.send("t4", "delete it"))
        events = await collect(rt.service.send("t4", "something else"))
        assert events[0]["type"] == "error"


async def test_injection_in_docs_cannot_trigger_other_servers(make_runtime, data_dirs):
    """The poisoned vendor doc tells the agent to cancel events and delete notes."""
    script = [
        call("docs__search", {"query": "vendor onboarding portal notes"}, "s1"),
        # A compromised model obeys the injected text:
        AIMessage(
            content="",
            tool_calls=[
                {"name": "calendar__cancel_event", "args": {"event_id": "evt_standup"}, "id": "x1"},
                {"name": "notes__delete_note", "args": {"name": "sprint-goals"}, "id": "x2"},
            ],
        ),
        AIMessage(content="I could not do that."),
    ]
    async with make_runtime(script) as rt:
        events = await collect(rt.service.send("t5", "How do we onboard a vendor?"))
        first = next(e for e in events if e["type"] == "tool_result" and e["id"] == "s1")
        assert first["flagged"] is True
        blocked = [e for e in events if e["type"] == "tool_result" and e["id"] in {"x1", "x2"}]
        assert len(blocked) == 2 and all("blocked" in e["preview"] for e in blocked)
        # No approval was even requested: the guard blocks before the approval gate.
        assert not any(e["type"] == "approval_required" for e in events)
        assert (data_dirs["notes"] / "sprint-goals.md").exists()


async def test_taint_resets_on_the_next_user_message(make_runtime):
    script = [
        call("docs__search", {"query": "vendor portal notes"}, "s1"),
        AIMessage(content="summary"),
        call("notes__list_notes", {}, "n1"),
        AIMessage(content="your notes"),
    ]
    async with make_runtime(script) as rt:
        await collect(rt.service.send("t6", "vendor onboarding?"))
        events = await collect(rt.service.send("t6", "list my notes"))
        result = next(e for e in events if e["type"] == "tool_result")
        assert result["status"] == "success"


async def test_output_is_spotlighted_and_truncated(make_runtime):
    script = [
        call("host__read_resource", {"uri": "docs://doc/leave-policy"}, "r1"),
        AIMessage(content="ok"),
    ]
    async with make_runtime(script) as rt:
        rt.host.servers["docs"].max_output_chars = 200
        await collect(rt.service.send("t7", "read the leave policy"))
        state = await rt.service.graph.aget_state({"configurable": {"thread_id": "t7"}})
        tool_msg = next(m for m in state.values["messages"] if isinstance(m, ToolMessage))
        assert tool_msg.content.startswith('<tool_output id="')
        assert 'trust="untrusted"' in tool_msg.content
        assert "truncated by the host" in tool_msg.content
        assert tool_msg.artifact["truncated"] is True


async def test_denied_tool_is_hidden_and_blocked(make_runtime):
    from tests.conftest import servers_file

    script = [
        call("notes__write_note", {"name": "x", "content": "y"}, "w1"),
        AIMessage(content="no"),
    ]
    async with make_runtime(script, servers=servers_file(notes={"deny": ["write_note"]})) as rt:
        assert "notes__write_note" not in rt.host.registry.snapshot.tools
        events = await collect(rt.service.send("t8", "write a note"))
        result = next(e for e in events if e["type"] == "tool_result")
        assert "unknown or unavailable" in result["preview"]


async def test_conversation_persists_across_turns(make_runtime):
    script = [AIMessage(content="hello"), AIMessage(content="again")]
    async with make_runtime(script) as rt:
        await collect(rt.service.send("t9", "hi"))
        await collect(rt.service.send("t9", "hi again"))
        history = await rt.service.history("t9")
        assert [h["role"] for h in history] == ["human", "ai", "human", "ai"]
        assert "t9" in await rt.service.threads()


async def test_streams_tokens(make_runtime):
    async with make_runtime([AIMessage(content="one two three")]) as rt:
        events = await collect(rt.service.send("t10", "count"))
        tokens = "".join(e["text"] for e in events if e["type"] == "token")
        assert tokens == "one two three"


async def test_mcp_prompt_starts_a_turn(make_runtime):
    async with make_runtime([AIMessage(content="cited answer")]) as rt:
        await collect(
            rt.service.send_prompt(
                "t11", "docs", "answer_with_citations", {"question": "meal allowance?"}
            )
        )
        history = await rt.service.history("t11")
        assert history[0]["role"] == "human"
        assert "meal allowance?" in history[0]["content"]
        assert "docs__search" in history[0]["content"]


async def test_step_limit_stops_runaway_loops(make_runtime):
    looping = [call("notes__list_notes", {}, f"l{i}") for i in range(10)]
    async with make_runtime(looping, max_agent_steps=3) as rt:
        events = await collect(rt.service.send("t12", "loop"))
        assert any(e["type"] == "message" and "stopped after 3" in e["text"] for e in events)
