---
id: afr-interoperability
title: "Agent Interoperability: MCP and A2A"
sidebar_label: "2 · MCP and A2A"
sidebar_position: 2
slug: /agentic-frontier/agent-interoperability-mcp-and-a2a
description: "Read the 2026-07-28 revision of the Model Context Protocol and version 1.0 of the Agent2Agent protocol side by side: what each message looks like on the wire, how discovery, state and authorisation work, and what the stateless redesign costs, all run against real servers."
tags: [mcp, a2a, interoperability, json-rpc, agents, protocols, authorisation, streamable-http]
---

import Infographic from '@site/src/components/Infographic';
import ProtocolFlowLab from '@site/src/components/viz/ProtocolFlowLab';

**In one line.** MCP is how one agent reaches tools and data, A2A is how one agent hands work to another, and both are plain JSON-RPC messages whose shape you can read, send by hand and test.

:::note Not from a lecture
Written for this site from the current official specifications, opened on 7 October 2026: the MCP specification at its `latest` address, which resolved to revision 2026-07-28 (changelog against 2025-11-25), and the A2A specification at its `latest` address, version 1.0.0. Every message below was produced by a real server: the MCP Python SDK 2.3.0 and the A2A Python SDK 1.2.2, run with Python 3.14.6 in `.lecture-import/venv-llm`. Where I could not check something against a running system, the page says so.
:::

:::warning The older MCP chapters describe an older revision
The video-based [MCP lifecycle](/docs/mcp/mcp-lifecycle) and [architecture](/docs/mcp/mcp-architecture) chapters teach the `initialize` handshake and sessions of the 2025 revisions. The 2026-07-28 revision removed both. That material is still correct for servers that speak 2025-11-25 and earlier, and the code below shows both eras side by side. Whenever the two disagree, this chapter follows the current specification.
:::

:::tip Before you start
You should already know:

- what a tool call is and why an agent needs tools ([Tools in LangGraph](/docs/agentic-ai/tools-in-langgraph));
- the idea of MCP as a connector between an agent and its tools ([MCP: the why](/docs/mcp/mcp-the-why));
- what JSON and an HTTP request are. JSON-RPC is explained below.

Reading time: about 45 minutes with the code.

After this chapter you can:

- send and read MCP and A2A messages by hand, and explain each field that matters;
- explain what changed in MCP in July 2026 and what it costs and buys;
- decide when a capability should be an MCP tool, an A2A agent, or neither.
:::

## In 30 seconds

Two kinds of connection matter for an agent. The first is to **things**: a database, a calendar, a file store. The Model Context Protocol (MCP) gives every such thing the same doorway, so one agent can use many tools without custom glue. The second is to **other agents**: a travel agent asking an expense agent for approval. The Agent2Agent protocol (A2A) gives agents a shared way to hand over a job that may take minutes, ask a question halfway and return a result.

Think of a shop. MCP is the shop's price list and tills: ask for an item, get an answer, finished. A2A is hiring a contractor: describe the job, they may ask which colour you want, and the work finishes later.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| JSON-RPC | A way to call a named function by sending JSON, and get a JSON reply with the same id | request `id: 7`, reply `id: 7` |
| MCP host, client, server | The app with the model, its connector to one server, the program offering tools | an IDE, its client, a pricing server |
| Capability | A feature a side says it supports | the client can ask the user questions |
| `_meta` | A side channel in an MCP message for protocol bookkeeping | the protocol version, on every request |
| Agent Card | A public JSON file describing an A2A agent | name, skills, URL, how to log in |
| Task | A unit of work an A2A agent tracks over time | "approve this expense", state WORKING |
| Artifact | A result produced by a task | the text "approved" |
| Stateless | Every request carries what the server needs; the server remembers nothing between them | send the protocol version each time |

## The idea in plain words

Imagine a calculator that must be introduced first. You say "I speak English and I can read tables", it says "me too, I can add and multiply", and only then do you ask for 2 + 2. That introduction is useful if you will ask a hundred questions. It is a nuisance if you ask one, and a disaster if the calculator is one of ten identical machines behind a counter and your second question reaches a different machine that never heard your introduction.

That is the story of MCP's biggest change. Until the 2025-11-25 revision, a client opened with an `initialize` request, the two sides agreed on a version and features, and the connection then carried that agreement as a **session**. The 2026-07-28 revision removes the handshake and the session. Each request now says which protocol version it speaks and which client features it has, so any copy of the server can answer it. The price is a few hundred bytes on every message, which the second code block measures.

A2A starts from a different problem. Another agent is not a function you call; it may be built by another company, may take ten minutes, may need to ask you something, and you may not be allowed to see how it works. So A2A models a **task** with a state that moves, a place where the agent publishes who it is (the Agent Card), and messages that can carry text, files or data. The two protocols answer different questions, and a real system often uses both: an agent uses MCP to reach its tools, and A2A to ask another agent for help.

<Infographic src="/img/afr/mcp-and-a2a-big-picture.svg" alt="Two panels: MCP connecting a host app, an MCP client and an MCP server for tools and data, and A2A connecting a client agent to an opaque remote agent through tasks, above a table comparing unit of work, discovery, state, transports and authorisation." caption="Start with the two panels: tools go left, other agents go right. The table underneath is the one-page comparison." />

## Worked example, step by step

**Part 1: counting messages for one cold call.** Say a client has never spoken to a server and wants one tool result.

1. Old style (2025-11-25). The client sends `initialize` (message 1) and the server replies (message 2). The client sends a `notifications/initialized` notice, which gets no reply (message 3). Only then can it send `tools/call` (message 4) and read the result (message 5). That is 5 messages: three from the client (the request, the notice and the call) and two replies from the server.
2. New style (2026-07-28). The client sends `tools/call` with the version and its capabilities inside (message 1), and the server replies (message 2). Two messages.

The code measures the bytes: the old style sends 310 bytes and receives 398, the new style 223 and 259. With one call the new style is cheaper.

**Part 2: where it stops being cheaper.** The old style pays a setup cost once, then each call is short. Measured on the same server: setup 447 bytes, then 261 bytes per call. The new style pays 482 bytes per call and no setup. The two lines cross where 447 + 261 × n equals 482 × n, so n = 447 ÷ 221 = 2.02 calls. At 2 calls the new style still wins (964 bytes against 969). From call 3, the old style is cheaper in bytes. Stateless costs bytes, and buys freedom to send each request to any server copy.

**Part 3: a task by hand.** An A2A client asks an expense agent to approve an expense. Turn 1: "please approve my expense". The agent has no amount, so it creates a Task with state `TASK_STATE_INPUT_REQUIRED` and asks "What is the amount?". Turn 2: the client sends "amount 120" with the same task id. The agent adds an artifact "expense of 120: approved" and moves the Task to `TASK_STATE_COMPLETED`. Five facts to check in the code: the card names the agent, turn 1 ends in INPUT_REQUIRED, turn 2 ends in COMPLETED, the ids match, and the task's history holds 3 messages.

<Infographic src="/img/afr/mcp-call-worked-example.svg" alt="Four panels showing one MCP tool call: refused because the client lacks the elicitation capability, answered with an input-required result, completed after a retry, and rejected when the state token is tampered with, above a table of bytes for the old and new MCP styles." caption="Read left to right: the same tool call, four outcomes. The table at the bottom is the byte arithmetic of Part 2." />

## How it works

### JSON-RPC in two minutes

Both protocols use JSON-RPC 2.0. A **request** has `jsonrpc: "2.0"`, a unique `id`, a `method` name and `params`. The **response** has the same `id` and either a `result` or an `error` with an integer `code`. A **notification** has no `id` and gets no reply. That is all. MCP forbids a `null` id and requires unique ids for requests still waiting; the reply is matched to the request by id alone, so many requests can be in flight at once.

### MCP in the 2026-07-28 revision

**Every request carries its own context.** The `params._meta` object must contain `io.modelcontextprotocol/protocolVersion` and `io.modelcontextprotocol/clientCapabilities`, and should contain `io.modelcontextprotocol/clientInfo`. A request without the required keys gets error `-32602` (HTTP 400). A server that does not speak the requested version answers `-32022` with the list of versions it supports, and the client retries with one of them.

**Discovery is a method, not a handshake.** Every server must implement `server/discover`, which returns its supported versions, capabilities, name and optional instructions for the model. Calling it is optional. `tools/list` returns the tools, with `ttlMs` and `cacheScope` so a client knows how long it may cache the list. Servers should list tools in a stable order, which helps the prompt cache from [chapter 1](/docs/agentic-frontier/context-engineering).

**A tool call is `tools/call`.** The result has `resultType: "complete"` and `content` (text, images, links), optionally `structuredContent` that matches the tool's `outputSchema`. A failure the model can fix (a bad date) is a normal result with `isError: true`. A broken request is a JSON-RPC error. This split matters: the model sees the first kind and can correct itself.

**A server cannot call the client any more; it asks through the result.** Earlier revisions let a server send its own request, for example to ask the user a question mid-call. That needed a connection that stays open, which a stateless server cannot rely on. The new rule, called multi round-trip requests (MRTR), is: the server answers the call with `resultType: "input_required"`, listing `inputRequests` (such as an elicitation form) and an opaque `requestState`. The client collects the answer and sends the original call again, with a new request id, plus `inputResponses` and the `requestState` unchanged. The server must treat that state as untrusted input and protect it with a signature or encryption, because it travels through the client.

**Other changes worth knowing.** Notification streams are now opened on purpose with `subscriptions/listen`. Tasks for long-running work moved out of the core into an official extension, `io.modelcontextprotocol/tasks`. Roots, sampling and logging are deprecated and stay usable for at least twelve months (the AAIF migration post gives July 2027 as the earliest removal). `ping` and `logging/setLevel` are gone. Error codes `-32020` to `-32099` are reserved for the specification.

**Transports.** Over **stdio** the client launches the server as a process and exchanges one JSON message per line. Over **Streamable HTTP** every message is its own POST to one endpoint, and the server answers with JSON or a short event stream. The request must carry headers `MCP-Protocol-Version`, `Mcp-Method` and, for calls that name something, `Mcp-Name`. Servers must check that they match the body, and reject a mismatch with `-32020`. Why? So a gateway or load balancer can route and allow or deny by header without parsing JSON, and a client cannot fool it by sending a harmless name in the header and a harmful one in the body. The fourth code block tries exactly that.

**Authorisation (HTTP only).** The MCP server acts as an OAuth resource server. An unauthenticated request gets `401` with a `WWW-Authenticate` header pointing at the server's protected-resource metadata (RFC 9728). The client reads the metadata to find the authorisation server, registers (the spec now prefers Client ID Metadata Documents and deprecates Dynamic Client Registration), and runs an authorisation-code flow with PKCE. The token request must include the `resource` parameter (RFC 8707) naming this MCP server, and the server must accept only tokens issued for it. A request with too few scopes gets `403` and `error="insufficient_scope"`. Local stdio servers are told not to follow this flow and to take credentials from the environment. I read this section and did not execute the flow.

**Security still depends on you.** The specification says hosts must obtain user consent before invoking tools, and that tool descriptions and annotations are untrusted unless they come from a trusted server. A server you did not write can put instructions in a description. See [AI security and guardrails](/docs/projects/ai-security/guardrails) and the [enterprise MCP gateway project](/docs/mcp/project-3-enterprise-mcp-gateway).

### A2A 1.0

**Discovery is a file.** An A2A server must publish an Agent Card, normally at `/.well-known/agent-card.json`. It names the agent, its `version`, its `skills` and default input and output media types, its `capabilities` (streaming, push notifications, an extended card for logged-in users) and, in `supportedInterfaces`, the URLs and bindings (`JSONRPC`, `GRPC`, `HTTP+JSON`) with the protocol version each speaks. The card also declares security schemes: API key, HTTP, OAuth 2.0, OpenID Connect or mutual TLS. Agent Cards can be signed, a feature the A2A project describes as part of 1.0.

**The unit is a Task.** `SendMessage` carries a message (`messageId`, `role` of `ROLE_USER` or `ROLE_AGENT`, and `parts`, each part being text, raw bytes, a URL or structured data). The reply is either a Message (quick answers) or a Task with an `id`, a `contextId` that groups related work, a `status` with a state, `artifacts` for results and a `history`.

**The states.** A task starts `TASK_STATE_SUBMITTED`, moves to `WORKING`, and can pause in `INPUT_REQUIRED` or `AUTH_REQUIRED` (interrupted states: the task resumes when the missing input or credential arrives) or end in `COMPLETED`, `FAILED`, `CANCELED` or `REJECTED` (terminal).

<Infographic src="/img/afr/a2a-task-lifecycle.svg" alt="A state diagram of an A2A task: submitted, working, the interrupted states input required and auth required, and the four terminal states, above the two-turn expense example." caption="Follow the arrows from SUBMITTED. The dashed arrow is turn 2 of the example: a new message with the task id. The panels at the bottom are block 5's output." />

Other methods: `SendStreamingMessage` and `SubscribeToTask` stream status and artifact updates, `GetTask` and `ListTasks` read state, `CancelTask` stops work, and push-notification methods register a webhook for tasks that outlive a connection.

**Versions.** Clients must send an `A2A-Version` header on each request, with major and minor only (for example `1.0`). An empty value is read as 0.3. A server that does not support the version returns `VersionNotSupportedError`.

**What A2A does not tell you.** An A2A agent is opaque on purpose. You see its card and its messages, not its tools or its prompts. That is good for companies that do not want to share internals, and it means your own evaluation and guardrails must treat the other agent as an untrusted party.

### Where each belongs

Use MCP when the thing is a function with a clear input and output that your agent calls: query a database, read a file. Use A2A when the thing has its own judgment and lifecycle: it may refuse, ask questions or take minutes. A useful test: if you would be comfortable describing it with a JSON schema and a timeout, it is a tool. If you would need to ask it a follow-up question, it is an agent. The two also compose: an A2A agent can use MCP tools internally, and an MCP server can wrap an agent as a tool, at the price of hiding its task states behind one call.

## Code you can run

Five blocks, each self-contained. They use `mcp` 2.3.0, `a2a-sdk` 1.2.2, `httpx` 0.28.1 and `tiktoken` 0.14.0. Every server is started as a subprocess from the block itself and stopped at the end. Blocks 4 and 5 pick a free port. The client side is plain `json` and `httpx`, so you read the real messages and not an SDK's view of them.

### 1. A real server, spoken to by hand

We start an MCP server over stdio and send it JSON lines. First we ask what it is, then list and call a tool, then break two rules on purpose.

```python
import json
import subprocess
import sys

import tiktoken

SERVER = '''
from mcp.server.mcpserver import MCPServer
app = MCPServer("pricing-demo")

@app.tool()
def quote(sku: str, quantity: int) -> str:
    """Price a quantity of one product."""
    return f"{quantity} x {sku} = {quantity * 12.5:.2f}"

@app.tool()
def stock(sku: str) -> int:
    """How many units are in stock."""
    return 40

app.run(transport="stdio")
'''
META = {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientInfo": {"name": "hand-client", "version": "0.1"},
        "io.modelcontextprotocol/clientCapabilities": {}}
proc = subprocess.Popen([sys.executable, "-c", SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL, text=True)

def send(request_id, method, params=None, meta=META):
    body = dict(params or {})
    if meta is not None:
        body["_meta"] = meta
    proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": body}) + "\n")
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())

found = send(1, "server/discover")["result"]
print("versions", found["supportedVersions"], "| capabilities", sorted(found["capabilities"]))
listing = send(2, "tools/list")["result"]
print("tools", [t["name"] for t in listing["tools"]], "| ttlMs", listing["ttlMs"], "| cacheScope", listing["cacheScope"])
enc = tiktoken.get_encoding("o200k_base")
print("tools/list payload:", len(json.dumps(listing["tools"])), "bytes,", len(enc.encode(json.dumps(listing["tools"]))), "tokens")
reply = send(3, "tools/call", {"name": "quote", "arguments": {"sku": "A1", "quantity": 4}})["result"]
print("call ->", reply["content"][0]["text"], "| resultType", reply["resultType"], "| isError", reply["isError"])
old = send(4, "tools/list", meta={**META, "io.modelcontextprotocol/protocolVersion": "1900-01-01"})["error"]
print("old version ->", old["code"], old["message"], old["data"])
bare = send(5, "tools/list", meta=None)["error"]
print("no _meta ->", bare["code"], bare["message"][:60])
proc.stdin.close()
proc.wait(timeout=10)
```

**Reading the output.** The server supports one version, `2026-07-28`, and offers `prompts`, `resources` and `tools`. Its two tools take 785 bytes and 223 tokens to describe, which is what every agent request that includes them pays (see [chapter 1](/docs/agentic-frontier/context-engineering)). `ttlMs 0` and `cacheScope private` mean this SDK's default tells clients not to cache the list; a server that wants caching has to opt in. The call returns text with `resultType: complete`. The unknown version `1900-01-01` gets error `-32022` with the list of supported versions, and a request with no `_meta` gets `-32602`.

**Line by line.**

- `SERVER` is source code for a small server built with `MCPServer`. We run it with `python -c`, so the block needs no extra file.
- `META` is the per-request envelope: version, optional client name, capabilities (an empty object: this client offers nothing special).
- `send` copies the parameters, adds `_meta`, writes one JSON line and reads one line back. Passing `meta=None` leaves it out.

### 2. What statelessness costs

Now we run the same tool call against the same server twice. Once with the old handshake (the SDK server still understands it) and once in the new style, and count the bytes.

```python
import json
import subprocess
import sys

SERVER = '''
from mcp.server.mcpserver import MCPServer
app = MCPServer("pricing-demo")

@app.tool()
def quote(sku: str, quantity: int) -> str:
    return f"{quantity} x {sku} = {quantity * 12.5:.2f}"

app.run(transport="stdio")
'''
CALL = {"name": "quote", "arguments": {"sku": "A1", "quantity": 4}}
MODERN_META = {"io.modelcontextprotocol/protocolVersion": "2026-07-28", "io.modelcontextprotocol/clientCapabilities": {}}

def exchange(messages):
    proc = subprocess.Popen([sys.executable, "-c", SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    sent = received = replies = 0
    for message in messages:
        line = json.dumps(message, separators=(",", ":"))
        proc.stdin.write(line + "\n")
        proc.stdin.flush()
        sent += len(line)
        if "id" in message:
            received += len(proc.stdout.readline().strip())
            replies += 1
    proc.stdin.close()
    proc.wait(timeout=10)
    return len(messages), replies, sent, received

hello = {"jsonrpc": "2.0", "id": 0, "method": "initialize",
         "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "c", "version": "1"}}}
ready = {"jsonrpc": "2.0", "method": "notifications/initialized"}

def legacy_session(calls):
    return [hello, ready] + [{"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": CALL} for i in range(1, calls + 1)]

def modern_session(calls):
    return [{"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {**CALL, "_meta": MODERN_META}}
            for i in range(1, calls + 1)]

def total(result):
    return result[2] + result[3]

legacy1, legacy2 = exchange(legacy_session(1)), exchange(legacy_session(2))
modern1, modern2 = exchange(modern_session(1)), exchange(modern_session(2))
print("one cold call, legacy: %d messages, %d replies, %d bytes out, %d bytes in" % legacy1)
print("one cold call, modern: %d messages, %d replies, %d bytes out, %d bytes in" % modern1)
legacy_per, modern_per = total(legacy2) - total(legacy1), total(modern2) - total(modern1)
legacy_setup = total(legacy1) - legacy_per
print("legacy: setup", legacy_setup, "bytes once, then", legacy_per, "bytes per call")
print("modern: setup 0 bytes, then", modern_per, "bytes per call")
for calls in (1, 5, 20, 100):
    print(f"{calls:4d} calls: legacy {legacy_setup + calls * legacy_per:6d} B, modern {calls * modern_per:6d} B")
first_worse = next(n for n in range(1, 200) if modern_per * n > legacy_setup + legacy_per * n)
print("stateless costs more bytes from call", first_worse, "onwards")
```

**Reading the output.** One cold call: 3 messages and 708 bytes the old way, 1 message and 482 bytes the new way. Fitting a line through sessions of one and two calls gives a setup cost of 447 bytes and 261 bytes per call for the old style, and 482 bytes per call for the new. At 100 calls that is 26,547 bytes against 48,200. The last line says the old style is cheaper from call 3 on, matching the by-hand calculation.

**Line by line.**

- `exchange` starts a fresh server, sends a list of messages and counts bytes in each direction. Messages with an `id` get a reply; the `initialized` notice does not.
- `legacy_session` and `modern_session` build the two conversations for any number of calls.
- The per-call cost is the difference between a 2-call and a 1-call session, and the setup is what is left of the 1-call session.

Bytes are not the whole story. The new style's extra 221 bytes per call are small beside a tool result, and they buy requests that can land on any server copy with no shared session store. The old style is cheaper per byte and needs sticky routing or a session store at scale.

### 3. A call that needs the user

A tool that has to ask the user a question mid-call, run under the new rules. The tool `refund` wants confirmation. We play the client and walk through four calls.

```python
import json
import subprocess
import sys

SERVER = '''
from typing import Annotated
from pydantic import BaseModel
from mcp.server.mcpserver import Elicit, MCPServer, Resolve

app = MCPServer("refund-demo")

class Confirm(BaseModel):
    approve: bool

def ask(order_id: str):
    return Elicit(f"Refund order {order_id}?", Confirm)

@app.tool()
def refund(order_id: str, ok: Annotated[Confirm, Resolve(ask)]) -> str:
    return f"refunded {order_id}" if ok.approve else "kept"

app.run(transport="stdio")
'''
proc = subprocess.Popen([sys.executable, "-c", SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL, text=True)

def call(request_id, capabilities, extra=None):
    meta = {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientCapabilities": capabilities}
    params = {"name": "refund", "arguments": {"order_id": "A-17"}, "_meta": meta, **(extra or {})}
    proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "method": "tools/call", "params": params}) + "\n")
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())

blind = call(1, {})
print("1. client without elicitation:", blind["error"]["code"], blind["error"]["data"])

form = {"elicitation": {"form": {}}}
first = call(2, form)["result"]
key = next(iter(first["inputRequests"]))
print("2. resultType:", first["resultType"], "| asks:", first["inputRequests"][key]["params"]["message"])
print("   request key:", key, "| state is an opaque string over 300 characters:", len(first["requestState"]) > 300)

retry = {"inputResponses": {key: {"action": "accept", "content": {"approve": True}}}, "requestState": first["requestState"]}
done = call(3, form, retry)["result"]
print("3. retry with answer:", done["resultType"], "|", done["content"][0]["text"])

forged = dict(retry, requestState=first["requestState"][:-4] + "AAAA")
tampered = call(4, form, forged)
print("4. tampered state:", tampered["error"]["code"], tampered["error"]["message"])
proc.stdin.close()
proc.wait(timeout=10)
```

**Reading the output.** Step 1: the client declared no capabilities, so the server cannot ask a question and returns `-32021` and names the capability it needs, `elicitation.form`. Step 2: with the capability the server answers `input_required`, with one question ("Refund order A-17?") and a state string over 300 characters. Step 3: the retry carries the answer and the unchanged state, and the call completes. Step 4: changing the last four characters of the state is rejected with `-32602`, so the state is protected against tampering.

**Line by line.**

- `call` sends the same tool call each time. Only the capabilities and the extra parameters change.
- `inputResponses` is keyed by the same key the server used in `inputRequests`. The client answers each question under its own key.
- `forged` replaces the last four characters of `requestState`. The SDK signs and encrypts the state, so the server notices.

### 4. Headers a gateway can trust

Streamable HTTP mirrors the method and tool name into headers. We start a real HTTP server and try an honest call, then two lies, then a missing header, with a tiny gateway rule in front.

```python
import socket
import subprocess
import sys
import time

import httpx

with socket.socket() as probe:
    probe.bind(("127.0.0.1", 0))
    PORT = probe.getsockname()[1]

SERVER = f'''
from mcp.server.mcpserver import MCPServer
app = MCPServer("pricing-demo")

@app.tool()
def quote(sku: str, quantity: int) -> str:
    return f"{{quantity}} x {{sku}} = {{quantity * 12.5:.2f}}"

app.run(transport="streamable-http", host="127.0.0.1", port={PORT})
'''
proc = subprocess.Popen([sys.executable, "-c", SERVER], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
URL = f"http://127.0.0.1:{PORT}/mcp"
for _ in range(60):
    try:
        httpx.get(URL, timeout=1)
        break
    except httpx.TransportError:
        time.sleep(0.25)

META = {"io.modelcontextprotocol/protocolVersion": "2026-07-28", "io.modelcontextprotocol/clientCapabilities": {}}

def body(tool):
    return {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": tool, "arguments": {"sku": "A1", "quantity": 2}, "_meta": META}}

HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
           "MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "tools/call", "Mcp-Name": "quote"}
ALLOWED = {"quote"}
cases = [
    ("honest call", body("quote"), HEADERS),
    ("header hides the real tool", body("delete_everything"), HEADERS),
    ("header names a forbidden tool", body("quote"), {**HEADERS, "Mcp-Name": "delete_everything"}),
    ("Mcp-Method missing", body("quote"), {k: v for k, v in HEADERS.items() if k != "Mcp-Method"}),
]
for label, payload, headers in cases:
    if headers.get("Mcp-Name") not in ALLOWED:
        print(f"{label:30s} gateway blocks on the header, body never parsed")
        continue
    reply = httpx.post(URL, json=payload, headers=headers, timeout=10)
    data = reply.json()
    outcome = data["result"]["content"][0]["text"] if "result" in data else f'{data["error"]["code"]} {data["error"]["message"]}'
    print(f"{label:30s} gateway forwards -> server HTTP {reply.status_code}: {outcome}")
proc.terminate()
proc.wait(timeout=10)
```

**Reading the output.** The honest call returns HTTP 200 and the quote. When the header says `quote` but the body names `delete_everything`, the gateway forwards it (it only looks at the header) and the server rejects it with `-32020`. When the header names a forbidden tool, the gateway blocks the request without parsing the body. A missing `Mcp-Method` is rejected with `-32020`.

**Line by line.**

- `ALLOWED` is the gateway's policy. It reads only `Mcp-Name`. That is the point of the header: cheap routing and allow-lists.
- The server's job is the other half: it compares header to body, so the cheap check cannot be bypassed.
- This SDK server is dual-era. A `GET` or an old version header would lead it down the legacy path, which the specification allows for dual-era servers. A server that speaks only the new revision should answer `405` to such requests. I did not test a modern-only server.

### 5. An A2A agent, discovered and used

The same pattern for A2A. We start a small agent built with the SDK, fetch its card, run the two-turn task from the worked example, read the task back and try a bad version.

```python
import json
import socket
import subprocess
import sys
import time
import uuid

import httpx

with socket.socket() as probe:
    probe.bind(("127.0.0.1", 0))
    PORT = probe.getsockname()[1]

SERVER = f'''
import uvicorn
from starlette.applications import Starlette
from a2a.helpers import new_task_from_user_message, new_text_message, new_text_part
from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, TaskState

class Approver(AgentExecutor):
    async def execute(self, context, event_queue):
        task = context.current_task or new_task_from_user_message(context.message)
        if context.current_task is None:
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue=event_queue, task_id=task.id, context_id=task.context_id)
        digits = "".join(c for p in context.message.parts for c in (p.text or "") if c.isdigit())
        if not digits:
            await updater.update_status(TaskState.TASK_STATE_INPUT_REQUIRED, message=new_text_message("What is the amount?"))
            return
        verdict = "approved" if int(digits) <= 500 else "needs a manager"
        await updater.add_artifact(parts=[new_text_part(text=f"expense of {{digits}}: {{verdict}}", media_type="text/plain")])
        await updater.update_status(TaskState.TASK_STATE_COMPLETED)

    async def cancel(self, context, event_queue):
        pass

card = AgentCard(name="Expense Approver", description="Approves small expenses.", version="0.1.0",
    default_input_modes=["text/plain"], default_output_modes=["text/plain"], capabilities=AgentCapabilities(streaming=True),
    supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url="http://127.0.0.1:{PORT}/", protocol_version="1.0")],
    skills=[AgentSkill(id="approve", name="Approve expense", description="Checks an amount against a limit.", tags=["finance"])])
handler = DefaultRequestHandler(agent_executor=Approver(), task_store=InMemoryTaskStore(), agent_card=card)
routes = create_agent_card_routes(card) + create_jsonrpc_routes(handler, "/")
uvicorn.run(Starlette(routes=routes), host="127.0.0.1", port={PORT}, log_level="error")
'''
proc = subprocess.Popen([sys.executable, "-c", SERVER], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
BASE = f"http://127.0.0.1:{PORT}"
for _ in range(60):
    try:
        httpx.get(BASE + "/.well-known/agent-card.json", timeout=1)
        break
    except httpx.TransportError:
        time.sleep(0.25)

card = httpx.get(BASE + "/.well-known/agent-card.json").json()
print("card:", card["name"], card["supportedInterfaces"][0]["protocolBinding"], card["supportedInterfaces"][0]["protocolVersion"],
      "| skills", [s["id"] for s in card["skills"]], "| streaming", card["capabilities"]["streaming"])

def rpc(request_id, method, params, version="1.0"):
    reply = httpx.post(BASE + "/", json={"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
                       headers={"A2A-Version": version}, timeout=10)
    return reply.json()

def message(text, **ids):
    return {"message": {"messageId": str(uuid.uuid4()), "role": "ROLE_USER", "parts": [{"text": text}], **ids}}

first = rpc(1, "SendMessage", message("please approve my expense"))["result"]["task"]
print("turn 1:", first["status"]["state"], "|", first["status"]["message"]["parts"][0]["text"])
second = rpc(2, "SendMessage", message("amount 120", taskId=first["id"], contextId=first["contextId"]))["result"]["task"]
print("turn 2:", second["status"]["state"], "|", second["artifacts"][0]["parts"][0]["text"], "| history", len(second["history"]), "messages")
print("same task id:", first["id"] == second["id"], "| same context:", first["contextId"] == second["contextId"])
print("GetTask:", rpc(3, "GetTask", {"id": first["id"]})["result"]["status"]["state"])
stale = rpc(4, "SendMessage", message("hello"), version="9.9")
print("version 9.9:", stale.get("error", {}).get("message"), "| code", stale.get("error", {}).get("code"))
proc.terminate()
proc.wait(timeout=10)
```

**Reading the output.** The card names the agent, one JSON-RPC interface at protocol version 1.0, one skill and `streaming True`. Turn 1 ends in `TASK_STATE_INPUT_REQUIRED` with the agent's question. Turn 2 reuses the task and context ids and ends in `TASK_STATE_COMPLETED` with the artifact "expense of 120: approved" and 3 messages in the history. `GetTask` returns the same final state later. A request with `A2A-Version: 9.9` is rejected with the SDK's version error, code `-32009` in this SDK; I did not look up whether the specification fixes that number for JSON-RPC.

**Line by line.**

- `Approver.execute` is the agent. It makes a Task if none exists, and replies with `INPUT_REQUIRED` when it finds no digits, or an artifact and `COMPLETED` when it does. `TaskUpdater` turns those calls into the events the protocol defines.
- `create_agent_card_routes` publishes the card at the well-known path, and `create_jsonrpc_routes` serves `SendMessage`, `GetTask` and the other methods.
- `rpc` adds the `A2A-Version` header. `message` builds a user message with a fresh `messageId`; passing `taskId` and `contextId` continues the task.

## The lab

<ProtocolFlowLab />

The default, the 2025-11-25 flow at step 5 with one call in the session, reproduces block 2: 708 bytes for the old style cold call against 482 for the new.

**What each control does.**

- **flow** chooses one of four captured conversations: the old handshake, the new single call, the user-input call (MRTR) and the two-turn A2A task.
- **step** reveals the messages one at a time. The highlighted arrow is the message shown in the box below.
- **calls in one session** changes only the byte comparison under the diagram: the handshake setup spread over more calls.
- **show data** lists each message with its direction and byte count.

**Try it yourself.**

1. With the first flow selected, drag **step** from 1 to 5 and watch the box. Messages 1 to 3 are the handshake; only message 4 does the work. Then pick the second flow: the whole job is message 1.
2. Set **calls in one session** to 2, then 3. At 2 the new style is still cheaper (964 against 969 bytes); at 3 the old style wins (1,230 against 1,446). Push it to 100 and the gap is 26,547 against 48,200 bytes.
3. Pick the MRTR flow and step to 2. The reply says `input_required` and carries the state. At step 3 the client sends the same call again with the answer, a new id and the state unchanged. Then pick the A2A flow and compare: the task id, not a retry, links turn 1 to turn 2.

## Designing with it

| Decision | MCP | A2A |
| --- | --- | --- |
| A function with clear inputs and outputs | tool | no |
| A peer with its own tools, judgment and lifecycle | wrap as a tool only if one call is enough | agent |
| Work that may take minutes | the Tasks extension, or poll a handle | native: task states, streaming, push |
| Needs the user's answer midway | `input_required` and retry | `INPUT_REQUIRED` state, then a message |
| Scale out behind a load balancer | designed for it now | server stores tasks; route by task id |
| Who can see the internals | you control the server | opaque by design |

Four rules. First, prefer a small, stable tool list: every tool you attach is tokens on every request, and cached only while the list stays the same. Second, treat any server or agent you did not write as untrusted: validate inputs and outputs and keep authority in a gateway, as in the [support agent case](/docs/senior/design-customer-support-agent-platform). Third, put identity in tokens, not in prompts: OAuth with the `resource` parameter binds a token to one server. Fourth, test the protocol like any API: a contract test that sends the messages in this chapter will catch a version change before your users do.

## Where this stands in 2026

:::info Industry view
MCP's 2026-07-28 revision was published as a release candidate on 21 May 2026 and was due to ship on 28 July 2026, according to the MCP blog and the Agentic AI Foundation's migration post of 21 July. It is described there as the largest revision since launch. The post gives 28 July 2027 as the earliest removal date for deprecated features, and says beta SDKs for Python, TypeScript, Go and C# already supported the candidate. The Python SDK I ran (2.3.0) speaks both eras on one server.

A2A was announced by Google in April 2025, donated to the Linux Foundation on 23 June 2025, and reached version 1.0 in March 2026, according to Google's open source blog of April 2026. That post counts "over 100" supporting companies, and describes MCP as handling tool integration and A2A coordination between autonomous agents, which matches this chapter's split. Adoption numbers are the vendors' own.

What I could not confirm: how many production systems use A2A today, and whether the two protocols will converge. Treat both as moving. Check the revision of the specification each library implements before you mix them.
:::

## Common mistakes

- **Treating an MCP connection as a session.** It feels natural, because earlier revisions had one. In 2026-07-28 a connection is not a conversation: state must travel as an explicit handle your tools take as an argument. Design tools that return an id and accept it later.
- **Trusting a tool description.** It feels like documentation. It is text from a server, and the specification calls annotations untrusted. Show tool names and arguments to the user, and keep the allow-list in code.
- **Dropping the `requestState` or editing it.** It looks like noise. The client must echo it exactly, and the server must reject a changed one. Forgetting it makes a retry start over.
- **Using A2A where one function call would do.** It feels more modern. A task lifecycle, an Agent Card and a second model add latency, cost and an untrusted party. If you can write the input and output as a schema, write a tool.
- **Skipping the version header.** It feels optional because things work in a demo. An A2A server reads an empty header as 0.3, and an MCP server rejects a request that lacks the version. Send it, and handle the error.

## Practice questions

<details>
<summary><strong>Easy.</strong> In MCP 2026-07-28, what two keys in `_meta` must every request carry, and what happens if one is missing?</summary>

`io.modelcontextprotocol/protocolVersion` and `io.modelcontextprotocol/clientCapabilities`. The server rejects the request with `-32602` (invalid params), and on HTTP with status 400. In block 1 the request with no `_meta` produced exactly that. `clientInfo` is recommended but not required.

</details>

<details>
<summary><strong>Easy.</strong> Which A2A state means "the agent needs another message from me", and what must the next message contain?</summary>

`TASK_STATE_INPUT_REQUIRED`. The client sends a new `SendMessage` with the same `taskId` (and `contextId`), so the server attaches it to the existing task. In block 5 that gave the same ids on both turns and a COMPLETED state.

</details>

<details>
<summary><strong>Medium.</strong> A tool has to ask the user to confirm a refund. Why can't the server just send a question over the open connection, as in 2025?</summary>

Because the 2026-07-28 revision is stateless: the server may not rely on an open connection or on earlier requests, and the retry may land on a different copy of the server. So the server returns `input_required` with the question and an opaque `requestState`, and the client sends the whole call again with the answer and the state. Any server copy can finish it because everything it needs is in the retry. The state must be integrity-protected because the client could edit it, which block 3 shows being rejected.

</details>

<details>
<summary><strong>Medium.</strong> Using the measured numbers (447 bytes setup, 261 per call for the handshake style, 482 per call for the stateless style), at how many calls do the two cost the same, and what does the answer tell you?</summary>

Set 447 + 261n = 482n, so n = 447 ÷ 221 = 2.02. At 2 calls the stateless style is still slightly cheaper (964 against 969 bytes); from 3 calls the handshake style is cheaper. It tells you stateless MCP trades roughly 200 extra bytes per call for requests that need no shared session, which matters for scale-out and not for byte count. If most of your traffic is long tool results, the overhead is small.

</details>

<details>
<summary><strong>Stretch.</strong> A gateway allows only tool `quote`. An attacker sends the header `Mcp-Name: quote` and a body that calls `delete_everything`. Why does it fail, and what if the server did not check?</summary>

The specification requires the server to compare the headers with the body and answer `-32020` on a mismatch, which block 4 shows. If the server skipped the check, the gateway would allow by the header and the server would run the body's tool: the gateway and server would disagree about what was asked. The mirrored headers are only safe because both layers are required to agree. An intermediary that enforces policy on headers should also check the protocol version header is a version that requires that validation.

</details>

<details>
<summary><strong>Stretch.</strong> You are designing a travel assistant that needs flight prices, a loyalty-points balance and approval from a separate corporate finance agent. Which pieces would you make MCP tools and which A2A, and what would you watch for?</summary>

Flight prices and the points balance are lookups with clear schemas: MCP tools, behind a gateway with scopes. Finance approval has its own policy, may ask questions and may take time: an A2A agent, called with a task and handled as untrusted. Watch the context cost of tool lists (chapter 1), user consent for anything that spends money, token audience binding for each MCP server, and the task states you must handle from the finance agent, especially `INPUT_REQUIRED`, `AUTH_REQUIRED` and `REJECTED`. Evaluate the finance agent with its answers as data, not as instructions.

</details>

## Go deeper

All opened on 7 October 2026.

- Model Context Protocol specification, revision 2026-07-28: overview, base protocol, versioning, tools, discovery, multi round-trip requests, Streamable HTTP and stdio transports, authorisation, and the changelog against 2025-11-25. Revision 2025-11-25 lifecycle page for the old handshake.
- Agentic AI Foundation, "MCP 2026-07-28: what's changing and how to migrate", 21 July 2026, and the MCP blog post "2026-07-28 release candidate", 21 May 2026.
- Agent2Agent protocol specification, version 1.0.0, the Agent discovery guide and the Python tutorial pages at a2a-protocol.org. Google Open Source Blog, "A year of open collaboration: celebrating the anniversary of A2A", April 2026.
- RFC 9728 (protected resource metadata), RFC 8707 (resource indicators) and RFC 9207 (issuer identification), cited by the MCP authorisation section.
- Libraries used: `mcp` 2.3.0 and `a2a-sdk` 1.2.2, installed from PyPI.
- On this site: [MCP architecture](/docs/mcp/mcp-architecture) and [lifecycle](/docs/mcp/mcp-lifecycle) (older revisions), [building MCP clients](/docs/mcp/build-mcp-clients), [project 3: an enterprise MCP gateway](/docs/mcp/project-3-enterprise-mcp-gateway), [MCP client with LangGraph](/docs/agentic-ai/mcp-client-langgraph), [multi-agent systems in LangChain](/docs/genai/langchain-advanced/multi-agent-systems), [support agent platform case](/docs/senior/design-customer-support-agent-platform).

**Not verified here.** The MCP authorisation flow (read, not run); MCP `subscriptions/listen`, the Tasks extension and Client ID Metadata Documents; A2A gRPC and HTTP+JSON bindings, streaming, push notifications and signed Agent Cards (read, not run); the error code the A2A specification assigns to a version error on JSON-RPC; any server or agent from another vendor. The measured byte counts are for the Python SDK's JSON on stdio and will differ with another SDK.

## Check yourself

- I can read a JSON-RPC request and its reply and match them by id.
- I can explain why MCP dropped the handshake and the session, and what each request carries instead.
- I can describe the multi round-trip pattern and why the state must be protected.
- I can explain what an Agent Card is and walk through an A2A task from submission to completion.
- I can choose between a tool and an agent for a given capability, and say what to distrust in each.

## Where to go next

Next: [Computer use and browser agents](/docs/agentic-frontier/computer-use-and-browser-agents), where the tools are a screen and a keyboard. Related: [Context engineering](/docs/agentic-frontier/context-engineering) for the cost of every tool you attach.
