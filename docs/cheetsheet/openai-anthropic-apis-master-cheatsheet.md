---
title: OpenAI and Anthropic APIs Master Cheatsheet
sidebar_position: 26
---

# OpenAI and Anthropic APIs Master Cheatsheet

## Client setup

| Method | Description | Code example |
|---|---|---|
| OpenAI install | Install the official Python SDK. | `pip install openai` |
| OpenAI client | `OpenAI(api_key=None)` reads `OPENAI_API_KEY` by default. | `from openai import OpenAI`<br/>`client = OpenAI()` |
| OpenAI basic response | Responses API is OpenAI's primary interface for text, images, tools, and stateful interactions. | `response = client.responses.create(model="gpt-5.1", input="Explain gradient descent in one paragraph.")`<br/>`print(response.output_text)` |
| Anthropic install | Install the official Anthropic Python SDK. | `pip install anthropic` |
| Anthropic client | `Anthropic(api_key=None)` reads `ANTHROPIC_API_KEY` by default. | `from anthropic import Anthropic`<br/>`client = Anthropic()` |
| Anthropic basic message | Messages API sends full conversation history and returns assistant content blocks. | `message = client.messages.create(model="claude-opus-4-1-20250805", max_tokens=512, messages=[{"role": "user", "content": "Explain gradient descent."}])`<br/>`print(message.content[0].text)` |

## Chat, streaming, and structured output

| Method | Description | Code example |
|---|---|---|
| OpenAI messages input | Responses accepts string input or role/content message items. | `response = client.responses.create(model="gpt-5.1", input=[{"role": "user", "content": "Summarize this text."}])` |
| OpenAI streaming | `stream=True` returns semantic streaming events. | `stream = client.responses.create(model="gpt-5.1", input="Count to five.", stream=True)`<br/>`for event in stream:`<br/>`    if event.type == "response.output_text.delta":`<br/>`        print(event.delta, end="")` |
| OpenAI structured output | Use JSON schema formatting for machine-readable responses. | `schema = {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"], "additionalProperties": False}`<br/>`response = client.responses.create(model="gpt-5.1", input="Classify: great movie", text={"format": {"type": "json_schema", "name": "classification", "schema": schema, "strict": True}})` |
| Anthropic messages | Messages are stateless; send the needed conversation each time. | `messages = [{"role": "user", "content": "Write a haiku about GPUs."}]`<br/>`client.messages.create(model=model, max_tokens=128, messages=messages)` |
| Anthropic streaming | Stream text and events from Messages API. | `with client.messages.stream(model=model, max_tokens=256, messages=messages) as stream:`<br/>`    for text in stream.text_stream:`<br/>`        print(text, end="")` |
| Anthropic JSON-style output | Ask for JSON and validate it in your code. For strict schemas, use tool input schemas. | `message = client.messages.create(model=model, max_tokens=256, messages=[{"role": "user", "content": "Return JSON with keys label and confidence."}])` |

## Tool and function calling

| Method | Description | Code example |
|---|---|---|
| OpenAI function tool | Define a tool with name, description, and JSON schema parameters. | `tools = [{"type": "function", "name": "get_weather", "description": "Get weather by city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": False}}]`<br/>`response = client.responses.create(model="gpt-5.1", input="Weather in Paris?", tools=tools)` |
| OpenAI tool result loop | Execute tool calls and send results back as follow-up input. | `for item in response.output:`<br/>`    if item.type == "function_call":`<br/>`        result = get_weather(**json.loads(item.arguments))`<br/>`        followup = client.responses.create(model="gpt-5.1", previous_response_id=response.id, input=[{"type": "function_call_output", "call_id": item.call_id, "output": json.dumps(result)}])` |
| Anthropic client tool | Define `tools` with `input_schema`; Claude returns `tool_use` blocks. | `tools = [{"name": "get_weather", "description": "Get weather by city", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]`<br/>`message = client.messages.create(model=model, max_tokens=512, tools=tools, messages=messages)` |
| Anthropic tool result | Return results as user `tool_result` content blocks. | `tool_result = {"type": "tool_result", "tool_use_id": tool_use.id, "content": json.dumps(result)}`<br/>`client.messages.create(model=model, max_tokens=512, tools=tools, messages=messages + [{"role": "assistant", "content": message.content}, {"role": "user", "content": [tool_result]}])` |
| Tool choice | Force or restrict tool usage when needed. | `response = client.responses.create(model="gpt-5.1", input="Use the weather tool.", tools=tools, tool_choice={"type": "function", "name": "get_weather"})` |

## Batch APIs and prompt caching

| Method | Description | Code example |
|---|---|---|
| OpenAI Batch API | Submit asynchronous request files for offline jobs such as evals and classification. | `file = client.files.create(file=open("batch.jsonl", "rb"), purpose="batch")`<br/>`batch = client.batches.create(input_file_id=file.id, endpoint="/v1/responses", completion_window="24h")` |
| OpenAI batch status | Poll batch until complete, then download output file. | `batch = client.batches.retrieve(batch.id)`<br/>`if batch.status == "completed":`<br/>`    content = client.files.content(batch.output_file_id)` |
| OpenAI prompt caching | Recent models cache long repeated prompt prefixes automatically; keep static instructions first. | `response = client.responses.create(model="gpt-5.1", input=long_static_prefix + user_question, prompt_cache_key="support-bot-v1")` |
| OpenAI extended cache | Some models support `prompt_cache_retention="24h"` for longer-lived cache prefixes. | `response = client.responses.create(model="gpt-5.1", input=prompt, prompt_cache_retention="24h")` |
| Anthropic prompt caching | Add `cache_control` to reusable system or message content blocks. | `system = [{"type": "text", "text": long_policy, "cache_control": {"type": "ephemeral"}}]`<br/>`client.messages.create(model=model, max_tokens=512, system=system, messages=messages)` |
| Anthropic batch processing | Use Message Batches for large asynchronous workloads. | `batch = client.messages.batches.create(requests=[{"custom_id": "row-1", "params": {"model": model, "max_tokens": 128, "messages": messages}}])` |

## Reliability, safety, and cost control

| Method | Description | Code example |
|---|---|---|
| Timeout | Bound request latency. | `client = OpenAI(timeout=30.0)` |
| Retry wrapper | Retry transient failures with backoff. | `for attempt in range(5):`<br/>`    try:`<br/>`        return call_model()`<br/>`    except Exception:`<br/>`        time.sleep(2 ** attempt)` |
| Token budget | Set output limits. | `client.responses.create(model="gpt-5.1", input=prompt, max_output_tokens=500)`<br/>`client.messages.create(model=model, max_tokens=500, messages=messages)` |
| Validate JSON | Never trust model JSON without validation. | `data = json.loads(response.output_text)`<br/>`validated = MySchema.model_validate(data)` |
| Redact secrets | Remove API keys and PII before sending prompts. | `safe_text = re.sub(r"sk-[A-Za-z0-9_-]+", "sk-REDACTED", text)` |
| Log IDs and usage | Store request IDs, model, latency, token usage, and errors. | `logger.info("model_call", extra={"model": model, "latency": latency, "usage": response.usage})` |

## Common patterns

| Method | Description | Code example |
|---|---|---|
| OpenAI classification | Structured output for labels. | `schema = {"type": "object", "properties": {"label": {"type": "string"}, "confidence": {"type": "number"}}, "required": ["label", "confidence"], "additionalProperties": False}`<br/>`response = client.responses.create(model="gpt-5.1", input=text, text={"format": {"type": "json_schema", "name": "classification", "schema": schema, "strict": True}})` |
| Anthropic extraction | Ask for structured JSON and parse it. | `message = client.messages.create(model=model, max_tokens=300, messages=[{"role": "user", "content": f"Extract company, person, and date as JSON: {text}"}])` |
| Streaming UI | Print deltas as they arrive. | `for event in stream:`<br/>`    if getattr(event, "type", "") == "response.output_text.delta":`<br/>`        ui.write(event.delta)` |
| Tool router | Dispatch tool calls by name. | `handlers = {"get_weather": get_weather, "search_docs": search_docs}`<br/>`result = handlers[name](**arguments)` |
| Long context caching | Put stable docs before user question. | `prompt = system_instructions + retrieved_docs + user_question`<br/>`client.responses.create(model="gpt-5.1", input=prompt, prompt_cache_key="docs-v3")` |
| Batch classification | Write one JSONL request per row. | `{"custom_id": "row-1", "method": "POST", "url": "/v1/responses", "body": {"model": "gpt-5.1", "input": "Classify this."}}` |
| Provider adapter | Hide provider differences behind one interface. | `class LLMClient:`<br/>`    def complete(self, messages):`<br/>`        raise NotImplementedError` |
| Eval regression | Compare outputs after prompt/model changes. | `for case in eval_cases:`<br/>`    output = llm.complete(case.input)`<br/>`    score = judge(case.expected, output)` |

## Senior API architecture

| Method | Description | Code example |
|---|---|---|
| Provider abstraction | Keep product code independent from provider SDKs. | `class ChatProvider(Protocol):`<br/>`    def complete(self, messages: list[dict], *, schema: Optional[type[BaseModel]] = None) -> str:`<br/>`        ...` |
| Capability registry | Route requests by required capabilities, not provider names. | `capabilities = {"structured_output": ["openai"], "long_context": ["anthropic", "openai"]}` |
| Model config object | Centralize model, timeout, token limit, and cost policy. | `@dataclass(frozen=True)`<br/>`class ModelConfig:`<br/>`    provider: str`<br/>`    model: str`<br/>`    max_tokens: int`<br/>`    timeout_s: float` |
| Fallback policy | Fall back only for safe, idempotent use cases and record it. | `try:`<br/>`    return primary.complete(messages)`<br/>`except TransientLLMError:`<br/>`    return fallback.complete(messages)` |
| Circuit breaker | Stop hammering a failing provider. | `if breaker.is_open("openai"):`<br/>`    raise ServiceUnavailable("provider temporarily disabled")` |
| Budget guard | Estimate cost before sending large requests. | `if estimated_input_tokens > config.max_input_tokens:`<br/>`    raise ValueError("prompt too large")` |
| Prompt versioning | Treat prompts like deployable artifacts. | `prompt = prompt_store.get("support_triage", version="2026-05-16")` |
| Response provenance | Store provider, model, prompt version, and request ID. | `trace = {"provider": provider, "model": model, "prompt_version": version, "request_id": request_id}` |

## Safety, evaluation, and operations

| Method | Description | Code example |
|---|---|---|
| Schema validation boundary | Validate model output before business logic sees it. | `parsed = OutputSchema.model_validate_json(response.output_text)` |
| Tool allowlist | Never dispatch arbitrary tool names from model output. | `if tool_name not in allowed_tools:`<br/>`    raise SecurityError("unknown tool")` |
| Tool argument validation | Validate tool arguments with Pydantic before execution. | `args = WeatherArgs.model_validate_json(item.arguments)`<br/>`result = get_weather(args.city)` |
| Prompt injection boundary | Separate trusted instructions, retrieved content, and user content. | `input = [{"role": "developer", "content": policy}, {"role": "user", "content": user_text}]` |
| Retrieval citation check | Require answers to cite retrieved document IDs. | `if not set(answer.citations).issubset(retrieved_doc_ids):`<br/>`    raise ValueError("invalid citation")` |
| Golden eval set | Run stable regression cases before prompt/model changes ship. | `for case in golden_set:`<br/>`    assert evaluate(llm(case.input), case.expected) >= case.min_score` |
| Latency histogram | Track p50, p95, p99 by provider and model. | `LLM_LATENCY.labels(provider, model).observe(elapsed)` |
| Token usage accounting | Log input, output, cached, and reasoning tokens when available. | `usage_log.write({"model": model, "usage": response.usage})` |
| PII minimization | Redact or hash sensitive fields before model calls. | `payload["email_hash"] = hashlib.sha256(email.encode()).hexdigest()`<br/>`payload.pop("email")` |
| Human handoff | Escalate low-confidence or high-risk outputs. | `if result.confidence < 0.7 or result.risk == "high":`<br/>`    return create_review_ticket(result)` |

## Advanced tool and batch patterns

| Method | Description | Code example |
|---|---|---|
| Deterministic tool loop | Continue until no tool calls or max turns is reached. | `for _ in range(max_tool_turns):`<br/>`    response = call_model(messages, tools=tools)`<br/>`    calls = extract_tool_calls(response)`<br/>`    if not calls: break`<br/>`    messages.extend(run_tools(calls))` |
| Parallel tool execution | Execute independent tool calls concurrently. | `results = await asyncio.gather(*(run_tool(call) for call in calls))` |
| Batch JSONL writer | Generate stable custom IDs for result joins. | `row = {"custom_id": f"ticket-{ticket.id}", "method": "POST", "url": "/v1/responses", "body": body}` |
| Batch result join | Join async results back to source records by ID. | `results_by_id = {row["custom_id"]: row for row in output_rows}`<br/>`df["llm_label"] = df.id.map(lambda id: results_by_id[f"ticket-{id}"])` |
| Cache-friendly prompt layout | Put stable policy, tools, and examples before variable user input. | `prompt = static_policy + examples + retrieved_context + user_question` |
| Streaming accumulator | Keep UI streaming separate from final parsed object. | `chunks = []`<br/>`for delta in stream_text():`<br/>`    ui.write(delta)`<br/>`    chunks.append(delta)`<br/>`final_text = "".join(chunks)` |
| Replayable traces | Persist enough context to reproduce a bad output. | `trace = {"messages": messages, "tools": tools, "model_config": asdict(config)}` |
| Red-team suite | Test prompt injection, tool abuse, data exfiltration, and refusal boundaries. | `for attack in red_team_cases:`<br/>`    result = llm.complete(attack.prompt)`<br/>`    assert not attack.succeeds(result)` |
