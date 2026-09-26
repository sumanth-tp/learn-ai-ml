"""Sampling: a server asks the host's LLM to generate text.

Sampling turns every server into a potential LLM spender and a potential prompt
author. So the host decides, per server, whether it may sample at all, caps the
tokens, strips any tool access (the sampled model gets no tools) and traces the call.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from mcp import types
from mcp.client.session import ClientSession, SamplingFnT
from mcp.shared.context import RequestContext

from mcp_host.llm import to_langchain
from mcp_host.settings import PolicyConfig
from mcp_host.tracing import mcp_span

log = logging.getLogger(__name__)


def make_sampling_callback(
    server: str, policy: PolicyConfig, llm: BaseChatModel, max_tokens: int, model_label: str
) -> SamplingFnT | None:
    """Return a callback for servers allowed to sample, ``None`` otherwise.

    Returning ``None`` matters: the SDK then does not advertise the ``sampling``
    capability at all, so a well-behaved server never even asks.
    """
    if not policy.allow_sampling:
        return None

    async def sampling_callback(
        context: RequestContext[ClientSession, Any], params: types.CreateMessageRequestParams
    ) -> types.CreateMessageResult | types.ErrorData:
        budget = min(params.maxTokens, max_tokens)
        with mcp_span(server, "sampling/createMessage", max_tokens=budget) as span:
            if params.tools:
                return types.ErrorData(
                    code=types.INVALID_REQUEST, message="this host does not grant tools to sampling"
                )
            messages = to_langchain(params.messages, params.systemPrompt)
            try:
                reply = await llm.bind(max_tokens=budget).ainvoke(messages)
            except Exception as exc:
                log.warning("sampling failed", extra={"server": server, "error": str(exc)})
                return types.ErrorData(code=types.INTERNAL_ERROR, message="host LLM call failed")
            text = reply.content if isinstance(reply.content, str) else str(reply.content)
            span.set_attribute("mcp.sampling.output_chars", len(text))
            log.info("sampling served", extra={"server": server, "chars": len(text)})
            return types.CreateMessageResult(
                role="assistant",
                content=types.TextContent(type="text", text=text),
                model=model_label,
                stopReason="endTurn",
            )

    return sampling_callback
