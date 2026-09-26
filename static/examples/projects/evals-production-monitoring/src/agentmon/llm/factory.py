"""Provider-agnostic model construction. Callers depend on BaseChatModel only."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from agentmon.clock import Clock
from agentmon.config import Settings
from agentmon.llm.fake_agent import ScriptedBankingModel
from agentmon.llm.fake_judge import RuleJudgeModel


def _openai(settings: Settings, model: str) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        temperature=0,
    )


def build_agent_model(settings: Settings, clock: Clock | None = None) -> BaseChatModel:
    if settings.llm_provider == "fake":
        return ScriptedBankingModel(clock=clock)
    return _openai(settings, settings.model_name)


def build_judge_model(settings: Settings, clock: Clock | None = None) -> BaseChatModel:
    if settings.llm_provider == "fake":
        return RuleJudgeModel(clock=clock)
    return _openai(settings, settings.judge_model_name)
