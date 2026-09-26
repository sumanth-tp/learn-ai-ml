"""Provider factory: the only place that knows which concrete chat model or embeddings run.

Everything else receives a LangChain ``BaseChatModel`` / ``Embeddings`` and never
imports a vendor SDK directly, so swapping provider is a config change.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from ragate.fakes import ExtractiveChatModel, FakeSynthChatModel, HashingEmbeddings
from ragate.settings import Provider, Settings

Role = Literal["generator", "judge", "synth", "rerank"]


class ProviderConfigError(RuntimeError):
    """Raised when a provider is selected but is not usable (missing key or package)."""


def _require(value: object, env_name: str, provider: str) -> None:
    if not value:
        raise ProviderConfigError(
            f"provider '{provider}' needs {env_name}; set it in .env or use the fake provider"
        )


def chat_model(
    settings: Settings,
    *,
    role: Role,
    model: str | None = None,
    temperature: float = 0.0,
    provider: Provider | None = None,
) -> BaseChatModel:
    provider = provider or (settings.judge_provider if role == "judge" else settings.provider)
    name = model or (settings.judge_model if role == "judge" else settings.chat_model)

    if provider == "fake":
        if role == "synth":
            return FakeSynthChatModel()
        if role in ("generator", "rerank"):
            return ExtractiveChatModel.for_model(name)
        raise ProviderConfigError(
            "the fake provider has no LLM judge; the offline judge is HeuristicJudge"
        )
    if provider == "openai":
        _require(settings.openai_api_key, "OPENAI_API_KEY", provider)
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=name,
            temperature=temperature,
            timeout=settings.request_timeout_s,
            max_retries=settings.max_retries,
            seed=7,
            api_key=settings.openai_api_key,
            stream_usage=True,
        )
    if provider == "anthropic":
        _require(settings.anthropic_api_key, "ANTHROPIC_API_KEY", provider)
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # optional extra
            raise ProviderConfigError("run `uv add langchain-anthropic` first") from exc
        return ChatAnthropic(
            model_name=name,
            temperature=temperature,
            timeout=settings.request_timeout_s,
            max_retries=settings.max_retries,
            api_key=settings.anthropic_api_key,
        )
    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ProviderConfigError("run `uv add langchain-ollama` first") from exc
        return ChatOllama(model=name, temperature=temperature, base_url=settings.ollama_base_url)
    raise ProviderConfigError(f"unknown provider {provider!r}")


def embeddings(settings: Settings) -> Embeddings:
    provider = settings.embedding_provider
    if provider == "fake":
        return HashingEmbeddings()
    if provider == "openai":
        _require(settings.openai_api_key, "OPENAI_API_KEY", provider)
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            timeout=settings.request_timeout_s,
            max_retries=settings.max_retries,
        )
    if provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as exc:
            raise ProviderConfigError("run `uv add langchain-ollama` first") from exc
        return OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_base_url)
    raise ProviderConfigError(f"provider {provider!r} does not offer embeddings")
