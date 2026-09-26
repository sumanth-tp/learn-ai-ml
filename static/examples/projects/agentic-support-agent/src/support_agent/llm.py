"""Model factories. Swapping provider or model is a config change, not a code change."""

from __future__ import annotations

import logging

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from support_agent.config import Settings
from support_agent.fakes import ScriptedSupportModel
from support_agent.intent import IntentClassifier, KeywordIntentClassifier, LLMIntentClassifier
from support_agent.services.faq import HashingEmbeddings

log = logging.getLogger(__name__)


def build_chat_model(settings: Settings) -> BaseChatModel:
    if settings.fake_llm:
        return ScriptedSupportModel()
    kwargs: dict[str, object] = {
        "temperature": settings.llm_temperature,
        "timeout": settings.llm_timeout_s,
        "max_retries": settings.llm_max_retries,
    }
    if settings.llm_provider == "openai" and settings.openai_api_key:
        kwargs["api_key"] = settings.openai_api_key.get_secret_value()
    if settings.llm_provider == "anthropic" and settings.anthropic_api_key:
        kwargs["api_key"] = settings.anthropic_api_key.get_secret_value()
    model = init_chat_model(settings.llm_model, model_provider=settings.llm_provider, **kwargs)
    assert isinstance(model, BaseChatModel)
    return model


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.fake_llm or settings.llm_provider != "openai":
        # Lexical local embeddings: fine for ten FAQ articles, no provider needed.
        return HashingEmbeddings()
    key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    return init_embeddings(settings.embeddings_model, provider="openai", api_key=key)


def build_classifier(settings: Settings, model: BaseChatModel) -> IntentClassifier:
    if settings.fake_llm:
        return KeywordIntentClassifier()
    return LLMIntentClassifier(model)
