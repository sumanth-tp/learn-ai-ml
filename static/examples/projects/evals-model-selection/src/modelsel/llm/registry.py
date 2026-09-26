"""The model catalogue (``data/models.toml``) and the factory that turns an id into a chat model.

A model id is ``provider:name`` (``openai:gpt-4o-mini``, ``anthropic:claude-haiku-4-5``,
``ollama:llama3.1:8b``, ``fake:balanced-mini``). Real providers go through
LangChain's ``init_chat_model``; ``fake`` builds a deterministic offline model with
the same ``BaseChatModel`` interface, so the harness cannot tell them apart.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from modelsel.config import Settings


class FakeProfile(BaseModel):
    skill: float = Field(default=0.8, ge=0, le=1)
    json_error_rate: float = Field(default=0.02, ge=0, le=1)
    verbosity: int = Field(default=0, ge=0, le=4)
    latency_ms: float = 800.0
    signature: str | None = None
    contaminated_on: list[str] = Field(default_factory=list)
    transient_failures: int = 0
    """Fail the first N calls of each distinct prompt with a 429, to exercise retries."""


class JudgeProfile(BaseModel):
    noise: float = 0.3
    position_bias: float = 0.0
    verbosity_bias: float = 0.0
    self_bonus: float = 0.0


class ModelSpec(BaseModel):
    id: str
    family: str
    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)
    rpm: int = Field(default=60, ge=1)
    supports_logprobs: bool = False
    fake: FakeProfile | None = None
    judge: JudgeProfile | None = None

    @property
    def provider(self) -> str:
        return self.id.split(":", 1)[0]


class Profile(BaseModel):
    candidates: list[str]
    baseline: str
    judge: str
    meta_judge: str


class DecisionConfig(BaseModel):
    weights: dict[str, float] = Field(default_factory=lambda: {"quality": 0.6, "cost": 0.25, "latency": 0.15})
    quality_weights: dict[str, float] = Field(
        default_factory=lambda: {"classification": 0.35, "extraction": 0.35, "reply": 0.30}
    )
    min_json_validity: float = 0.95
    max_p95_latency_ms: float = 4000.0
    max_cost_per_1k_tickets_usd: float = 5.0
    min_detectable_effect: float = 0.05


class Catalogue(BaseModel):
    models: dict[str, ModelSpec]
    profiles: dict[str, Profile]
    decision: DecisionConfig

    def spec(self, model_id: str) -> ModelSpec:
        try:
            return self.models[model_id]
        except KeyError as exc:
            raise KeyError(f"model {model_id!r} is not in models.toml") from exc


def load_catalogue(path: Path) -> Catalogue:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    models = {mid: ModelSpec(id=mid, **cfg) for mid, cfg in raw.get("models", {}).items()}
    cat = Catalogue(
        models=models,
        profiles={k: Profile(**v) for k, v in raw["profiles"].items()},
        decision=DecisionConfig(**raw.get("decision", {})),
    )
    for name, prof in cat.profiles.items():
        for mid in [*prof.candidates, prof.baseline, prof.judge, prof.meta_judge]:
            if mid not in cat.models:
                raise ValueError(f"profile {name!r} references unknown model {mid!r}")
        if prof.baseline not in prof.candidates:
            raise ValueError(f"profile {name!r}: baseline must be one of the candidates")
    return cat


def build_chat_model(spec: ModelSpec, settings: Settings, *, memorised: dict[str, Any] | None = None) -> BaseChatModel:
    """Create the chat model behind one catalogue entry.

    Provider SDK retries are switched off (``max_retries=0``) because the harness
    owns retries: two layers of retry multiply the worst-case latency and hide 429s.
    """
    if spec.provider == "fake":
        from modelsel.llm.fakes import FakeJudgeModel, FakeTicketModel

        if spec.judge is not None:
            return FakeJudgeModel(model_id=spec.id, profile=spec.judge, sleep_scale=settings.fake_sleep_scale)
        return FakeTicketModel(
            model_id=spec.id,
            profile=spec.fake or FakeProfile(),
            memorised=memorised or {},
            sleep_scale=settings.fake_sleep_scale,
        )
    kwargs: dict[str, Any] = {"temperature": 0}
    if spec.provider in {"openai", "anthropic"}:
        kwargs |= {"max_retries": 0, "timeout": settings.request_timeout_s}
    if spec.provider == "openai" and spec.supports_logprobs:
        kwargs |= {"logprobs": True, "top_logprobs": 5}
    return init_chat_model(spec.id, **kwargs)
