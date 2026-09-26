"""Versioned pipeline configuration (YAML) and pricing.

A PipelineConfig is the unit of an experiment: two runs are comparable only if you
know exactly which config produced each, so its hash is stored in every run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from ragate.models import Usage


class PipelineConfig(BaseModel):
    name: str = "baseline"
    chunk_size: int = Field(default=400, ge=100, le=4000)
    chunk_overlap: int = Field(default=80, ge=0)
    fetch_k: int = Field(default=20, ge=1, le=100)
    k: int = Field(default=4, ge=1, le=20)
    hybrid: bool = True
    rrf_k: int = 60
    reranker: Literal["none", "lexical", "llm"] = "lexical"
    generator_model: str | None = None
    include_restricted: bool = False
    pii_redaction: bool = True
    input_guard: bool = True

    @model_validator(mode="after")
    def _check(self) -> PipelineConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.k > self.fetch_k:
            raise ValueError("k cannot exceed fetch_k")
        return self

    def config_hash(self) -> str:
        payload = self.model_dump(exclude={"name"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    def with_overrides(self, name: str, overrides: dict[str, Any]) -> PipelineConfig:
        return PipelineConfig.model_validate({**self.model_dump(), **overrides, "name": name})


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def load_pipeline_config(path: Path) -> PipelineConfig:
    return PipelineConfig.model_validate(load_yaml(path))


class Price(BaseModel):
    input_per_1m: float
    output_per_1m: float = 0.0


class PriceTable(BaseModel):
    models: dict[str, Price]

    def cost(self, model: str, usage: Usage) -> float:
        price = self.models.get(model)
        if price is None:
            return 0.0
        return (
            usage.input_tokens * price.input_per_1m + usage.output_tokens * price.output_per_1m
        ) / 1_000_000


def load_prices(path: Path) -> PriceTable:
    return PriceTable.model_validate(load_yaml(path))
