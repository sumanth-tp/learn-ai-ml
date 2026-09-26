"""Run and item results: the artefacts the gate compares."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, Field


class ItemResult(BaseModel):
    item_id: str
    question_type: str
    expected_behaviour: str
    question: str
    answer: str = ""
    reference: str = ""
    citations: list[str] = Field(default_factory=list)
    retrieved: list[str] = Field(default_factory=list)
    refused: bool = False
    pii_leaks: list[str] = Field(default_factory=list)
    scores: dict[str, float | None] = Field(default_factory=dict)
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    errors: list[str] = Field(default_factory=list)


class JudgeInfo(BaseModel):
    model_id: str
    backend: str
    geval_model: str
    prompts: dict[str, dict[str, str]]
    deepeval_version: str

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.model_dump(), sort_keys=True).encode()
        ).hexdigest()[:12]


class RunResult(BaseModel):
    run_id: str
    name: str
    created_at: str
    git_sha: str
    code_version: str = ""
    config: dict
    config_hash: str
    dataset_version: str
    dataset_sha: str
    judge: JudgeInfo
    provider: dict[str, str]
    duration_s: float
    items: list[ItemResult]
    aggregates: dict[str, float | None]
