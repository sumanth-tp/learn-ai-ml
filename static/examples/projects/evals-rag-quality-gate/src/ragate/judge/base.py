"""The Judge interface. Real (LLM) and stub (heuristic) judges both implement it."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from ragate.judge.prompts import JudgePrompt


class JudgeOutput(BaseModel):
    """Structured output requested from an LLM judge. Reason first, then score."""

    reason: str = Field(description="One sentence justifying the score")
    score: int = Field(ge=0, le=10)


class Verdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reason: str
    cached: bool = False


class JudgeError(RuntimeError):
    """The judge could not produce a valid verdict (after retries)."""


class Judge(Protocol):
    @property
    def model_id(self) -> str: ...

    def evaluate(self, prompt: JudgePrompt, variables: dict[str, str]) -> Verdict: ...
