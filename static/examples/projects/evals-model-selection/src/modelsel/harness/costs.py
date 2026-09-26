"""Cost accounting from token usage and the price table in models.toml."""

from __future__ import annotations

from modelsel.llm.registry import ModelSpec
from modelsel.schemas import Usage


def cost_usd(spec: ModelSpec, usage: Usage) -> float:
    """Prices are per million tokens. Local models (Ollama) are priced at 0 here;
    put your GPU-hour cost into models.toml if you want a fair comparison."""
    return (usage.input_tokens * spec.input_per_mtok + usage.output_tokens * spec.output_per_mtok) / 1_000_000
