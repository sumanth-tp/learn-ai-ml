"""Dependency container: the only place that decides fake vs real providers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from research_analyst.config import Settings
from research_analyst.index import InternalIndex
from research_analyst.providers.embeddings import build_embeddings
from research_analyst.providers.llm import Brain, build_brain
from research_analyst.providers.search import WebSearch, build_web_search


@dataclass
class Deps:
    settings: Settings
    brain: Brain
    judge: Brain
    index: InternalIndex
    search: WebSearch


def index_path_for(settings: Settings) -> str:
    """Vectors from different embedding models must never be mixed in one index file."""
    tag = "hashing" if settings.mode == "offline" else settings.embedding_model
    p = settings.index_path
    return str(p.with_name(f"{p.stem}-{tag}{p.suffix}"))


def build_deps(settings: Settings) -> Deps:
    embeddings = build_embeddings(settings)
    index = InternalIndex.load_or_build(
        Path(index_path_for(settings)), settings.corpus_dir / "internal", embeddings
    )
    return Deps(
        settings=settings,
        brain=build_brain(settings),
        judge=build_brain(settings, judge=True),
        index=index,
        search=build_web_search(settings),
    )
