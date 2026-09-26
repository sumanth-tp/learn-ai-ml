"""The RAG pipeline: guard -> retrieve -> rerank -> generate, with timings, tokens and cost."""

from __future__ import annotations

import time

from langsmith import traceable

from ragate.config import PipelineConfig, PriceTable, load_prices
from ragate.fakes import REFUSAL_TEXT
from ragate.log import get_logger
from ragate.models import RagAnswer, StageTimings
from ragate.providers import chat_model, embeddings
from ragate.rag.corpus import load_corpus
from ragate.rag.generator import Generator, is_refusal, parse_citations
from ragate.rag.guards import check_input
from ragate.rag.index import HandbookIndex
from ragate.rag.rerank import LexicalReranker, LLMReranker, NoReranker, Reranker
from ragate.rag.retriever import Retriever
from ragate.settings import Settings

log = get_logger(__name__)


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


class RagPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        retriever: Retriever,
        reranker: Reranker,
        generator: Generator,
        prices: PriceTable,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.prices = prices

    @traceable(name="rag_ask", run_type="chain")
    def ask(self, question: str) -> RagAnswer:
        timings = StageTimings()
        model = self.generator.model_name
        if self.config.input_guard and (reason := check_input(question)):
            log.info("input_guard_refusal", reason=reason)
            return RagAnswer(question=question, answer=REFUSAL_TEXT, refused=True, model=model)

        t = time.perf_counter()
        candidates = self.retriever.retrieve(question, self.config.fetch_k)
        timings.retrieve_ms = _ms(t)

        t = time.perf_counter()
        contexts = self.reranker.rerank(question, candidates, self.config.k)
        timings.rerank_ms = _ms(t)

        t = time.perf_counter()
        answer, usage = self.generator.generate(question, contexts)
        timings.generate_ms = _ms(t)

        return RagAnswer(
            question=question,
            answer=answer,
            citations=parse_citations(answer),
            contexts=contexts,
            refused=is_refusal(answer),
            model=model,
            usage=usage,
            cost_usd=self.prices.cost(model, usage),
            timings=timings,
        )


def build_pipeline(config: PipelineConfig, settings: Settings) -> RagPipeline:
    model_name = config.generator_model or settings.chat_model
    emb = embeddings(settings)
    embedding_id = f"{settings.embedding_provider}:{settings.embedding_model}"
    docs = load_corpus(settings.corpus_dir)
    index = HandbookIndex.build_or_load(docs, config, emb, embedding_id, settings.index_dir)
    retriever = Retriever(index, emb, hybrid=config.hybrid, rrf_k=config.rrf_k)
    reranker: Reranker
    if config.reranker == "lexical":
        reranker = LexicalReranker()
    elif config.reranker == "llm":
        reranker = LLMReranker(chat_model(settings, role="rerank", model=model_name))
    else:
        reranker = NoReranker()
    generator = Generator(
        chat_model(settings, role="generator", model=model_name),
        model_name,
        attempts=settings.max_retries,
        pii_redaction=config.pii_redaction,
    )
    prices = load_prices(settings.config_dir / "pricing.yaml")
    return RagPipeline(config, retriever, reranker, generator, prices)
