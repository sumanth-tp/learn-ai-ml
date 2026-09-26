"""Composition root: builds every dependency from Settings in one place."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from support_agent.config import Settings
from support_agent.db import create_schema, make_engine, make_session_factory
from support_agent.graph import GraphDeps
from support_agent.llm import build_chat_model, build_classifier, build_embeddings
from support_agent.seed import seed
from support_agent.services.faq import FaqRetriever
from support_agent.services.orders import OrderService
from support_agent.services.refunds import (
    HttpRefundGateway,
    RefundGateway,
    RefundService,
    StubRefundGateway,
)
from support_agent.tools import ToolServices


@dataclass
class Container:
    settings: Settings
    engine: Engine
    sessions: sessionmaker[Session]
    gateway: RefundGateway
    deps: GraphDeps


def build_gateway(settings: Settings) -> RefundGateway:
    if settings.refund_gateway == "http":
        key = settings.refund_api_key.get_secret_value() if settings.refund_api_key else None
        return HttpRefundGateway(settings.refund_api_url, key, settings.refund_timeout_s)
    return StubRefundGateway(failure_rate=settings.stub_refund_failure_rate)


def build_container(
    settings: Settings, *, gateway: RefundGateway | None = None, do_seed: bool = True
) -> Container:
    engine = make_engine(settings.database_url)
    create_schema(engine)
    sessions = make_session_factory(engine)
    if do_seed:
        seed(sessions)
    gw = gateway or build_gateway(settings)
    model = build_chat_model(settings)
    tools = ToolServices(
        settings=settings,
        orders=OrderService(sessions),
        refunds=RefundService(sessions, gw),
        faq=FaqRetriever(build_embeddings(settings)),
    )
    deps = GraphDeps(
        settings=settings, model=model, classifier=build_classifier(settings, model), tools=tools
    )
    return Container(settings=settings, engine=engine, sessions=sessions, gateway=gw, deps=deps)
