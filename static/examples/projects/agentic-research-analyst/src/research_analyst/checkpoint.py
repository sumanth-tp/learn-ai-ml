"""Checkpointer factory with an explicit deserialisation allowlist.

LangGraph checkpoints Pydantic objects with msgpack. Newer versions warn (and will
later refuse) to rebuild classes that are not allow-listed, because deserialising
arbitrary types from a database is a code-execution risk. We list exactly our models.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from enum import StrEnum
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

from research_analyst import models


def _allowlist() -> list[tuple[str, str]]:
    allowed = []
    for name, obj in vars(models).items():
        if (
            inspect.isclass(obj)
            and obj.__module__ == models.__name__
            and (issubclass(obj, BaseModel) or issubclass(obj, StrEnum))
        ):
            allowed.append((models.__name__, name))
    return allowed


def make_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=_allowlist())


def memory_checkpointer() -> InMemorySaver:
    return InMemorySaver(serde=make_serde())


@asynccontextmanager
async def sqlite_checkpointer(path: Path) -> AsyncIterator[AsyncSqliteSaver]:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(path))
    try:
        await conn.execute("PRAGMA journal_mode=WAL")  # readers do not block the writer
        saver = AsyncSqliteSaver(conn, serde=make_serde())
        await saver.setup()
        yield saver
    finally:
        await conn.close()
