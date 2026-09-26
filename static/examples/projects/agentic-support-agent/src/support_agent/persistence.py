"""Checkpointer (short-term memory) and Store (long-term memory) factories.

memory   : InMemorySaver + InMemoryStore          (unit tests)
sqlite   : AsyncSqliteSaver + AsyncSqliteStore     (local dev, single process)
postgres : AsyncPostgresSaver + AsyncPostgresStore (compose / production)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from support_agent.config import Settings


@asynccontextmanager
async def open_persistence(
    settings: Settings,
) -> AsyncIterator[tuple[BaseCheckpointSaver, BaseStore]]:
    backend = settings.checkpoint_backend
    if backend == "memory":
        yield InMemorySaver(), InMemoryStore()
        return
    async with AsyncExitStack() as stack:
        if backend == "sqlite":
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            from langgraph.store.sqlite.aio import AsyncSqliteStore

            for path in (settings.checkpoint_sqlite_path, settings.store_sqlite_path):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            saver = await stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string(settings.checkpoint_sqlite_path)
            )
            store = await stack.enter_async_context(
                AsyncSqliteStore.from_conn_string(settings.store_sqlite_path)
            )
        else:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            from langgraph.store.postgres.aio import AsyncPostgresStore

            assert settings.postgres_url
            saver = await stack.enter_async_context(
                AsyncPostgresSaver.from_conn_string(settings.postgres_url)
            )
            store = await stack.enter_async_context(
                AsyncPostgresStore.from_conn_string(settings.postgres_url)
            )
        await saver.setup()  # idempotent: creates tables on first run
        await store.setup()
        yield saver, store
