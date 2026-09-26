"""Async engine and session factory.

One engine per process. SQLite gets ``foreign_keys=ON`` (off by default, which
would silently break ``ON DELETE CASCADE``); Postgres gets a sized pool with
``pool_pre_ping`` so a database failover does not hand out dead connections.
"""

from __future__ import annotations

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from helpdesk_mcp.config import Settings


def make_engine(settings: Settings) -> AsyncEngine:
    url = settings.database_url
    if url.startswith("sqlite"):
        engine = create_async_engine(url, echo=settings.db_echo)

        @event.listens_for(engine.sync_engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record) -> None:  # pragma: no cover - driver hook
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

        return engine
    return create_async_engine(
        url,
        echo=settings.db_echo,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_pool_size,
        pool_pre_ping=True,
        pool_recycle=1800,
    )


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker:
    # expire_on_commit=False: we serialise ORM objects after commit, and an
    # expired attribute would trigger lazy IO outside the session.
    return async_sessionmaker(engine, expire_on_commit=False)


async def ping(engine: AsyncEngine) -> bool:
    """Readiness probe: can we run a trivial query right now?"""
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    return True
