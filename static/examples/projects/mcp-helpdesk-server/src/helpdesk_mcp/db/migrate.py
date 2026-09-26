"""Run Alembic migrations programmatically (used by the CLI, compose and tests).

The migration scripts live inside the package, so they ship in the wheel and
the Docker image without copying extra folders.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


def alembic_config(database_url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


def upgrade(database_url: str, revision: str = "head") -> None:
    """Blocking. Call from sync code, or via ``asyncio.to_thread`` from async code,
    because env.py starts its own event loop."""
    command.upgrade(alembic_config(database_url), revision)


def downgrade(database_url: str, revision: str) -> None:
    command.downgrade(alembic_config(database_url), revision)
