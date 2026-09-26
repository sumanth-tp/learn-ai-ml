"""Migrations apply cleanly, match the ORM models, and roll back."""

from __future__ import annotations

from pathlib import Path

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect

from helpdesk_mcp.db.migrate import downgrade, upgrade
from helpdesk_mcp.db.models import Base


def test_upgrade_matches_models_and_downgrade_is_clean(tmp_path: Path) -> None:
    db = tmp_path / "migrate.db"
    upgrade(f"sqlite+aiosqlite:///{db}")

    sync_engine = create_engine(f"sqlite:///{db}")
    with sync_engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    # An empty diff means `alembic revision --autogenerate` would produce nothing:
    # the migration and the models describe the same schema.
    assert diff == [], diff

    downgrade(f"sqlite+aiosqlite:///{db}", "base")
    assert set(inspect(sync_engine).get_table_names()) == {"alembic_version"}
    sync_engine.dispose()


def test_upgrade_is_idempotent(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'twice.db'}"
    upgrade(url)
    upgrade(url)  # second run is a no-op, which is what compose relies on
