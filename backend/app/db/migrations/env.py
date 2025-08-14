from __future__ import annotations

from logging.config import fileConfig
import sys
from pathlib import Path
import os

from sqlalchemy import pool
from sqlalchemy import engine_from_config, text
from alembic import context

# Make sure '/app' is on sys.path when alembic runs from various CWDs
sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.core.config import get_settings
from app.db.base import Base
from app.models import tenant, user, doc  # noqa: F401  Ensure models imported for metadata


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url() -> str:
    url = os.getenv("MIGRATIONS_DSN") or get_settings().database_url
    return url.replace("+asyncpg", "")


def run_migrations_offline() -> None:
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table="alembic_version",
        version_table_schema="app",
        include_schemas=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        {"sqlalchemy.url": get_url()},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        # Ensure we create objects in the 'app' schema first
        connection.execute(text("SET search_path TO app, public"))
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table="alembic_version",
            version_table_schema="app",
            include_schemas=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()


