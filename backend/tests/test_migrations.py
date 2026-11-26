"""
Alembic migrations run end to end against a throwaway SQLite file.

The production container runs ``alembic upgrade head`` before uvicorn, on a
brand-new Postgres database. These tests cover the parts that used to fail
there: the URL must come from DATABASE_URL (not alembic.ini), and the first
revision must work on a database that has no tables yet.
"""
import asyncio
from pathlib import Path

import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _config() -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    # Safety net: never fall back to the placeholder Postgres URL in alembic.ini.
    cfg.set_main_option("sqlalchemy.url", "sqlite+aiosqlite:///:memory:")
    return cfg


def _head() -> str:
    return ScriptDirectory.from_config(_config()).get_current_head()


def _upgrade(monkeypatch, db_file: Path) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_file}")
    command.upgrade(_config(), "head")


def _inspect(db_file: Path):
    engine = sa.create_engine(f"sqlite:///{db_file}")
    return engine, sa.inspect(engine)


def test_fresh_database_upgrades_to_head(tmp_path, monkeypatch):
    db_file = tmp_path / "fresh.db"
    _upgrade(monkeypatch, db_file)

    engine, insp = _inspect(db_file)
    try:
        tables = set(insp.get_table_names())
        assert {"source_configs", "cycles", "collected_data", "summaries",
                "extracted_mentions", "alembic_version"} <= tables
        assert "purpose" in {c["name"] for c in insp.get_columns("source_configs")}
        assert "mentioned_at" in {c["name"] for c in insp.get_columns("extracted_mentions")}
        with engine.connect() as conn:
            version = conn.execute(sa.text("select version_num from alembic_version")).scalar_one()
        assert version == _head()
    finally:
        engine.dispose()


def test_upgrade_twice_is_a_no_op(tmp_path, monkeypatch):
    db_file = tmp_path / "twice.db"
    _upgrade(monkeypatch, db_file)
    _upgrade(monkeypatch, db_file)  # the container runs it on every start

    engine, insp = _inspect(db_file)
    try:
        assert "extracted_mentions" in insp.get_table_names()
    finally:
        engine.dispose()


def test_database_created_by_create_all_before_the_feeder_existed(tmp_path, monkeypatch):
    """A dev database from before the feeder: tables exist, the new bits do not."""
    from src.models import Base

    db_file = tmp_path / "legacy.db"
    engine = sa.create_engine(f"sqlite:///{db_file}")
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(sa.text("drop table extracted_mentions"))
        conn.execute(sa.text("alter table source_configs drop column purpose"))
    engine.dispose()

    _upgrade(monkeypatch, db_file)

    engine, insp = _inspect(db_file)
    try:
        assert "purpose" in {c["name"] for c in insp.get_columns("source_configs")}
        assert "mentioned_at" in {c["name"] for c in insp.get_columns("extracted_mentions")}
    finally:
        engine.dispose()


def test_env_py_does_not_disable_other_loggers():
    """alembic's fileConfig must not silence the app's loggers for later tests."""
    import logging

    logger = logging.getLogger("src.services.scheduler_service")
    assert not logger.disabled
