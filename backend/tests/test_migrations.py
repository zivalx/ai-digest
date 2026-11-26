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


# --- track record (chained after d41b7a90c2f1) -----------------------------------------
TR_MENTION_COLUMNS = {
    "is_recommendation", "action", "horizon", "entry_price", "target_price", "stop_price",
    "trigger_price", "quote", "said_at", "said_at_precision", "pushed_calls_to_trender",
    "trender_calls_response", "call_payload_json",
}


def test_single_head_and_track_record_chain():
    script = ScriptDirectory.from_config(_config())
    assert script.get_heads() == ["f7b9d1e3a5c7"]
    assert script.get_revision("d4f6b8c0e2a4").down_revision == "d41b7a90c2f1"
    chain = [r.revision for r in script.walk_revisions("base", "heads")]
    assert chain == ["f7b9d1e3a5c7", "e5a7c9d1f3b5", "d4f6b8c0e2a4", "d41b7a90c2f1", "c8e07325e3a6"]


def test_fresh_database_has_track_record_schema(tmp_path, monkeypatch):
    db_file = tmp_path / "fresh_tr.db"
    _upgrade(monkeypatch, db_file)  # the baseline's create_all already includes them
    engine, insp = _inspect(db_file)
    try:
        assert TR_MENTION_COLUMNS <= {c["name"] for c in insp.get_columns("extracted_mentions")}
        assert "track_record" in {c["name"] for c in insp.get_columns("source_configs")}
        assert "track_record_processed" in insp.get_table_names()
    finally:
        engine.dispose()


def test_deployed_database_at_d41b_gains_track_record_schema(tmp_path, monkeypatch):
    """The Hetzner database: built before the track record, stamped d41b7a90c2f1."""
    from src.models import Base

    db_file = tmp_path / "deployed.db"
    engine = sa.create_engine(f"sqlite:///{db_file}")
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for col in sorted(TR_MENTION_COLUMNS):
            conn.execute(sa.text(f"alter table extracted_mentions drop column {col}"))
        conn.execute(sa.text("alter table source_configs drop column track_record"))
        conn.execute(sa.text("drop table track_record_processed"))
        conn.execute(sa.text("create table alembic_version (version_num varchar(32) primary key)"))
        conn.execute(sa.text("insert into alembic_version values ('d41b7a90c2f1')"))
    engine.dispose()

    _upgrade(monkeypatch, db_file)
    _upgrade(monkeypatch, db_file)  # idempotent

    engine, insp = _inspect(db_file)
    try:
        assert TR_MENTION_COLUMNS <= {c["name"] for c in insp.get_columns("extracted_mentions")}
        assert "track_record" in {c["name"] for c in insp.get_columns("source_configs")}
        assert "track_record_processed" in insp.get_table_names()
        with engine.connect() as conn:
            assert conn.execute(sa.text("select version_num from alembic_version")).scalar_one() == _head()
    finally:
        engine.dispose()
