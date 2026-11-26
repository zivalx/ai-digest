"""
/health/deep: the shape the zivalx-ops watchdog reads, and the feeder-age logic.

    {"status": "ok" | "degraded" | "down", "checks": {name: {"ok": bool, "detail": str}}}
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from src.config import FeederSettings
from src.health import deep_health, status_code_for
from src.models import Cycle, ExtractedMention, SourceConfig
from src.models.cycle import CycleStatus

NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
INTERVAL = 6 * 3600
LIMIT = 2 * INTERVAL + 1800  # a missed tick is fine, two are not


def _settings(enabled=True, interval=INTERVAL):
    return FeederSettings(
        trender_base_url="http://trender-app:8000", trender_ingest_token=None,
        anthropic_model="m", xai_model="x", deepseek_model="d", llm_providers="anthropic",
        feeder_interval_seconds=interval, feeder_enabled=enabled,
    )


async def _source(session_factory, enabled=True, purpose="finance"):
    async with session_factory() as db:
        db.add(SourceConfig(id=uuid.uuid4(), name="F", source_type="youtube", credential_ref="X",
                            collect_spec={}, purpose=purpose, enabled=enabled))
        await db.commit()


async def _cycle(session_factory, age, status=CycleStatus.COMPLETED, stats=None, name="finance-feeder:F"):
    done = NOW - age
    async with session_factory() as db:
        db.add(Cycle(name=name, status=status, started_at=done - timedelta(minutes=5),
                     completed_at=done, config_snapshot={"stats": stats} if stats else {}))
        await db.commit()


async def _mention(session_factory, age, response=None, pushed=False):
    async with session_factory() as db:
        cycle = Cycle(name="finance-feeder:F", status=CycleStatus.COMPLETED, config_snapshot={})
        db.add(cycle)
        await db.commit()
        db.add(ExtractedMention(
            cycle_id=cycle.id, symbol="NVDA", citation_json={}, dedupe_key=str(uuid.uuid4()),
            pushed_to_trender=pushed, trender_response=response, created_at=NOW - age))
        await db.commit()


async def _health(session_factory, **kw):
    return await deep_health(
        session_factory=session_factory, settings=kw.pop("settings", _settings()),
        now=NOW, started_at=kw.pop("started_at", NOW - timedelta(days=3)))


def _failing(result):
    return sorted(n for n, c in result["checks"].items() if not c["ok"])


@pytest.mark.asyncio
async def test_shape_is_what_the_watchdog_reads(session_factory):
    await _source(session_factory)
    await _cycle(session_factory, timedelta(hours=1))
    result = await _health(session_factory)
    assert result["status"] == "ok"
    assert set(result["checks"]) == {"database", "feeder", "extraction", "push"}
    for check in result["checks"].values():
        assert isinstance(check["ok"], bool) and isinstance(check["detail"], str)
    assert status_code_for(result) == 200


@pytest.mark.asyncio
async def test_recent_cycle_is_ok_and_one_missed_tick_is_still_ok(session_factory):
    await _source(session_factory)
    await _cycle(session_factory, timedelta(seconds=LIMIT - 60))
    assert (await _health(session_factory))["status"] == "ok"


@pytest.mark.asyncio
async def test_two_missed_ticks_is_degraded_not_down(session_factory):
    await _source(session_factory)
    await _cycle(session_factory, timedelta(seconds=LIMIT + 60))
    result = await _health(session_factory)
    assert result["status"] == "degraded"
    assert _failing(result) == ["feeder"]
    assert status_code_for(result) == 200


@pytest.mark.asyncio
async def test_failed_cycles_do_not_count_as_fresh(session_factory):
    await _source(session_factory)
    await _cycle(session_factory, timedelta(days=2))
    await _cycle(session_factory, timedelta(minutes=10), status=CycleStatus.FAILED)
    assert _failing(await _health(session_factory)) == ["feeder"]


@pytest.mark.asyncio
async def test_backfill_cycles_do_not_count_as_feeder_cycles(session_factory):
    await _source(session_factory)
    await _cycle(session_factory, timedelta(minutes=10), name="finance-backfill:F")
    assert _failing(await _health(session_factory)) == ["feeder"]


@pytest.mark.asyncio
async def test_no_cycle_yet_is_ok_right_after_start_and_degraded_later(session_factory):
    await _source(session_factory)
    fresh = await _health(session_factory, started_at=NOW - timedelta(minutes=20))
    assert fresh["status"] == "ok" and "starting" in fresh["checks"]["feeder"]["detail"]
    old = await _health(session_factory, started_at=NOW - timedelta(seconds=LIMIT + 60))
    assert _failing(old) == ["feeder"]


@pytest.mark.asyncio
async def test_old_cycle_but_just_restarted_gets_a_grace_period(session_factory):
    await _source(session_factory)
    await _cycle(session_factory, timedelta(days=4))
    result = await _health(session_factory, started_at=NOW - timedelta(minutes=10))
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_feeder_disabled_or_no_sources_is_not_a_fault(session_factory):
    off = await _health(session_factory, settings=_settings(enabled=False))
    assert off["status"] == "ok" and "disabled" in off["checks"]["feeder"]["detail"]

    none = await _health(session_factory)  # enabled, but no finance source configured
    assert none["status"] == "ok" and "no finance" in none["checks"]["feeder"]["detail"]


@pytest.mark.asyncio
async def test_disabled_and_general_sources_do_not_count(session_factory):
    await _source(session_factory, enabled=False)
    await _source(session_factory, purpose="general")
    assert (await _health(session_factory))["status"] == "ok"


@pytest.mark.asyncio
async def test_limit_follows_the_configured_interval(session_factory):
    await _source(session_factory)
    await _cycle(session_factory, timedelta(hours=3))
    fast = await _health(session_factory, settings=_settings(interval=600))  # limit 1800 s
    assert _failing(fast) == ["feeder"]


@pytest.mark.asyncio
async def test_extraction_check_flags_a_cycle_where_every_item_failed(session_factory):
    await _source(session_factory)
    await _cycle(session_factory, timedelta(hours=1), stats={"new": 4, "extraction_failed": 4})
    assert _failing(await _health(session_factory)) == ["extraction"]

    await _cycle(session_factory, timedelta(minutes=5), stats={"new": 4, "extraction_failed": 1})
    assert (await _health(session_factory))["status"] == "ok"  # the latest cycle is what counts


@pytest.mark.asyncio
async def test_push_check_flags_old_unpushed_mentions(session_factory):
    await _source(session_factory)
    await _cycle(session_factory, timedelta(hours=1))
    await _mention(session_factory, timedelta(seconds=LIMIT + 60), response="error: boom")
    assert _failing(await _health(session_factory)) == ["push"]


@pytest.mark.asyncio
async def test_push_check_ignores_recent_pushed_refused_and_given_up_rows(session_factory):
    await _source(session_factory)
    await _cycle(session_factory, timedelta(hours=1))
    await _mention(session_factory, timedelta(minutes=30), response="error: x")           # recent
    await _mention(session_factory, timedelta(days=2), pushed=True, response="{'ok': 1}")  # delivered
    await _mention(session_factory, timedelta(days=2), response="error: trender returned 422: bad")
    await _mention(session_factory, timedelta(days=30), response="error: x")              # given up
    assert (await _health(session_factory))["status"] == "ok"


@pytest.mark.asyncio
async def test_database_down_is_down_with_503():
    class Broken:
        def __call__(self):
            return self

        async def __aenter__(self):
            raise ConnectionRefusedError("db gone")

        async def __aexit__(self, *a):
            return False

    result = await deep_health(session_factory=Broken(), settings=_settings(), now=NOW, started_at=NOW)
    assert result["status"] == "down"
    assert _failing(result) == ["database"]
    assert status_code_for(result) == 503
    assert "db gone" not in str(result)  # no error text leaks to the unauthenticated caller
