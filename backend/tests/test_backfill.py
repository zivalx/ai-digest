"""
One-shot backfill: window cap, ledger dedupe, batching, safety guards (fakes only).
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from src import backfill
from src.models import Cycle, ExtractedMention, SourceConfig
from src.models.cycle import CycleStatus
from src.services.scheduler_service import PUSH_BATCH_SIZE, SchedulerService
from tests.test_scheduler_ledger import (  # reuse the fakes
    FakeExtraction, FakeOrchestrator, RecordingTrender,
)

NOW = datetime(2026, 11, 25, 15, 30, tzinfo=timezone.utc)


def _ymd(days_ago):
    return (NOW - timedelta(days=days_ago)).strftime("%Y%m%d")


def _video(vid, days_ago):
    return {"video_id": vid, "channel": "@f", "title": vid, "transcript": "nvda " * 20,
            "status": "success", "upload_date": _ymd(days_ago)}


async def _add_config(session_factory, name="Finance YT", source_type="youtube",
                      purpose="finance", enabled=True, spec=None):
    async with session_factory() as db:
        cfg = SourceConfig(id=uuid.uuid4(), name=name, source_type=source_type,
                           credential_ref="X", collect_spec=spec or {"channels": ["@f"]},
                           purpose=purpose, enabled=enabled)
        db.add(cfg)
        await db.commit()
        return cfg


def _scheduler(session_factory, data, trender=None, extraction=None):
    return SchedulerService(
        session_factory=session_factory, orchestrator=FakeOrchestrator(data),
        extraction_service=extraction or FakeExtraction(),
        trender_client=trender or RecordingTrender(),
    )


# ---------------------------------------------------------------- arguments

def test_days_above_the_cap_are_rejected(capsys):
    with pytest.raises(SystemExit) as exc:
        backfill.parse_args(["--days", str(backfill.MAX_BACKFILL_DAYS + 1)])
    assert exc.value.code == 2
    assert str(backfill.MAX_BACKFILL_DAYS) in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["0", "-3", "abc"])
def test_days_must_be_a_positive_integer(bad):
    with pytest.raises(SystemExit):
        backfill.parse_args(["--days", bad])


def test_max_items_is_capped():
    with pytest.raises(SystemExit):
        backfill.parse_args(["--days", "10", "--max-items", str(backfill.MAX_ITEMS_CAP + 1)])


def test_window_starts_at_midnight_utc_n_days_back():
    assert backfill.window_start(NOW, 10) == datetime(2026, 11, 15, 0, 0, tzinfo=timezone.utc)


def test_spec_overrides_per_source_type():
    assert backfill.spec_overrides_for("youtube", 30, 40) == {
        "days_back": 30, "max_videos_per_channel": 40}
    assert backfill.spec_overrides_for("telegram", 30, 200) == {"max_messages": 200}
    assert backfill.spec_overrides_for("reddit", 30, 40) == {}


# ---------------------------------------------------------------- the run

@pytest.mark.asyncio
async def test_backfill_widens_the_collection_and_filters_to_the_window(session_factory):
    await _add_config(session_factory)
    data = [_video("IN1", 3), _video("IN2", 20), _video("OUT", 45)]
    svc = _scheduler(session_factory, data)
    report = await backfill.run_backfill(days=30, scheduler=svc, session_factory=session_factory, now=NOW)

    call = svc.orchestrator.calls[0]
    assert call["days"] == 30
    assert call["spec"]["days_back"] == 30
    assert call["spec"]["max_videos_per_channel"] == backfill.DEFAULT_MAX_ITEMS["youtube"]

    (src,) = report.sources
    assert src.ok and src.stats["out_of_window"] == 1 and src.stats["new"] == 2
    assert {r.citation_json["video_id"] for r in await _all_rows(session_factory)} == {"IN1", "IN2"}

    async with session_factory() as db:
        cycle = (await db.execute(select(Cycle))).scalars().one()
    assert cycle.name == "finance-backfill:Finance YT"
    assert cycle.status == CycleStatus.COMPLETED


async def _all_rows(session_factory):
    async with session_factory() as db:
        return (await db.execute(select(ExtractedMention))).scalars().all()


@pytest.mark.asyncio
async def test_second_run_pushes_nothing_twice(session_factory):
    await _add_config(session_factory)
    data = [_video("A", 2), _video("B", 9)]

    t1 = RecordingTrender()
    await backfill.run_backfill(days=14, scheduler=_scheduler(session_factory, data, trender=t1),
                                session_factory=session_factory, now=NOW)
    assert sum(len(b) for b in t1.batches) == 2

    t2, ex2 = RecordingTrender(), FakeExtraction()
    report = await backfill.run_backfill(
        days=14, scheduler=_scheduler(session_factory, data, trender=t2, extraction=ex2),
        session_factory=session_factory, now=NOW)
    assert t2.batches == [] and ex2.received == []
    assert report.sources[0].stats["already_seen"] == 2
    assert len(await _all_rows(session_factory)) == 2


@pytest.mark.asyncio
async def test_a_longer_second_backfill_picks_up_only_the_older_items(session_factory):
    await _add_config(session_factory)
    data = [_video("NEW", 2), _video("OLD", 25)]
    await backfill.run_backfill(days=7, scheduler=_scheduler(session_factory, data),
                                session_factory=session_factory, now=NOW)
    assert {r.citation_json["video_id"] for r in await _all_rows(session_factory)} == {"NEW"}

    ex = FakeExtraction()
    await backfill.run_backfill(days=30, scheduler=_scheduler(session_factory, data, extraction=ex),
                                session_factory=session_factory, now=NOW)
    assert [i["video_id"] for i in ex.received] == ["OLD"]


@pytest.mark.asyncio
async def test_large_backfill_is_pushed_in_batches(session_factory):
    await _add_config(session_factory)
    n = PUSH_BATCH_SIZE * 2 + 1
    data = [_video(f"V{i:03d}", 1 + i % 20) for i in range(n)]
    trender = RecordingTrender()
    report = await backfill.run_backfill(
        days=30, scheduler=_scheduler(session_factory, data, trender=trender),
        session_factory=session_factory, now=NOW)
    assert [len(b) for b in trender.batches] == [PUSH_BATCH_SIZE, PUSH_BATCH_SIZE, 1]
    assert report.totals["pushed"] == n


@pytest.mark.asyncio
async def test_only_enabled_finance_sources_run_and_source_filter_works(session_factory):
    await _add_config(session_factory, name="Finance YT")
    await _add_config(session_factory, name="Other YT", spec={"channels": ["@o"]})
    await _add_config(session_factory, name="General", purpose="general")
    await _add_config(session_factory, name="Disabled", enabled=False)

    svc = _scheduler(session_factory, [])
    report = await backfill.run_backfill(days=7, scheduler=svc, session_factory=session_factory, now=NOW)
    assert {s.name for s in report.sources} == {"Finance YT", "Other YT"}

    svc2 = _scheduler(session_factory, [])
    report = await backfill.run_backfill(days=7, source="other yt", scheduler=svc2,
                                         session_factory=session_factory, now=NOW)
    assert [s.name for s in report.sources] == ["Other YT"]


@pytest.mark.asyncio
async def test_unknown_source_name_lists_the_available_ones(session_factory):
    await _add_config(session_factory, name="Finance YT")
    with pytest.raises(backfill.BackfillError) as exc:
        await backfill.run_backfill(days=7, source="nope", scheduler=_scheduler(session_factory, []),
                                    session_factory=session_factory, now=NOW)
    assert "Finance YT" in str(exc.value)


@pytest.mark.asyncio
async def test_no_finance_sources_is_an_error(session_factory):
    with pytest.raises(backfill.BackfillError):
        await backfill.run_backfill(days=7, scheduler=_scheduler(session_factory, []),
                                    session_factory=session_factory, now=NOW)


@pytest.mark.asyncio
async def test_refuses_while_a_cycle_is_running_unless_forced(session_factory):
    await _add_config(session_factory)
    async with session_factory() as db:
        db.add(Cycle(name="finance-feeder:Finance YT", status=CycleStatus.SUMMARIZING,
                     config_snapshot={}, started_at=NOW - timedelta(minutes=20)))
        await db.commit()

    svc = _scheduler(session_factory, [_video("A", 1)])
    with pytest.raises(backfill.BackfillError) as exc:
        await backfill.run_backfill(days=7, scheduler=svc, session_factory=session_factory, now=NOW)
    assert "--force" in str(exc.value)
    assert svc.orchestrator.calls == []

    report = await backfill.run_backfill(days=7, force=True, scheduler=svc,
                                         session_factory=session_factory, now=NOW)
    assert report.sources[0].ok


@pytest.mark.asyncio
async def test_an_old_stuck_cycle_does_not_block(session_factory):
    await _add_config(session_factory)
    async with session_factory() as db:
        db.add(Cycle(name="finance-feeder:x", status=CycleStatus.COLLECTING, config_snapshot={},
                     started_at=NOW - timedelta(hours=9)))
        await db.commit()
    report = await backfill.run_backfill(days=7, scheduler=_scheduler(session_factory, []),
                                         session_factory=session_factory, now=NOW)
    assert report.sources[0].ok


@pytest.mark.asyncio
async def test_one_failing_source_does_not_stop_the_others(session_factory):
    await _add_config(session_factory, name="A")
    await _add_config(session_factory, name="B", spec={"channels": ["@b"]})

    class FlakyOrchestrator(FakeOrchestrator):
        async def collect(self, source_type, credential_ref, collect_spec, timeframe_days):
            if collect_spec.get("channels") == ["@f"]:
                raise RuntimeError("yt-dlp blocked")
            return await super().collect(source_type, credential_ref, collect_spec, timeframe_days)

    svc = SchedulerService(session_factory=session_factory, orchestrator=FlakyOrchestrator([]),
                           extraction_service=FakeExtraction(), trender_client=RecordingTrender())
    report = await backfill.run_backfill(days=7, scheduler=svc, session_factory=session_factory, now=NOW)
    by_name = {s.name: s for s in report.sources}
    assert not by_name["A"].ok and "yt-dlp blocked" in by_name["A"].error
    assert by_name["B"].ok
    assert report.exit_code() == 1


@pytest.mark.asyncio
async def test_dry_run_writes_and_pushes_nothing(session_factory):
    await _add_config(session_factory)
    trender, ex = RecordingTrender(), FakeExtraction()
    svc = _scheduler(session_factory, [_video("A", 1), _video("OLD", 50)], trender=trender, extraction=ex)
    report = await backfill.run_backfill(days=30, dry_run=True, scheduler=svc,
                                         session_factory=session_factory, now=NOW)
    assert report.sources[0].stats["new"] == 1 and report.sources[0].stats["out_of_window"] == 1
    assert [i["video_id"] for i in report.sources[0].preview] == ["A"]
    assert trender.batches == [] and ex.received == []
    async with session_factory() as db:
        assert (await db.execute(select(Cycle))).scalars().all() == []
        assert await _all_rows(session_factory) == []


@pytest.mark.asyncio
async def test_report_text_and_exit_codes(session_factory):
    await _add_config(session_factory)
    report = await backfill.run_backfill(days=30, scheduler=_scheduler(session_factory, [_video("A", 1)]),
                                         session_factory=session_factory, now=NOW)
    text = backfill.format_report(report)
    assert "Finance YT" in text and "pushed 1" in text and "30 days" in text
    assert report.exit_code() == 0

    report.sources[0].stats["push_failed"] = 2
    assert report.exit_code() == 3
    assert "run the same command again" in backfill.format_report(report)
