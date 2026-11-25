"""
Ledger, retry and batching behaviour of the finance feeder (fakes only, no network).

Covers what a long-running server needs: items that failed are tried again, a push that
failed is retried later with its original date, big pushes are batched, and the stored
ledger does not keep transcripts.
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from src.models import CollectedData, ExtractedMention, SourceConfig
from src.models.cycle import CycleStatus
from src.services.scheduler_service import PUSH_BATCH_SIZE, SchedulerService
from src.services.trender_client import TrenderPushError


class FakeOrchestrator:
    def __init__(self, data):
        self.data = data
        self.calls = []

    async def collect(self, source_type, credential_ref, collect_spec, timeframe_days):
        self.calls.append({"spec": dict(collect_spec), "days": timeframe_days})
        return {
            "data": list(self.data),
            "item_count": len(self.data),
            "source_info": {"channels": collect_spec.get("channels", [])},
            "metadata": {"collection_time_ms": 1, "source_type": source_type},
        }


class FakeExtraction:
    """One NVDA pick per item; ``fail_ids`` simulates an LLM error for those videos."""

    def __init__(self, fail_ids=()):
        self.fail_ids = set(fail_ids)
        self.received = []
        self.last_failed_indexes = []

    async def extract(self, items):
        self.received = items
        self.last_failed_indexes = []
        out = []
        for i, it in enumerate(items):
            if it.get("video_id") in self.fail_ids:
                self.last_failed_indexes.append(i)
                continue
            vid = it.get("video_id")
            out.append({
                "symbol": "NVDA", "company_name": "NVIDIA", "source": "youtube",
                "stance": "bullish", "sentiment_score": 0.5, "confidence": 0.9,
                "context": "thesis",
                "citation": {"kind": "youtube", "url": f"https://youtu.be/{vid}?t=1",
                             "video_id": vid, "start_seconds": 1},
                "mentioned_at": it.get("ts") or "2026-11-01T00:00:00Z",
            })
        return out


class RecordingTrender:
    def __init__(self):
        self.batches = []

    async def push_mentions(self, mentions):
        self.batches.append(list(mentions))
        return {"accepted": len(mentions), "rejected": [], "candidates_scored": 1}


class DownTrender:
    async def push_mentions(self, mentions):
        raise TrenderPushError("Failed to push mentions after retries: boom")


class PoisonTrender:
    """Answers 422 for any batch containing the poison video, accepts the rest."""

    def __init__(self, poison_video):
        self.poison_video = poison_video
        self.batches = []

    async def push_mentions(self, mentions):
        self.batches.append(list(mentions))
        if any(m["citation"].get("video_id") == self.poison_video for m in mentions):
            raise TrenderPushError("trender returned 422: bad", status_code=422)
        return {"accepted": len(mentions), "rejected": []}


async def _config(session_factory, source_type="youtube", spec=None):
    async with session_factory() as db:
        cfg = SourceConfig(
            id=uuid.uuid4(), name="Finance", source_type=source_type, credential_ref="X",
            collect_spec=spec or {"channels": ["@finance"]}, purpose="finance", enabled=True,
        )
        db.add(cfg)
        await db.commit()
        await db.refresh(cfg)
        return cfg


def _svc(session_factory, data, extraction=None, trender=None):
    return SchedulerService(
        session_factory=session_factory,
        orchestrator=FakeOrchestrator(data),
        extraction_service=extraction or FakeExtraction(),
        trender_client=trender or RecordingTrender(),
    )


def _video(vid, **extra):
    return {"video_id": vid, "channel": "@finance", "title": vid, "transcript": "buy nvda " * 50,
            "transcript_segments": [{"start": 1.0, "text": "buy nvda"}], "description": "long",
            "status": "success", "published_at": None, **extra}


async def _rows(session_factory):
    async with session_factory() as db:
        return (await db.execute(select(ExtractedMention))).scalars().all()


async def _ledger_ids(session_factory):
    async with session_factory() as db:
        rows = (await db.execute(select(CollectedData))).scalars().all()
    return {it["video_id"] for r in rows for it in r.data}


@pytest.mark.asyncio
async def test_collector_failures_are_not_ledgered_and_retried(session_factory):
    cfg = await _config(session_factory)
    failed = {"video_id": "BAD", "status": "failed", "transcript": "", "error": "boom"}
    svc = _svc(session_factory, [_video("OK1"), failed])
    await svc.run_finance_cycle(cfg)

    assert await _ledger_ids(session_factory) == {"OK1"}

    # The next run sees BAD again (this time it works) and processes it.
    extraction = FakeExtraction()
    svc2 = _svc(session_factory, [_video("OK1"), _video("BAD")], extraction=extraction)
    await svc2.run_finance_cycle(cfg)
    assert [it["video_id"] for it in extraction.received] == ["BAD"]


@pytest.mark.asyncio
async def test_llm_failure_keeps_item_out_of_ledger(session_factory):
    cfg = await _config(session_factory)
    svc = _svc(session_factory, [_video("A"), _video("B")], extraction=FakeExtraction(fail_ids={"B"}))
    cycle = await svc.run_finance_cycle(cfg)

    assert cycle.status == CycleStatus.COMPLETED
    assert await _ledger_ids(session_factory) == {"A"}

    extraction = FakeExtraction()
    await _svc(session_factory, [_video("A"), _video("B")], extraction=extraction).run_finance_cycle(cfg)
    assert [it["video_id"] for it in extraction.received] == ["B"]


@pytest.mark.asyncio
async def test_extraction_crash_leaves_nothing_in_the_ledger(session_factory):
    cfg = await _config(session_factory)

    class Boom:
        async def extract(self, items):
            raise RuntimeError("llm down")

    svc = _svc(session_factory, [_video("A")], extraction=Boom())
    with pytest.raises(RuntimeError):
        await svc.run_finance_cycle(cfg)
    assert await _ledger_ids(session_factory) == set()


@pytest.mark.asyncio
async def test_ledger_copy_has_no_transcript(session_factory):
    cfg = await _config(session_factory)
    await _svc(session_factory, [_video("A")]).run_finance_cycle(cfg)
    async with session_factory() as db:
        row = (await db.execute(select(CollectedData))).scalars().one()
    item = row.data[0]
    assert item["video_id"] == "A"
    assert "transcript" not in item and "transcript_segments" not in item and "description" not in item


@pytest.mark.asyncio
async def test_failed_push_is_retried_next_cycle_with_original_date(session_factory):
    cfg = await _config(session_factory)
    published = "2026-11-02T00:00:00Z"
    svc = _svc(session_factory, [_video("A", published_at=published)], trender=DownTrender())
    await svc.run_finance_cycle(cfg)

    rows = await _rows(session_factory)
    assert len(rows) == 1 and rows[0].pushed_to_trender is False
    assert rows[0].mentioned_at is not None

    trender = RecordingTrender()
    # Nothing new this time: the old mention is delivered from the backlog.
    await _svc(session_factory, [_video("A", published_at=published)], trender=trender).run_finance_cycle(cfg)

    rows = await _rows(session_factory)
    assert len(rows) == 1 and rows[0].pushed_to_trender is True
    assert len(trender.batches) == 1
    payload = trender.batches[0][0]
    assert payload["symbol"] == "NVDA"
    assert payload["mentioned_at"] == "2026-11-02T00:00:00Z"
    assert payload["citation"]["video_id"] == "A"
    assert payload["source"] == "youtube"
    assert isinstance(payload["sentiment_score"], float)

    # And it is not sent a third time.
    trender2 = RecordingTrender()
    await _svc(session_factory, [_video("A")], trender=trender2).run_finance_cycle(cfg)
    assert trender2.batches == []


@pytest.mark.asyncio
async def test_old_unpushed_mentions_are_not_retried_forever(session_factory):
    cfg = await _config(session_factory)
    await _svc(session_factory, [_video("A")], trender=DownTrender()).run_finance_cycle(cfg)
    async with session_factory() as db:
        row = (await db.execute(select(ExtractedMention))).scalars().one()
        row.created_at = datetime.now(timezone.utc) - timedelta(days=30)
        await db.commit()

    trender = RecordingTrender()
    await _svc(session_factory, [], trender=trender).run_finance_cycle(cfg)
    assert trender.batches == []


@pytest.mark.asyncio
async def test_pushes_are_batched(session_factory):
    cfg = await _config(session_factory)
    total = PUSH_BATCH_SIZE * 2 + 7
    data = [_video(f"V{i:03d}") for i in range(total)]
    trender = RecordingTrender()
    await _svc(session_factory, data, trender=trender).run_finance_cycle(cfg)

    sizes = [len(b) for b in trender.batches]
    assert sizes == [PUSH_BATCH_SIZE, PUSH_BATCH_SIZE, 7]
    assert all(r.pushed_to_trender for r in await _rows(session_factory))


@pytest.mark.asyncio
async def test_transport_failure_stops_after_first_batch(session_factory):
    cfg = await _config(session_factory)
    data = [_video(f"V{i:03d}") for i in range(PUSH_BATCH_SIZE + 5)]

    calls = []

    class CountingDown:
        async def push_mentions(self, mentions):
            calls.append(len(mentions))
            raise TrenderPushError("unreachable")

    cycle, stats = await _svc(session_factory, data, trender=CountingDown()).run_cycle_with_stats(cfg)
    assert cycle.status == CycleStatus.COMPLETED
    assert calls == [PUSH_BATCH_SIZE]  # no hammering of a dead trender
    assert stats["push_failed"] == PUSH_BATCH_SIZE + 5


@pytest.mark.asyncio
async def test_http_error_batch_is_split_so_one_bad_mention_does_not_block(session_factory):
    cfg = await _config(session_factory)
    data = [_video("A"), _video("POISON"), _video("C")]
    trender = PoisonTrender("POISON")
    cycle, stats = await _svc(session_factory, data, trender=trender).run_cycle_with_stats(cfg)

    assert cycle.status == CycleStatus.COMPLETED
    by_video = {r.citation_json["video_id"]: r for r in await _rows(session_factory)}
    assert by_video["A"].pushed_to_trender and by_video["C"].pushed_to_trender
    assert not by_video["POISON"].pushed_to_trender
    assert "422" in by_video["POISON"].trender_response
    assert stats["pushed"] == 2 and stats["push_failed"] == 1


@pytest.mark.asyncio
async def test_cutoff_drops_old_and_undated_items_and_keeps_them_out_of_the_ledger(session_factory):
    cfg = await _config(session_factory)
    now = datetime.now(timezone.utc)
    recent = (now - timedelta(days=3)).strftime("%Y%m%d")
    old = (now - timedelta(days=40)).strftime("%Y%m%d")
    data = [
        _video("RECENT", published_at=None, upload_date=recent),
        _video("OLD", published_at=None, upload_date=old),
        _video("NODATE", published_at=None),
    ]
    extraction = FakeExtraction()
    svc = _svc(session_factory, data, extraction=extraction)
    cutoff = (now - timedelta(days=10)).replace(hour=0, minute=0, second=0, microsecond=0)
    _, stats = await svc.run_cycle_with_stats(cfg, cutoff=cutoff, cycle_prefix="finance-backfill")

    assert [it["video_id"] for it in extraction.received] == ["RECENT"]
    assert stats["out_of_window"] == 2 and stats["collected"] == 3
    # OLD must stay un-ledgered so a later, longer backfill still processes it.
    assert await _ledger_ids(session_factory) == {"RECENT"}


@pytest.mark.asyncio
async def test_spec_overrides_win_over_the_stored_collect_spec(session_factory):
    cfg = await _config(session_factory, spec={"channels": ["@a"], "days_back": 7, "max_videos": 5})
    svc = _svc(session_factory, [])
    await svc.run_cycle_with_stats(
        cfg, timeframe_days=30, spec_overrides={"days_back": 30, "max_videos_per_channel": 40})

    call = svc.orchestrator.calls[0]
    assert call["days"] == 30
    assert call["spec"]["days_back"] == 30
    assert call["spec"]["max_videos_per_channel"] == 40
    assert call["spec"]["channels"] == ["@a"]
    # The stored config itself is untouched.
    assert cfg.collect_spec["days_back"] == 7


@pytest.mark.asyncio
async def test_cycle_prefix_separates_backfill_from_feeder(session_factory):
    cfg = await _config(session_factory)
    cycle, _ = await _svc(session_factory, []).run_cycle_with_stats(cfg, cycle_prefix="finance-backfill")
    assert cycle.name == "finance-backfill:Finance"


def test_telegram_item_ids_are_per_channel():
    a = SchedulerService._item_id({"message_id": 5, "channel": "@a"}, "telegram")
    b = SchedulerService._item_id({"message_id": 5, "channel": "@b"}, "telegram")
    assert a != b


@pytest.mark.asyncio
async def test_failed_cycle_is_recorded_as_failed_not_left_running(session_factory):
    """A collector error must end the cycle as FAILED (it used to raise from the handler)."""
    from src.models import Cycle

    cfg = await _config(session_factory)

    class Broken:
        async def collect(self, **kw):
            raise RuntimeError("collector blew up")

    svc = SchedulerService(session_factory=session_factory, orchestrator=Broken(),
                           extraction_service=FakeExtraction(), trender_client=RecordingTrender())
    with pytest.raises(RuntimeError, match="collector blew up"):
        await svc.run_finance_cycle(cfg)

    async with session_factory() as db:
        cycle = (await db.execute(select(Cycle))).scalars().one()
    assert cycle.status == CycleStatus.FAILED
    assert "collector blew up" in cycle.error_message
    assert cycle.completed_at is not None


@pytest.mark.asyncio
async def test_auth_error_stops_the_loop_instead_of_splitting(session_factory):
    """A 401 (token mismatch) is a config problem: one request, no per-mention hammering."""
    cfg = await _config(session_factory)
    calls = []

    class Unauthorized:
        async def push_mentions(self, mentions):
            calls.append(len(mentions))
            raise TrenderPushError("trender returned 401: nope", status_code=401)

    data = [_video(f"V{i}") for i in range(5)]
    await _svc(session_factory, data, trender=Unauthorized()).run_finance_cycle(cfg)
    assert calls == [5]
    assert all(not r.pushed_to_trender for r in await _rows(session_factory))

    # ...and once the token is right the whole backlog goes out.
    trender = RecordingTrender()
    await _svc(session_factory, [], trender=trender).run_finance_cycle(cfg)
    assert sum(len(b) for b in trender.batches) == 5


@pytest.mark.asyncio
async def test_mentions_trender_refused_as_invalid_are_not_retried(session_factory):
    cfg = await _config(session_factory)
    await _svc(session_factory, [_video("POISON")], trender=PoisonTrender("POISON")).run_finance_cycle(cfg)

    trender = PoisonTrender("POISON")
    await _svc(session_factory, [], trender=trender).run_finance_cycle(cfg)
    assert trender.batches == []
