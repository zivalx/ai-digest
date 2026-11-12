"""
BackfillService tests against fakes (no network, no LLM).
"""
import uuid
from datetime import date, timedelta

import pytest
from sqlalchemy import delete, select

from src.models import CollectedData, Cycle, ExtractedMention, SourceConfig, TrackRecordProcessed
from src.models.cycle import CycleStatus
from src.services.backfill_service import BackfillService, estimate_cost
from src.services.extraction_service import EXTRACTION_VERSION


class FakeOrchestrator:
    def __init__(self, data):
        self._data = data
        self.specs = []

    async def collect(self, source_type, credential_ref, collect_spec, timeframe_days):
        self.specs.append((dict(collect_spec), timeframe_days))
        return {"data": self._data, "item_count": len(self._data), "source_info": {}}


class FakeExtraction:
    model = "fake-model"

    def __init__(self, fail_ids=(), raise_all=False, symbols=None):
        self.received = []
        self.fail_ids = set(fail_ids)
        self.raise_all = raise_all
        self.symbols = symbols or {}  # video_id -> symbol (default NVDA)

    async def extract(self, items, on_error=None):
        if self.raise_all:
            raise RuntimeError("llm down")
        self.received.extend(items)
        out = []
        for it in items:
            vid = it["video_id"]
            if vid in self.fail_ids:
                on_error(it, RuntimeError("boom"))
                continue
            out.append({
                "symbol": self.symbols.get(vid, "NVDA"), "company_name": "NVIDIA", "source": "youtube",
                "stance": "bullish", "sentiment_score": 0.7, "confidence": 0.8,
                "context": "thesis",
                "citation": {
                    "kind": "youtube", "url": f"https://youtu.be/{vid}?t=10",
                    "video_id": vid, "channel": it.get("channel"),
                    "title": it.get("title"), "start_seconds": 10,
                },
                "is_recommendation": True, "action": "buy", "horizon": "short",
                "thesis": "thesis", "quote": "buy it",
                "said_at": "2026-01-05T10:00:00Z", "said_at_precision": "datetime",
            })
        return out


class FakeTrender:
    """Mimics trender's ingest response: {accepted, duplicates, rejected:[{symbol, reason}]}."""

    def __init__(self, calls_error=None, reject=None, duplicates=False):
        self.calls_error = calls_error
        self.reject = reject or {}  # symbol -> reason
        self.duplicates = duplicates
        self.pushed_mentions = []
        self.pushed_calls = []

    async def push_mentions(self, mentions):
        self.pushed_mentions.append(mentions)
        return {"accepted": len(mentions)}

    async def push_calls(self, calls):
        if self.calls_error:
            raise self.calls_error
        self.pushed_calls.append(calls)
        rejected = [{"symbol": c["symbol"], "reason": self.reject[c["symbol"]]}
                    for c in calls if c["symbol"] in self.reject]
        ok = len(calls) - len(rejected)
        return {"accepted": 0 if self.duplicates else ok,
                "duplicates": ok if self.duplicates else 0, "rejected": rejected}


def _days_ago(n):
    return f"{(date.today() - timedelta(days=n)).isoformat()}T12:00:00Z"


DATA = [
    {"video_id": "AAA", "channel": "@fin", "title": "One", "transcript": "a" * 400,
     "published_at": _days_ago(10), "status": "success"},
    {"video_id": "BBB", "channel": "@fin", "title": "Two", "transcript": "b" * 400,
     "published_at": _days_ago(20), "status": "success"},
]


async def _config(session_factory, source_type="youtube"):
    async with session_factory() as db:
        db.add(SourceConfig(
            id=uuid.uuid4(), name="Fin YT", source_type=source_type,
            credential_ref="YT", collect_spec={"channels": ["@fin"], "days_back": 7,
                                               "max_videos_per_channel": 3},
            purpose="finance", track_record=True, enabled=True,
        ))
        await db.commit()


def _svc(session_factory, data=DATA, ext=None, trender=None):
    orch, ext, trender = FakeOrchestrator(data), ext or FakeExtraction(), trender or FakeTrender()
    return BackfillService(orch, ext, trender, session_factory), orch, ext, trender


def _run(svc, dry_run=False, confirm=lambda m: True, since=None, limit=50):
    return svc.run("Fin YT", since or date.today() - timedelta(days=100), limit, dry_run, confirm)


async def _ledger(session_factory):
    async with session_factory() as db:
        return (await db.execute(select(TrackRecordProcessed))).scalars().all()


def test_estimate_cost():
    tokens, cost = estimate_cost([{"transcript": "a" * 400}, {"transcript": "b" * 600}, {"transcript": None}], 3.0)
    assert tokens == 250
    assert cost == pytest.approx(250 / 1_000_000 * 3.0)


@pytest.mark.asyncio
async def test_dry_run_makes_no_llm_calls_and_no_pushes(session_factory):
    await _config(session_factory)
    svc, _, ext, trender = _svc(session_factory)
    asked = []
    report = await _run(svc, dry_run=True, confirm=lambda m: asked.append(m) or True)
    assert report.dry_run and report.items_found == 2 and report.items_new == 2
    assert report.est_input_tokens == 200 and report.calls_pushed == 0
    assert ext.received == [] and trender.pushed_calls == [] and trender.pushed_mentions == []
    assert asked == [] and await _ledger(session_factory) == []


@pytest.mark.asyncio
async def test_run_extracts_pushes_with_backfill_origin_and_writes_ledger(session_factory):
    await _config(session_factory)
    svc, _, ext, trender = _svc(session_factory)
    report = await _run(svc)
    assert report.calls_pushed == 2 and not report.dry_run
    assert len(trender.pushed_mentions[0]) == 2
    assert {c["origin"] for c in trender.pushed_calls[0]} == {"backfill"}
    ledger = await _ledger(session_factory)
    assert {(r.item_id, r.extraction_version, r.source_type) for r in ledger} == {
        ("AAA", EXTRACTION_VERSION, "youtube"), ("BBB", EXTRACTION_VERSION, "youtube")}
    async with session_factory() as db:
        rows = (await db.execute(select(ExtractedMention))).scalars().all()
    assert len(rows) == 2 and all(r.pushed_calls_to_trender for r in rows)


@pytest.mark.asyncio
async def test_ledger_skips_processed_items_for_same_version(session_factory):
    await _config(session_factory)
    svc, _, ext, trender = _svc(session_factory)
    await _run(svc)
    ext.received.clear()
    report = await _run(svc)
    assert report.items_found == 2 and report.items_new == 0
    assert ext.received == [] and report.calls_pushed == 0


@pytest.mark.asyncio
async def test_new_version_reprocesses(session_factory):
    await _config(session_factory)
    async with session_factory() as db:
        for vid in ("AAA", "BBB"):
            db.add(TrackRecordProcessed(source_type="youtube", item_id=vid, extraction_version="tr-0"))
        await db.commit()
    svc, _, ext, _ = _svc(session_factory)
    report = await _run(svc)
    assert report.items_new == 2 and len(ext.received) == 2


@pytest.mark.asyncio
async def test_collect_spec_overrides_days_back(session_factory):
    await _config(session_factory)
    svc, orch, _, _ = _svc(session_factory)
    await _run(svc, dry_run=True, since=date.today() - timedelta(days=100), limit=42)
    spec, timeframe = orch.specs[0]
    assert spec["days_back"] == 100 and spec["max_videos_per_channel"] == 42
    assert timeframe == 100 and spec["channels"] == ["@fin"]
    async with session_factory() as db:  # stored spec untouched
        cfg = (await db.execute(select(SourceConfig))).scalar_one()
    assert cfg.collect_spec["days_back"] == 7


@pytest.mark.asyncio
async def test_declined_confirmation_aborts(session_factory):
    await _config(session_factory)
    svc, _, ext, trender = _svc(session_factory)
    report = await _run(svc, confirm=lambda m: False)
    assert report.calls_pushed == 0 and ext.received == []
    assert trender.pushed_calls == [] and await _ledger(session_factory) == []


@pytest.mark.asyncio
async def test_dry_run_lists_items(session_factory):
    await _config(session_factory)
    svc, *_ = _svc(session_factory)
    report = await _run(svc, dry_run=True)
    assert [i["id"] for i in report.items] == ["AAA", "BBB"]
    assert report.items[0]["transcript_chars"] == 400 and report.items[0]["title"] == "One"


@pytest.mark.asyncio
async def test_failed_extraction_is_not_ledgered_and_retried(session_factory):
    await _config(session_factory)
    svc, _, ext, _ = _svc(session_factory, ext=FakeExtraction(fail_ids={"BBB"}))
    await _run(svc)
    assert [r.item_id for r in await _ledger(session_factory)] == ["AAA"]
    ext.fail_ids.clear()
    ext.received.clear()
    report = await _run(svc)
    assert report.items_new == 1 and [e["video_id"] for e in ext.received] == ["BBB"]
    assert {r.item_id for r in await _ledger(session_factory)} == {"AAA", "BBB"}


@pytest.mark.asyncio
async def test_push_failure_is_reported_not_blocking(session_factory):
    await _config(session_factory)
    svc, *_ = _svc(session_factory, trender=FakeTrender(calls_error=RuntimeError("down")))
    report = await _run(svc)
    assert report.calls_failed == 2 and report.calls_pushed == 0 and report.mentions_failed == 0
    assert len(await _ledger(session_factory)) == 2
    async with session_factory() as db:  # payload kept for the retry sweep
        rows = (await db.execute(select(ExtractedMention))).scalars().all()
    assert all(r.call_payload_json and not r.pushed_calls_to_trender for r in rows)


@pytest.mark.asyncio
async def test_extraction_exception_marks_cycle_failed(session_factory):
    await _config(session_factory)
    svc, *_ = _svc(session_factory, ext=FakeExtraction(raise_all=True))
    with pytest.raises(RuntimeError):
        await _run(svc)
    async with session_factory() as db:
        cycle = (await db.execute(select(Cycle))).scalar_one()
    assert cycle.status.value == "failed" and "llm down" in cycle.error_message
    assert await _ledger(session_factory) == []


@pytest.mark.asyncio
async def test_duplicate_item_ids_are_deduped(session_factory):
    await _config(session_factory)
    svc, _, ext, _ = _svc(session_factory, data=DATA + [DATA[0]])
    report = await _run(svc)
    assert report.items_found == 3 and report.items_new == 2 and len(ext.received) == 2
    assert len(await _ledger(session_factory)) == 2


async def _mentions(session_factory):
    async with session_factory() as db:
        return (await db.execute(select(ExtractedMention))).scalars().all()


@pytest.mark.asyncio
async def test_rerun_after_version_bump_pushes_calls_for_existing_mentions(session_factory):
    """C1 (probe_backfill.py): run 1's call push fails; after a version bump the
    re-run collides on every dedupe_key and must still push the calls."""
    await _config(session_factory)
    svc1, *_ = _svc(session_factory, trender=FakeTrender(calls_error=RuntimeError("down")))
    r1 = await _run(svc1)
    assert r1.calls_failed == 2
    async with session_factory() as db:  # simulate EXTRACTION_VERSION bump
        await db.execute(delete(TrackRecordProcessed))
        await db.commit()

    svc2, _, ext2, trender2 = _svc(session_factory)
    r2 = await _run(svc2)
    assert r2.items_new == 2 and len(ext2.received) == 2
    assert sum(len(b) for b in trender2.pushed_calls) == 2
    assert r2.calls_pushed == 2 and r2.calls_accepted == 2 and r2.calls_failed == 0
    rows = await _mentions(session_factory)
    assert len(rows) == 2 and all(r.pushed_calls_to_trender for r in rows)


@pytest.mark.asyncio
async def test_backfill_over_feeder_seen_items_attaches_and_pushes_calls(session_factory):
    """Pre-existing mention rows (e.g. from a non-track-record feeder run) get the
    call payload attached without overwriting discovery fields, and are pushed."""
    await _config(session_factory)
    async with session_factory() as db:
        cycle = Cycle(name="finance-feeder:other", status=CycleStatus.COMPLETED, config_snapshot={})
        db.add(cycle)
        await db.commit()
        for vid in ("AAA", "BBB"):
            db.add(ExtractedMention(
                cycle_id=cycle.id, symbol="NVDA", thesis="feeder thesis", stance="bullish",
                citation_json={"video_id": vid}, dedupe_key=f"NVDA|https://youtu.be/{vid}?t=10|10",
                pushed_to_trender=True, pushed_calls_to_trender=False,
            ))
            db.add(TrackRecordProcessed(source_type="youtube", item_id=vid, extraction_version="tr-0"))
        await db.commit()

    svc, _, _, trender = _svc(session_factory)
    report = await _run(svc)
    assert report.calls_pushed == 2
    assert {c["source_item_id"] for b in trender.pushed_calls for c in b} == {"AAA", "BBB"}
    assert trender.pushed_mentions == []  # discovery mentions were already pushed
    rows = await _mentions(session_factory)
    assert len(rows) == 2
    assert all(r.pushed_calls_to_trender and r.call_payload_json["origin"] == "backfill" for r in rows)
    assert all(r.thesis == "feeder thesis" for r in rows)  # discovery fields untouched


@pytest.mark.asyncio
async def test_repush_resends_failed_calls_and_marks_pushed(session_factory):
    await _config(session_factory)
    svc1, *_ = _svc(session_factory, trender=FakeTrender(calls_error=RuntimeError("down")))
    await _run(svc1)

    svc2, orch2, ext2, trender2 = _svc(session_factory)
    result = await svc2.repush()
    assert orch2.specs == [] and ext2.received == []  # no collection, no LLM
    assert result.accepted == 2 and result.failed == 0 and result.rejected == {}
    assert sum(len(b) for b in trender2.pushed_calls) == 2
    assert all(r.pushed_calls_to_trender for r in await _mentions(session_factory))
    assert (await svc2.repush()).sent == 0  # nothing left


@pytest.mark.asyncio
async def test_rejected_calls_stay_unpushed_and_are_counted_by_reason(session_factory):
    await _config(session_factory)
    trender = FakeTrender(reject={"NVDA": "unmapped channel"})
    svc, *_ = _svc(session_factory, trender=trender)
    report = await _run(svc)
    assert report.calls_rejected == {"unmapped channel": 2}
    assert report.calls_pushed == 0 and report.calls_accepted == 0 and report.calls_failed == 0
    # Same symbol twice -> separate requests, so rejections map to exact rows.
    assert [len(b) for b in trender.pushed_calls] == [1, 1]
    rows = await _mentions(session_factory)
    assert all(not r.pushed_calls_to_trender for r in rows)
    assert all("unmapped channel" in r.trender_calls_response for r in rows)
    assert len(await _ledger(session_factory)) == 2  # extraction done; call retryable via sweep


@pytest.mark.asyncio
async def test_partial_rejection_marks_only_accepted_rows(session_factory):
    await _config(session_factory)
    trender = FakeTrender(reject={"ZZZZ": "unknown symbol"})
    svc, *_ = _svc(session_factory, ext=FakeExtraction(symbols={"BBB": "ZZZZ"}), trender=trender)
    report = await _run(svc)
    assert report.calls_accepted == 1 and report.calls_rejected == {"unknown symbol": 1}
    pushed = {r.symbol: r.pushed_calls_to_trender for r in await _mentions(session_factory)}
    assert pushed == {"NVDA": True, "ZZZZ": False}


@pytest.mark.asyncio
async def test_duplicates_count_as_pushed(session_factory):
    await _config(session_factory)
    svc, *_ = _svc(session_factory, trender=FakeTrender(duplicates=True))
    report = await _run(svc)
    assert report.calls_duplicate == 2 and report.calls_pushed == 2
    assert all(r.pushed_calls_to_trender for r in await _mentions(session_factory))


@pytest.mark.asyncio
async def test_failed_or_empty_items_are_unavailable_and_never_ledgered(session_factory):
    await _config(session_factory)
    data = DATA + [
        {"video_id": "FFF", "title": "Rate limited", "status": "failed", "error": "429 Too Many",
         "transcript": "", "published_at": None},
        {"video_id": "EEE", "title": "No captions", "status": "success", "transcript": "",
         "published_at": _days_ago(5)},
    ]
    svc, _, ext, _ = _svc(session_factory, data=data)
    report = await _run(svc)
    assert report.items_found == 4 and report.items_new == 2
    assert [e["video_id"] for e in ext.received] == ["AAA", "BBB"]
    assert {(u["id"], u["title"]) for u in report.unavailable} == {
        ("FFF", "Rate limited"), ("EEE", "No captions")}
    reasons = {u["id"]: u["reason"] for u in report.unavailable}
    assert "429" in reasons["FFF"] and reasons["EEE"] == "no transcript"
    assert {r.item_id for r in await _ledger(session_factory)} == {"AAA", "BBB"}


@pytest.mark.asyncio
async def test_since_filters_by_publish_time(session_factory):
    await _config(session_factory)
    data = [
        {**DATA[0], "published_at": _days_ago(10)},
        {**DATA[1], "published_at": None, "upload_date": (date.today() - timedelta(days=50)).strftime("%Y%m%d")},
        {"video_id": "OLD", "title": "Old", "transcript": "x" * 40, "status": "success",
         "published_at": _days_ago(400)},
        {"video_id": "NOTS", "title": "No time", "transcript": "x" * 40, "status": "success"},
    ]
    svc, _, ext, _ = _svc(session_factory, data=data)
    report = await _run(svc, since=date.today() - timedelta(days=60), dry_run=True)
    assert [i["id"] for i in report.items] == ["AAA", "BBB"]
    assert report.unavailable == [{"id": "NOTS", "title": "No time", "reason": "no publish time"}]


# ---------------------------------------------------------------- persistence
async def _collected(session_factory):
    async with session_factory() as db:
        rows = (await db.execute(select(CollectedData))).scalars().all()
        cycles = (await db.execute(select(Cycle))).scalars().all()
        return rows, cycles


@pytest.mark.asyncio
async def test_real_run_stores_collected_data(session_factory):
    await _config(session_factory)
    svc, *_ = _svc(session_factory)
    await _run(svc)
    rows, cycles = await _collected(session_factory)
    assert len(rows) == 1 and rows[0].item_count == 2
    assert [i["video_id"] for i in rows[0].data] == ["AAA", "BBB"]
    assert rows[0].data[0]["transcript"] == "a" * 400
    assert len(cycles) == 1 and cycles[0].name == "backfill:Fin YT"
    assert cycles[0].status == CycleStatus.COMPLETED and rows[0].cycle_id == cycles[0].id


@pytest.mark.asyncio
async def test_dry_run_stores_collected_data_without_side_effects(session_factory):
    await _config(session_factory)
    svc, _, ext, trender = _svc(session_factory)
    await _run(svc, dry_run=True)
    rows, cycles = await _collected(session_factory)
    assert len(rows) == 1 and rows[0].item_count == 2
    assert cycles[0].status == CycleStatus.COMPLETED
    assert ext.received == [] and trender.pushed_calls == [] and await _ledger(session_factory) == []


@pytest.mark.asyncio
async def test_declined_run_stores_collected_data(session_factory):
    await _config(session_factory)
    svc, _, ext, trender = _svc(session_factory)
    await _run(svc, confirm=lambda m: False)
    rows, cycles = await _collected(session_factory)
    assert len(rows) == 1 and cycles[0].status == CycleStatus.COMPLETED
    assert ext.received == [] and trender.pushed_calls == []


@pytest.mark.asyncio
async def test_collect_failure_marks_cycle_failed(session_factory):
    await _config(session_factory)
    svc, orch, *_ = _svc(session_factory)

    async def boom(**kw):
        raise RuntimeError("yt down")
    orch.collect = boom
    with pytest.raises(RuntimeError):
        await _run(svc)
    rows, cycles = await _collected(session_factory)
    assert rows == [] and cycles[0].status == CycleStatus.FAILED


@pytest.mark.asyncio
async def test_reuse_collected_skips_orchestrator(session_factory):
    await _config(session_factory)
    svc1, *_ = _svc(session_factory)
    await _run(svc1, dry_run=True)

    svc2, orch2, ext2, _ = _svc(session_factory, data=[])  # orchestrator would return nothing
    report = await svc2.run("Fin YT", date.today() - timedelta(days=100), 50, False,
                            lambda m: True, reuse_collected=True)
    assert orch2.specs == []
    assert report.items_found == 2 and report.calls_pushed >= 1
    assert sorted(e["video_id"] for e in ext2.received) == ["AAA", "BBB"]
    rows, _ = await _collected(session_factory)
    assert len(rows) == 1  # reuse does not store a duplicate row


@pytest.mark.asyncio
async def test_reuse_collected_without_stored_data_errors(session_factory):
    await _config(session_factory)
    svc, orch, *_ = _svc(session_factory)
    with pytest.raises(ValueError, match="No stored collected data"):
        await svc.run("Fin YT", date.today() - timedelta(days=100), 50, False,
                      lambda m: True, reuse_collected=True)
    assert orch.specs == []
    _, cycles = await _collected(session_factory)
    assert cycles[0].status == CycleStatus.FAILED


@pytest.mark.asyncio
async def test_backfill_rows_do_not_count_as_seen_for_feeder(session_factory):
    from src.services.scheduler_service import SchedulerService
    await _config(session_factory)
    svc, *_ = _svc(session_factory)
    await _run(svc, dry_run=True)
    feeder = svc._feeder
    async with session_factory() as db:
        assert await feeder._seen_ids(db, "youtube") == set()


# ------------------------------------------------- incremental / throttling
def _item(vid, status="success", transcript="t" * 400, days=10):
    return {"video_id": vid, "channel": "@fin", "title": vid, "transcript": transcript,
            "published_at": _days_ago(days), "status": status}


@pytest.mark.asyncio
async def test_first_run_passes_no_skip_ids_and_throttle_defaults(session_factory):
    await _config(session_factory)
    svc, orch, *_ = _svc(session_factory)
    await _run(svc, dry_run=True)
    spec = orch.specs[0][0]
    assert "skip_video_ids" not in spec
    assert spec["max_concurrency"] == 1 and spec["delay_seconds"] == 6


@pytest.mark.asyncio
async def test_collect_spec_throttle_values_beat_defaults_and_overrides_beat_spec(session_factory):
    async with session_factory() as db:
        db.add(SourceConfig(
            id=uuid.uuid4(), name="Fin YT", source_type="youtube", credential_ref="YT",
            collect_spec={"channels": ["@fin"], "max_concurrency": 2, "delay_seconds": 3},
            purpose="finance", track_record=True, enabled=True,
        ))
        await db.commit()
    svc, orch, *_ = _svc(session_factory)
    await _run(svc, dry_run=True)
    assert (orch.specs[0][0]["max_concurrency"], orch.specs[0][0]["delay_seconds"]) == (2, 3)
    await svc.run("Fin YT", date.today() - timedelta(days=100), 50, True, lambda m: True,
                  concurrency=3, delay=0.5, cookies_file="/c.txt")
    s = orch.specs[1][0]
    assert (s["max_concurrency"], s["delay_seconds"], s["cookies_file"]) == (3, 0.5, "/c.txt")
    assert "whisper_model" not in s
    await svc.run("Fin YT", date.today() - timedelta(days=100), 50, True, lambda m: True,
                  no_whisper=True)
    assert orch.specs[2][0]["whisper_model"] == ""


@pytest.mark.asyncio
async def test_second_run_skips_stored_successful_and_retries_failed(session_factory):
    await _config(session_factory)
    first = [_item("AAA"), _item("BBB", status="failed", transcript=""), _item("CCC", transcript="")]
    svc1, *_ = _svc(session_factory, data=first)
    await _run(svc1, dry_run=True)

    svc2, orch2, *_ = _svc(session_factory, data=[_item("BBB")])
    await _run(svc2, dry_run=True)
    assert orch2.specs[0][0]["skip_video_ids"] == ["AAA"]


@pytest.mark.asyncio
async def test_merge_stored_and_new_and_persist_only_new(session_factory):
    await _config(session_factory)
    svc1, *_ = _svc(session_factory, data=[_item("AAA")])
    await _run(svc1, dry_run=True)

    svc2, orch2, ext2, _ = _svc(session_factory, data=[_item("NEW", days=5)])
    report = await _run(svc2)
    assert report.items_found == 2 and report.items_new == 2
    assert sorted(e["video_id"] for e in ext2.received) == ["AAA", "NEW"]
    rows, _ = await _collected(session_factory)
    assert len(rows) == 2
    assert sorted(len(r.data) for r in rows) == [1, 1]  # archive not duplicated
    assert sorted(r.item_count for r in rows) == [1, 1]


@pytest.mark.asyncio
async def test_new_item_wins_on_id_clash(session_factory):
    await _config(session_factory)
    old = _item("AAA")
    old["title"] = "old"
    svc1, *_ = _svc(session_factory, data=[old])
    await _run(svc1, dry_run=True)
    new = _item("AAA")
    new["title"] = "new"
    svc2, _, ext2, _ = _svc(session_factory, data=[new])
    await _run(svc2, dry_run=True)
    report = await _run(_svc(session_factory, data=[new])[0], dry_run=True)
    assert [i["title"] for i in report.items] == ["new"]


@pytest.mark.asyncio
async def test_from_collected_still_works_with_incremental_rows(session_factory):
    await _config(session_factory)
    svc1, *_ = _svc(session_factory, data=[_item("AAA")])
    await _run(svc1, dry_run=True)
    svc2, *_ = _svc(session_factory, data=[_item("BBB")])
    await _run(svc2, dry_run=True)
    svc3, orch3, ext3, _ = _svc(session_factory, data=[])
    report = await svc3.run("Fin YT", date.today() - timedelta(days=100), 50, False,
                            lambda m: True, reuse_collected=True)
    assert orch3.specs == [] and report.items_found == 2
