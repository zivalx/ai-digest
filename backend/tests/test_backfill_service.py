"""
BackfillService tests against fakes (no network, no LLM).
"""
import uuid
from datetime import date, timedelta

import pytest
from sqlalchemy import select

from src.models import Cycle, ExtractedMention, SourceConfig, TrackRecordProcessed
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

    def __init__(self, fail_ids=(), raise_all=False):
        self.received = []
        self.fail_ids = set(fail_ids)
        self.raise_all = raise_all

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
                "symbol": "NVDA", "company_name": "NVIDIA", "source": "youtube",
                "stance": "bullish", "sentiment_score": 0.7, "confidence": 0.8,
                "context": "thesis",
                "citation": {
                    "kind": "youtube", "url": f"https://youtu.be/{vid}?t=10",
                    "video_id": vid, "channel": it.get("channel"),
                    "title": it.get("title"), "start_seconds": 10,
                },
                "is_recommendation": True, "action": "buy", "horizon": "swing",
                "thesis": "thesis", "quote": "buy it",
                "said_at": "2026-01-05T10:00:00Z", "said_at_precision": "datetime",
            })
        return out


class FakeTrender:
    def __init__(self, calls_error=None):
        self.calls_error = calls_error
        self.pushed_mentions = []
        self.pushed_calls = []

    async def push_mentions(self, mentions):
        self.pushed_mentions.append(mentions)
        return {"accepted": len(mentions)}

    async def push_calls(self, calls):
        if self.calls_error:
            raise self.calls_error
        self.pushed_calls.append(calls)
        return {"accepted": len(calls)}


DATA = [
    {"video_id": "AAA", "channel": "@fin", "title": "One", "transcript": "a" * 400},
    {"video_id": "BBB", "channel": "@fin", "title": "Two", "transcript": "b" * 400},
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
