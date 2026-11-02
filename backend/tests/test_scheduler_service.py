"""
Scheduler feeder-cycle tests against fakes (fake orchestrator, extraction, trender).

Verifies the new-item ledger filter and that pushed_to_trender is set end-to-end.
No real collectors, LLM, or network.
"""
import uuid

import pytest
from sqlalchemy import select

from src.models import CollectedData, Cycle, ExtractedMention, SourceConfig
from src.models.cycle import CycleStatus
from src.services.scheduler_service import SchedulerService


class FakeOrchestrator:
    def __init__(self, data):
        self._data = data
        self.calls = []

    async def collect(self, source_type, credential_ref, collect_spec, timeframe_days):
        self.calls.append((source_type, credential_ref))
        return {
            "data": self._data,
            "item_count": len(self._data),
            "source_info": {"channels": collect_spec.get("channels", [])},
            "metadata": {"collection_time_ms": 5, "source_type": source_type},
        }


class FakeExtraction:
    """Records the items it was given; returns one mention per item."""

    model = "fake-model"

    def __init__(self, said_at="2026-08-25T11:00:00Z"):
        self.received = None
        self.said_at = said_at

    async def extract(self, items):
        self.received = items
        mentions = []
        for it in items:
            vid = it.get("video_id")
            mentions.append({
                "symbol": "NVDA",
                "company_name": "NVIDIA",
                "source": "youtube",
                "stance": "bullish",
                "sentiment_score": 0.7,
                "confidence": 0.8,
                "context": "thesis + quote",
                "citation": {
                    "kind": "youtube",
                    "url": f"https://youtu.be/{vid}?t=10",
                    "video_id": vid,
                    "channel": it.get("channel"),
                    "title": it.get("title"),
                    "start_seconds": 10,
                    "quote": "q",
                },
                "mentioned_at": "2026-08-25T12:00:00Z",
                "is_recommendation": True,
                "action": "buy",
                "horizon": "short",
                "entry_price": 120.5,
                "target_price": 150,
                "stop_price": 110,
                "trigger_price": None,
                "thesis": "thesis",
                "quote": "q",
                "said_at": self.said_at,
                "said_at_precision": "datetime",
            })
        return mentions


class FakeTrender:
    def __init__(self, calls_error=None):
        self.pushed_payloads = []
        self.pushed_calls = []
        self.calls_error = calls_error

    async def push_calls(self, calls):
        if self.calls_error:
            raise self.calls_error
        self.pushed_calls.append(calls)
        return {"accepted": len(calls), "duplicates": 0, "rejected": []}

    async def push_mentions(self, mentions):
        self.pushed_payloads.append(mentions)
        return {"accepted": len(mentions), "rejected": 0}


async def _make_finance_config(session_factory, track_record=False):
    async with session_factory() as db:
        cfg = SourceConfig(
            id=uuid.uuid4(),
            name="Finance YT",
            source_type="youtube",
            credential_ref="YOUTUBE_API",
            collect_spec={"channels": ["@finance"]},
            purpose="finance",
            track_record=track_record,
            enabled=True,
        )
        db.add(cfg)
        await db.commit()
        await db.refresh(cfg)
        return cfg


@pytest.mark.asyncio
async def test_full_cycle_stores_and_pushes(session_factory):
    cfg = await _make_finance_config(session_factory)
    data = [
        {"video_id": "AAA", "channel": "@finance", "title": "One", "transcript": "buy nvda"},
        {"video_id": "BBB", "channel": "@finance", "title": "Two", "transcript": "still nvda"},
    ]
    orch, extraction, trender = FakeOrchestrator(data), FakeExtraction(), FakeTrender()
    svc = SchedulerService(
        session_factory=session_factory,
        orchestrator=orch,
        extraction_service=extraction,
        trender_client=trender,
    )

    cycle = await svc.run_finance_cycle(cfg)

    assert cycle.status == CycleStatus.COMPLETED

    async with session_factory() as db:
        rows = (await db.execute(select(ExtractedMention))).scalars().all()
        assert len(rows) == 2
        assert all(r.pushed_to_trender for r in rows)
        assert all("accepted" in (r.trender_response or "") for r in rows)

    # Both new items were extracted and one batch pushed.
    assert len(extraction.received) == 2
    assert len(trender.pushed_payloads) == 1
    assert len(trender.pushed_payloads[0]) == 2


@pytest.mark.asyncio
async def test_new_item_filter_skips_already_seen(session_factory):
    cfg = await _make_finance_config(session_factory)

    # Prior cycle already collected video AAA.
    async with session_factory() as db:
        prior = Cycle(name="prior", status=CycleStatus.COMPLETED, config_snapshot={})
        db.add(prior)
        await db.commit()
        await db.refresh(prior)
        db.add(CollectedData(
            cycle_id=prior.id,
            source_type="youtube",
            source_name="@finance",
            data=[{"video_id": "AAA", "transcript": "old"}],
            item_count=1,
        ))
        await db.commit()

    data = [
        {"video_id": "AAA", "channel": "@finance", "title": "Old", "transcript": "seen"},
        {"video_id": "CCC", "channel": "@finance", "title": "New", "transcript": "fresh"},
    ]
    orch, extraction, trender = FakeOrchestrator(data), FakeExtraction(), FakeTrender()
    svc = SchedulerService(
        session_factory=session_factory,
        orchestrator=orch,
        extraction_service=extraction,
        trender_client=trender,
    )

    await svc.run_finance_cycle(cfg)

    # Only the unseen video (CCC) was passed to extraction.
    assert [it["video_id"] for it in extraction.received] == ["CCC"]

    async with session_factory() as db:
        rows = (await db.execute(select(ExtractedMention))).scalars().all()
        assert len(rows) == 1
        assert rows[0].citation_json["video_id"] == "CCC"
        assert rows[0].pushed_to_trender is True


@pytest.mark.asyncio
async def test_dedupe_key_prevents_duplicate_insert(session_factory):
    """Re-running the same content does not double-insert mentions."""
    cfg = await _make_finance_config(session_factory)
    data = [{"video_id": "AAA", "channel": "@finance", "title": "One", "transcript": "buy nvda"}]

    # First run.
    svc1 = SchedulerService(
        session_factory=session_factory,
        orchestrator=FakeOrchestrator(data),
        extraction_service=FakeExtraction(),
        trender_client=FakeTrender(),
    )
    await svc1.run_finance_cycle(cfg)

    # Second run: ledger filters the item, so extraction gets nothing new.
    extraction2 = FakeExtraction()
    svc2 = SchedulerService(
        session_factory=session_factory,
        orchestrator=FakeOrchestrator(data),
        extraction_service=extraction2,
        trender_client=FakeTrender(),
    )
    await svc2.run_finance_cycle(cfg)

    assert extraction2.received == []
    async with session_factory() as db:
        rows = (await db.execute(select(ExtractedMention))).scalars().all()
        assert len(rows) == 1  # no duplicate


DATA_ONE = [{"video_id": "AAA", "channel": "@finance", "title": "One", "transcript": "buy nvda"}]


def _svc(session_factory, extraction, trender):
    return SchedulerService(
        session_factory=session_factory,
        orchestrator=FakeOrchestrator(DATA_ONE),
        extraction_service=extraction,
        trender_client=trender,
    )


@pytest.mark.asyncio
async def test_cycle_pushes_calls_only_for_track_record_configs(session_factory):
    # Non-track-record config: mentions pushed, no calls.
    plain = await _make_finance_config(session_factory, track_record=False)
    trender = FakeTrender()
    await _svc(session_factory, FakeExtraction(), trender).run_finance_cycle(plain)
    assert len(trender.pushed_payloads) == 1
    assert trender.pushed_calls == []
    async with session_factory() as db:
        row = (await db.execute(select(ExtractedMention))).scalars().one()
        assert row.pushed_calls_to_trender is False
        assert row.is_recommendation is True
        assert row.action == "buy"
        assert row.horizon == "short"
        assert float(row.entry_price) == 120.5
        assert row.said_at == "2026-08-25T11:00:00Z"
        assert row.said_at_precision == "datetime"
        assert row.quote == "q"

    # Track-record config (new video) -> calls pushed with the call payload.
    tracked = await _make_finance_config(session_factory, track_record=True)
    data = [{"video_id": "BBB", "channel": "@finance", "title": "Two", "transcript": "buy"}]
    trender2 = FakeTrender()
    svc = SchedulerService(
        session_factory=session_factory,
        orchestrator=FakeOrchestrator(data),
        extraction_service=FakeExtraction(),
        trender_client=trender2,
    )
    await svc.run_finance_cycle(tracked)
    assert len(trender2.pushed_calls) == 1
    call = trender2.pushed_calls[0][0]
    assert call["symbol"] == "NVDA"
    assert call["origin"] == "feeder"
    assert call["extractor_model"] == "fake-model"
    assert call["source_item_id"] == "BBB"
    async with session_factory() as db:
        row = (
            await db.execute(select(ExtractedMention).where(ExtractedMention.dedupe_key.contains("BBB")))
        ).scalars().one()
        assert row.pushed_calls_to_trender is True
        assert "accepted" in row.trender_calls_response


@pytest.mark.asyncio
async def test_cycle_skips_calls_without_said_at(session_factory):
    cfg = await _make_finance_config(session_factory, track_record=True)
    trender = FakeTrender()
    await _svc(session_factory, FakeExtraction(said_at=None), trender).run_finance_cycle(cfg)

    assert len(trender.pushed_payloads) == 1  # mention still pushed for discovery
    assert trender.pushed_calls == []  # no call payload -> nothing sent
    async with session_factory() as db:
        row = (await db.execute(select(ExtractedMention))).scalars().one()
        assert row.pushed_to_trender is True
        assert row.pushed_calls_to_trender is False
        assert row.trender_calls_response is None


@pytest.mark.asyncio
async def test_call_push_failure_leaves_flag_false(session_factory):
    cfg = await _make_finance_config(session_factory, track_record=True)
    trender = FakeTrender(calls_error=RuntimeError("calls down"))
    cycle = await _svc(session_factory, FakeExtraction(), trender).run_finance_cycle(cfg)

    assert cycle.status == CycleStatus.COMPLETED  # call-push failure doesn't fail the cycle
    async with session_factory() as db:
        row = (await db.execute(select(ExtractedMention))).scalars().one()
        assert row.pushed_to_trender is True
        assert row.pushed_calls_to_trender is False
        assert "calls down" in row.trender_calls_response


@pytest.mark.asyncio
async def test_track_record_cycle_sweeps_previously_failed_calls(session_factory):
    cfg = await _make_finance_config(session_factory, track_record=True)
    await _svc(session_factory, FakeExtraction(), FakeTrender(calls_error=RuntimeError("down"))) \
        .run_finance_cycle(cfg)
    async with session_factory() as db:
        row = (await db.execute(select(ExtractedMention))).scalars().one()
        assert row.pushed_calls_to_trender is False and row.call_payload_json["source_item_id"] == "AAA"

    # Next cycle sees no new items (ledger) but re-sends the stored payload first.
    extraction, trender = FakeExtraction(), FakeTrender()
    await _svc(session_factory, extraction, trender).run_finance_cycle(cfg)
    assert extraction.received == []
    assert [[c["source_item_id"] for c in batch] for batch in trender.pushed_calls] == [["AAA"]]
    async with session_factory() as db:
        row = (await db.execute(select(ExtractedMention))).scalars().one()
        assert row.pushed_calls_to_trender is True


@pytest.mark.asyncio
async def test_non_track_record_cycle_stores_no_payload_and_does_not_sweep(session_factory):
    tracked = await _make_finance_config(session_factory, track_record=True)
    await _svc(session_factory, FakeExtraction(), FakeTrender(calls_error=RuntimeError("down"))) \
        .run_finance_cycle(tracked)

    plain = await _make_finance_config(session_factory, track_record=False)
    data = [{"video_id": "BBB", "channel": "@finance", "title": "Two", "transcript": "buy"}]
    trender = FakeTrender()
    await SchedulerService(
        session_factory=session_factory, orchestrator=FakeOrchestrator(data),
        extraction_service=FakeExtraction(), trender_client=trender,
    ).run_finance_cycle(plain)
    assert trender.pushed_calls == []
    async with session_factory() as db:
        row = (
            await db.execute(select(ExtractedMention).where(ExtractedMention.dedupe_key.contains("BBB")))
        ).scalars().one()
        assert row.call_payload_json is None
