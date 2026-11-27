"""
Track-record call pushes follow the feeder's push rules (see scheduler_service docstring):
batches of at most PUSH_BATCH_SIZE, a call is flagged only once trender has it, 400/413/422
batches split down to single calls, any other failure stops the loop, unpushed calls are
retried for BACKLOG_MAX_AGE_DAYS, a single call refused as invalid is never retried.
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from src.models import Cycle, ExtractedMention
from src.models.cycle import CycleStatus
from src.services.scheduler_service import (
    BACKLOG_MAX_AGE_DAYS,
    PUSH_BATCH_SIZE,
    SchedulerService,
)
from src.services.trender_client import TrenderPushError


def _payload(symbol: str, item: str) -> dict:
    return {
        "symbol": symbol, "company_name": None, "source": "youtube", "action": "buy",
        "stance": "bullish", "horizon": "short", "conviction": 0.8, "entry_price": None,
        "target_price": None, "stop_price": None, "trigger_price": None, "thesis": "t",
        "quote": "q", "citation": {"channel": "@fin", "video_id": item}, "source_item_id": item,
        "said_at": "2026-11-20T12:00:00Z", "said_at_precision": "datetime", "origin": "feeder",
        "extractor_model": "m", "extraction_version": "tr-1",
    }


async def _rows(session_factory, symbols, *, created_at=None, start=0):
    async with session_factory() as db:
        cycle = Cycle(name="finance-feeder:x", status=CycleStatus.COMPLETED, config_snapshot={})
        db.add(cycle)
        await db.flush()
        for i, symbol in enumerate(symbols):
            row = ExtractedMention(
                cycle_id=cycle.id, symbol=symbol, citation_json={}, dedupe_key=f"k-{uuid.uuid4()}",
                pushed_to_trender=True, pushed_calls_to_trender=False,
                call_payload_json=_payload(symbol, f"V{start + i}"),
            )
            if created_at is not None:
                row.created_at = created_at
            db.add(row)
        await db.commit()


async def _state(session_factory):
    async with session_factory() as db:
        rows = (await db.execute(select(ExtractedMention))).scalars().all()
        return {r.call_payload_json["source_item_id"]: (r.pushed_calls_to_trender, r.trender_calls_response)
                for r in rows}


class ScriptedTrender:
    """push_calls answers from `script(calls)`: a dict response or an exception to raise."""

    def __init__(self, script=None):
        self.requests = []
        self.script = script or (lambda calls: {"accepted": len(calls), "duplicates": 0, "rejected": []})

    async def push_calls(self, calls):
        self.requests.append([c["source_item_id"] for c in calls])
        outcome = self.script(calls)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


async def _repush(session_factory, trender):
    svc = SchedulerService(session_factory=session_factory, orchestrator=None,
                           extraction_service=None, trender_client=trender)
    async with session_factory() as db:
        return await svc.repush_unpushed_calls(db)


@pytest.mark.asyncio
async def test_calls_go_in_batches_of_push_batch_size(session_factory):
    await _rows(session_factory, [f"S{i}" for i in range(2 * PUSH_BATCH_SIZE + 20)])
    trender = ScriptedTrender()
    result = await _repush(session_factory, trender)
    assert [len(r) for r in trender.requests] == [PUSH_BATCH_SIZE, PUSH_BATCH_SIZE, 20]
    assert result.sent == result.accepted == 2 * PUSH_BATCH_SIZE + 20 and result.failed == 0
    assert all(pushed for pushed, _ in (await _state(session_factory)).values())


@pytest.mark.asyncio
async def test_nothing_is_pushed_twice(session_factory):
    await _rows(session_factory, ["NVDA", "AAPL"])
    trender = ScriptedTrender()
    await _repush(session_factory, trender)
    await _repush(session_factory, trender)
    assert trender.requests == [["V0", "V1"]]


@pytest.mark.asyncio
async def test_a_symbol_repeats_only_across_requests(session_factory):
    await _rows(session_factory, ["NVDA", "NVDA", "AAPL"])
    trender = ScriptedTrender()
    await _repush(session_factory, trender)
    assert trender.requests == [["V0", "V2"], ["V1"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [
    TrenderPushError("Failed to push after retries: connect failed"),
    TrenderPushError("trender returned 401: bad token", status_code=401),
    TrenderPushError("trender returned 503: not configured", status_code=503),
])
async def test_unreachable_or_unauthorised_stops_and_retries_next_time(session_factory, error):
    await _rows(session_factory, [f"S{i}" for i in range(PUSH_BATCH_SIZE + 5)])
    trender = ScriptedTrender(lambda calls: error)
    result = await _repush(session_factory, trender)
    assert len(trender.requests) == 1  # the loop stops at the first failed request
    assert result.failed == PUSH_BATCH_SIZE + 5
    assert not any(pushed for pushed, _ in (await _state(session_factory)).values())

    retry = ScriptedTrender()
    await _repush(session_factory, retry)
    assert sum(len(r) for r in retry.requests) == PUSH_BATCH_SIZE + 5
    assert all(pushed for pushed, _ in (await _state(session_factory)).values())


@pytest.mark.asyncio
async def test_invalid_batch_is_split_and_the_bad_call_is_not_retried(session_factory):
    await _rows(session_factory, ["NVDA", "BAD", "AAPL"])

    def script(calls):
        if any(c["symbol"] == "BAD" for c in calls):
            return TrenderPushError("trender returned 422: invalid", status_code=422)
        return {"accepted": len(calls), "duplicates": 0, "rejected": []}

    trender = ScriptedTrender(script)
    result = await _repush(session_factory, trender)
    assert trender.requests == [["V0", "V1", "V2"], ["V0"], ["V1"], ["V2"]]
    state = await _state(session_factory)
    assert state["V0"][0] and state["V2"][0]
    assert state["V1"] == (False, "error: trender returned 422: invalid")
    assert result.accepted == 2 and result.failed == 1

    later = ScriptedTrender()
    await _repush(session_factory, later)
    assert later.requests == []  # refused as invalid: permanent


@pytest.mark.asyncio
async def test_rejected_call_stays_unpushed_and_is_retried(session_factory):
    await _rows(session_factory, ["NVDA", "AAPL"])
    trender = ScriptedTrender(lambda calls: {
        "accepted": 1, "duplicates": 0, "rejected": [{"symbol": "AAPL", "reason": "unmapped channel"}],
    })
    result = await _repush(session_factory, trender)
    assert result.rejected == {"unmapped channel": 1} and result.failed == 0
    state = await _state(session_factory)
    assert state["V0"][0] is True
    assert state["V1"][0] is False and state["V1"][1].startswith("rejected: unmapped channel")

    # The channel gets mapped in trender: the next run sends only the rejected call.
    mapped = ScriptedTrender()
    await _repush(session_factory, mapped)
    assert mapped.requests == [["V1"]]
    assert (await _state(session_factory))["V1"][0] is True


@pytest.mark.asyncio
async def test_duplicate_counts_as_pushed(session_factory):
    await _rows(session_factory, ["NVDA"])
    await _repush(session_factory, ScriptedTrender(lambda calls: {"accepted": 0, "duplicates": 1, "rejected": []}))
    assert (await _state(session_factory))["V0"][0] is True


@pytest.mark.asyncio
async def test_calls_older_than_the_backlog_window_are_not_retried(session_factory):
    old = datetime.now(timezone.utc) - timedelta(days=BACKLOG_MAX_AGE_DAYS + 1)
    await _rows(session_factory, ["OLD"], created_at=old)
    await _rows(session_factory, ["NEW"], start=1)
    trender = ScriptedTrender()
    await _repush(session_factory, trender)
    assert trender.requests == [["V1"]]
    state = await _state(session_factory)
    assert state["V0"][0] is False and state["V1"][0] is True


@pytest.mark.asyncio
async def test_manual_repush_has_no_age_window_but_skips_refused_calls(session_factory):
    """backfill_track_record.py --repush is an operator action: every eligible call, any age."""
    old = datetime.now(timezone.utc) - timedelta(days=BACKLOG_MAX_AGE_DAYS + 30)
    await _rows(session_factory, ["OLD", "BAD"], created_at=old)
    async with session_factory() as db:
        bad = (await db.execute(select(ExtractedMention).where(ExtractedMention.symbol == "BAD"))).scalars().one()
        bad.trender_calls_response = "error: trender returned 422: invalid"
        await db.commit()

    trender = ScriptedTrender()
    svc = SchedulerService(session_factory=session_factory, orchestrator=None,
                           extraction_service=None, trender_client=trender)
    async with session_factory() as db:
        await svc.repush_unpushed_calls(db, max_age_days=None)
    assert trender.requests == [["V0"]]


@pytest.mark.asyncio
async def test_batch_size_follows_trender_push_batch_size(session_factory, monkeypatch):
    monkeypatch.setenv("TRENDER_PUSH_BATCH_SIZE", "7")
    await _rows(session_factory, [f"S{i}" for i in range(16)])
    trender = ScriptedTrender()
    await _repush(session_factory, trender)
    assert [len(r) for r in trender.requests] == [7, 7, 2]


# --- one log line per call -------------------------------------------------------------------

def _call_lines(caplog):
    return [r for r in caplog.records if r.getMessage().startswith("call ")]


@pytest.mark.asyncio
async def test_every_call_gets_a_line_with_channel_item_symbol_outcome_and_reason(session_factory, caplog):
    import logging

    await _rows(session_factory, ["NVDA", "AYAL", "TEVA"])

    def script(calls):
        return {"accepted": 1, "duplicates": 1,
                "rejected": [{"symbol": "AYAL", "reason": "unknown symbol"}]}

    with caplog.at_level(logging.INFO):
        await _repush(session_factory, ScriptedTrender(script))
    lines = {r.getMessage().split("symbol=")[1].split()[0]: r for r in _call_lines(caplog)}
    assert set(lines) == {"NVDA", "AYAL", "TEVA"}
    bad = lines["AYAL"]
    assert bad.levelno == logging.WARNING
    assert "outcome=rejected" in bad.getMessage() and "reason=unknown symbol" in bad.getMessage()
    assert "channel=@fin" in bad.getMessage() and "item=V1" in bad.getMessage()
    # trender does not say which of the two good calls was the duplicate
    assert "outcome=accepted_or_duplicate" in lines["NVDA"].getMessage()
    assert lines["NVDA"].levelno == logging.INFO


@pytest.mark.asyncio
async def test_all_accepted_and_all_duplicate_are_named_exactly(session_factory, caplog):
    import logging

    await _rows(session_factory, ["NVDA"])
    with caplog.at_level(logging.INFO):
        await _repush(session_factory, ScriptedTrender())
    assert "outcome=accepted " in _call_lines(caplog)[0].getMessage() + " "

    caplog.clear()
    await _rows(session_factory, ["AAPL"], start=5)
    with caplog.at_level(logging.INFO):
        await _repush(session_factory, ScriptedTrender(
            lambda c: {"accepted": 0, "duplicates": 1, "rejected": []}))
    assert "outcome=duplicate " in _call_lines(caplog)[0].getMessage() + " "


@pytest.mark.asyncio
async def test_failed_to_send_is_logged_per_call_with_the_reason(session_factory, caplog):
    import logging

    await _rows(session_factory, [f"S{i}" for i in range(PUSH_BATCH_SIZE + 2)])
    err = TrenderPushError("Failed to push 50 call(s) to http://t/api: ReadTimeout after 120s",
                           reason="ReadTimeout after 120s")
    with caplog.at_level(logging.INFO):
        await _repush(session_factory, ScriptedTrender(lambda calls: err))
    lines = _call_lines(caplog)
    assert len(lines) == PUSH_BATCH_SIZE + 2  # the unattempted second batch is logged too
    assert all("outcome=failed to send" in r.getMessage() and r.levelno == logging.WARNING for r in lines)
    assert all("ReadTimeout after 120s" in r.getMessage() or "not attempted" in r.getMessage() for r in lines)


@pytest.mark.asyncio
async def test_a_split_batch_logs_each_call_once_with_its_own_outcome(session_factory, caplog):
    import logging

    await _rows(session_factory, ["NVDA", "BAD"])

    def script(calls):
        if len(calls) > 1 or calls[0]["symbol"] == "BAD":
            return TrenderPushError("trender returned 422: invalid", status_code=422, reason="HTTP 422")
        return {"accepted": 1, "duplicates": 0, "rejected": []}

    with caplog.at_level(logging.INFO):
        await _repush(session_factory, ScriptedTrender(script))
    msgs = [r.getMessage() for r in _call_lines(caplog)]
    assert len(msgs) == 2
    assert any("symbol=NVDA" in m and "outcome=accepted" in m for m in msgs)
    assert any("symbol=BAD" in m and "outcome=failed to send" in m and "HTTP 422" in m for m in msgs)


@pytest.mark.asyncio
async def test_outcomes_are_counted_in_the_run_summary(session_factory, tmp_path):
    from src import run_log

    await _rows(session_factory, ["NVDA", "AYAL"])

    def script(calls):
        return {"accepted": 1, "duplicates": 0, "rejected": [{"symbol": "AYAL", "reason": "unknown symbol"}]}

    with run_log.run_log("repush", None, log_dir=tmp_path) as run:
        await _repush(session_factory, ScriptedTrender(script))
    assert run.summary["calls"]["accepted"] == 1
    assert run.summary["calls"]["rejected"] == {"unknown symbol": 1}
