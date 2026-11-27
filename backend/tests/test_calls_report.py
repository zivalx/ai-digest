"""calls_report: per channel, track-record calls pending / rejected / failed, newest first."""
import csv
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from src import calls_report
from src.models import Cycle, ExtractedMention
from src.models.cycle import CycleStatus

T0 = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)


def _payload(symbol, item, channel, action="buy"):
    return {"symbol": symbol, "action": action, "source": "telegram", "source_item_id": item,
            "said_at": "2026-11-19T10:00:00Z",
            "citation": {"kind": "telegram", "channel": channel, "message_link": item}}


async def _seed(session_factory):
    rows = [  # symbol, channel, response, pushed, age (days older than T0)
        ("AYAL", "ch_il", "rejected: unknown symbol; {'accepted': 0}", False, 0),
        ("ELAL", "ch_il", "rejected: unknown symbol; {'accepted': 0}", False, 1),
        ("TEVA", "ch_il", None, False, 2),
        ("NVDA", "ch_us", "error: Failed to push 3 call(s) to http://t/x: ReadTimeout after 120s", False, 0),
        ("AAPL", "ch_us", "{'accepted': 1}", True, 0),  # pushed: not in the report
    ]
    async with session_factory() as db:
        cycle = Cycle(name="c", status=CycleStatus.COMPLETED, config_snapshot={})
        db.add(cycle)
        await db.flush()
        for symbol, channel, response, pushed, age in rows:
            row = ExtractedMention(
                cycle_id=cycle.id, symbol=symbol, citation_json={}, dedupe_key=f"k-{uuid.uuid4()}",
                call_payload_json=_payload(symbol, f"https://t.me/{channel}/{symbol}", channel),
                pushed_calls_to_trender=pushed, trender_calls_response=response,
            )
            row.created_at = T0 - timedelta(days=age)
            db.add(row)
        # a mention that never became a call (no payload) must not appear
        db.add(ExtractedMention(cycle_id=cycle.id, symbol="JUNK", citation_json={},
                                dedupe_key="k-junk", pushed_calls_to_trender=False))
        await db.commit()


@pytest.mark.asyncio
async def test_classifies_and_groups_by_channel_newest_first(session_factory):
    await _seed(session_factory)
    report = await calls_report.collect(session_factory)
    assert [r.symbol for r in report] == ["AYAL", "ELAL", "TEVA", "NVDA"]
    by = {r.symbol: r for r in report}
    assert (by["AYAL"].status, by["AYAL"].reason) == ("rejected", "unknown symbol")
    assert by["TEVA"].status == "pending" and by["TEVA"].reason == ""
    assert by["NVDA"].status == "failed" and "ReadTimeout after 120s" in by["NVDA"].reason
    assert by["AYAL"].link == "https://t.me/ch_il/AYAL"

    text = calls_report.format_report(report)
    il, us = text.index("== ch_il"), text.index("== ch_us")
    assert il < us
    assert "pending 1, rejected 2, failed 0" in text and "pending 0, rejected 0, failed 1" in text
    # newest first within a channel
    assert text.index("AYAL") < text.index("ELAL") < text.index("TEVA")
    assert "https://t.me/ch_il/AYAL" in text and "unknown symbol" in text


@pytest.mark.asyncio
async def test_is_read_only_and_filters(session_factory):
    await _seed(session_factory)
    report = await calls_report.collect(session_factory, channel="ch_us")
    assert [r.symbol for r in report] == ["NVDA"]
    report = await calls_report.collect(session_factory, statuses={"rejected"})
    assert {r.symbol for r in report} == {"AYAL", "ELAL"}
    async with session_factory() as db:
        rows = (await db.execute(select(ExtractedMention))).scalars().all()
    assert sum(1 for r in rows if r.pushed_calls_to_trender) == 1  # nothing was changed


@pytest.mark.asyncio
async def test_csv_has_the_same_rows(session_factory, tmp_path):
    await _seed(session_factory)
    report = await calls_report.collect(session_factory)
    path = tmp_path / "calls.csv"
    calls_report.write_csv(report, path)
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    assert len(rows) == 4
    assert set(rows[0]) == {"channel", "status", "symbol", "action", "said_at", "reason", "link",
                            "source_item_id", "created_at"}
    assert {r["status"] for r in rows} == {"pending", "rejected", "failed"}


@pytest.mark.asyncio
async def test_empty_report_says_so(session_factory):
    assert "No pending, rejected or failed calls" in calls_report.format_report(
        await calls_report.collect(session_factory))


def test_cli_prints_and_writes_csv(session_factory, tmp_path, monkeypatch, capsys):
    import asyncio

    asyncio.run(_seed(session_factory))
    monkeypatch.setattr(calls_report, "AsyncSessionLocal", session_factory)
    out = tmp_path / "r.csv"
    assert calls_report.main(["--csv", str(out)]) == 0
    assert "== ch_il" in capsys.readouterr().out
    assert out.exists()
