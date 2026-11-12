"""export_transcripts script: paths, content, undated/Hebrew handling, idempotency."""
import csv
import importlib.util
import json
import uuid
from pathlib import Path

import pytest

from src.models import CollectedData, Cycle
from src.models.cycle import CycleStatus

_spec = importlib.util.spec_from_file_location(
    "export_transcripts", Path(__file__).resolve().parent.parent / "scripts" / "export_transcripts.py")
et = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(et)


async def _store(session_factory, items, source_type="youtube"):
    async with session_factory() as db:
        cycle = Cycle(id=uuid.uuid4(), name="backfill:x", status=CycleStatus.COMPLETED,
                      config_snapshot={})
        db.add(cycle)
        await db.flush()
        db.add(CollectedData(cycle_id=cycle.id, source_type=source_type, data=items,
                             item_count=len(items)))
        await db.commit()


ITEMS = [
    {"video_id": "AAA", "channel": "Fin Guy!", "title": "One", "url": "https://y/AAA",
     "transcript": "hello", "upload_date": "20260115", "transcript_source": "captions",
     "transcript_segments": [{"start": 0, "text": "hello"}]},
    {"video_id": "BBB", "channel": "ערוץ הכסף", "title": "שתיים", "transcript": "שלום עולם",
     "published_at": "2026-02-03T10:00:00Z"},
    {"video_id": "CCC", "channel": "Fin Guy!", "title": "No date", "transcript": "x"},
]


@pytest.mark.asyncio
async def test_export_paths_and_content(session_factory, tmp_path):
    await _store(session_factory, ITEMS)
    n = et.export(await et.load_rows(session_factory), tmp_path)
    assert n == 3
    a = tmp_path / "fin-guy" / "2026-01" / "2026-01-15_AAA.json"
    rec = json.loads(a.read_text(encoding="utf-8"))
    assert rec["item_id"] == "AAA" and rec["source_type"] == "youtube"
    assert rec["published_at"] == "2026-01-15T00:00:00Z" and rec["ts_precision"] == "date"
    assert rec["transcript"] == "hello" and rec["transcript_source"] == "captions"
    assert rec["transcript_segments"] == [{"start": 0, "text": "hello"}]
    assert (tmp_path / "fin-guy" / "undated" / "CCC.json").exists()
    heb = tmp_path / "ערוץ-הכסף" / "2026-02" / "2026-02-03_BBB.json"
    assert "שלום עולם" in heb.read_text(encoding="utf-8")  # ensure_ascii=False
    rows = list(csv.DictReader(open(tmp_path / "fin-guy" / "index.csv", encoding="utf-8")))
    assert [r["item_id"] for r in rows] == ["CCC", "AAA"]  # undated ("") sorts first
    assert rows[1]["transcript_chars"] == "5" and rows[1]["url"] == "https://y/AAA"
    assert rows[1]["file"] == "fin-guy/2026-01/2026-01-15_AAA.json"


@pytest.mark.asyncio
async def test_export_idempotent_dedupe_latest_wins_and_filter(session_factory, tmp_path):
    await _store(session_factory, ITEMS)
    await _store(session_factory, [dict(ITEMS[0], transcript="newer")])
    await _store(session_factory, [{"message_id": 7, "channel": "tg", "text": "msg",
                                    "date": "2026-03-01T00:00:00Z"}], source_type="telegram")
    rows = await et.load_rows(session_factory)
    assert et.export(rows, tmp_path, "youtube") == 3
    first = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert not (tmp_path / "tg").exists()
    assert json.loads((tmp_path / "fin-guy/2026-01/2026-01-15_AAA.json").read_text())["transcript"] == "newer"
    et.export(rows, tmp_path, "youtube")
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == first
    et.export(rows, tmp_path, "telegram")
    assert (tmp_path / "tg/2026-03/2026-03-01_7.json").exists()


@pytest.mark.asyncio
async def test_export_skips_failed_items_and_keeps_telegram_link(session_factory, tmp_path):
    await _store(session_factory, ITEMS[:1])
    await _store(session_factory, [{"video_id": "AAA", "status": "failed", "error": "blocked"},
                                   {"video_id": "DDD", "status": "failed", "error": "blocked"}])
    await _store(session_factory, [{"message_id": 9, "channel": "tg", "text": "msg",
                                    "message_link": "https://t.me/tg/9",
                                    "date": "2026-03-01T00:00:00Z"}], source_type="telegram")
    assert et.export(await et.load_rows(session_factory), tmp_path) == 2
    assert not (tmp_path / "unknown").exists()
    assert json.loads((tmp_path / "fin-guy/2026-01/2026-01-15_AAA.json").read_text())["transcript"] == "hello"
    rows = list(csv.DictReader(open(tmp_path / "tg" / "index.csv", encoding="utf-8")))
    assert rows[0]["url"] == "https://t.me/tg/9"
