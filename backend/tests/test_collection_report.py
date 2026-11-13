"""collection_report: one status row per item, newest attempt wins (a success beats a later failure), errors kept."""
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

_spec = importlib.util.spec_from_file_location(
    "collection_report", Path(__file__).resolve().parent.parent / "scripts" / "collection_report.py")
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def _row(items, day):
    return (SimpleNamespace(data=items), datetime(2026, 11, day, tzinfo=timezone.utc))


def test_newest_attempt_wins_and_errors_are_kept():
    newest = _row([
        {"video_id": "A", "url": "u/A", "title": "A", "transcript": "hello", "transcript_source": "whisper"},
        {"video_id": "B", "status": "failed", "error": "Failed to download audio"},
    ], 12)
    older = _row([
        {"video_id": "A", "status": "failed", "error": "429"},
        {"video_id": "C", "title": "C", "transcript": "", "transcript_source": "youtube_api"},
    ], 7)
    rows = {r["item_id"]: r for r in report.status_rows([newest, older], "youtube")}
    assert rows["A"]["status"] == "ok" and rows["A"]["transcript_source"] == "whisper"
    assert rows["A"]["transcript_chars"] == 5 and rows["A"]["attempted_at"].startswith("2026-11-12")
    assert rows["B"]["status"] == "failed" and rows["B"]["error"] == "Failed to download audio"
    assert rows["C"]["status"] == "no_text"


def test_success_not_hidden_by_later_failure():
    newest = _row([{"video_id": "A", "status": "failed", "error": "Sign in to confirm"}], 12)
    older = _row([{"video_id": "A", "title": "A", "transcript": "hi", "transcript_source": "youtube_api"}], 6)
    rows = {r["item_id"]: r for r in report.status_rows([newest, older], "youtube")}
    assert rows["A"]["status"] == "ok" and rows["A"]["attempted_at"].startswith("2026-11-06")
