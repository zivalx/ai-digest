"""
normalize_published tests — publish timestamps must reach trender as
ISO-8601 UTC, never as yt-dlp's bare YYYYMMDD (which pydantic parses as a
Unix epoch, i.e. 1970).
"""
from datetime import datetime, timezone

from src.services.scheduler_service import SchedulerService
from src.services.timestamps import normalize_published


def test_yt_dlp_upload_date_becomes_iso_date():
    assert normalize_published("20260815") == ("2026-08-15T00:00:00Z", "date")


def test_epoch_seconds_become_iso_datetime():
    assert normalize_published(1786795200) == ("2026-08-15T12:00:00Z", "datetime")


def test_aware_datetime_converted_to_utc():
    dt = datetime(2026, 8, 15, 15, 30, tzinfo=timezone.utc)
    assert normalize_published(dt) == ("2026-08-15T15:30:00Z", "datetime")


def test_naive_datetime_assumed_utc():
    assert normalize_published(datetime(2026, 8, 15, 15, 30)) == ("2026-08-15T15:30:00Z", "datetime")


def test_iso_strings_normalized():
    assert normalize_published("2026-08-15T15:30:00+03:00") == ("2026-08-15T12:30:00Z", "datetime")
    assert normalize_published("2026-08-15") == ("2026-08-15T00:00:00Z", "date")


def test_missing_or_garbage_returns_none():
    for value in (None, "", "Unknown", "2026-13-45", "123"):
        assert normalize_published(value) == (None, None)


def test_youtube_item_mapping_uses_normalized_upload_date():
    item = {"video_id": "abc", "upload_date": "20260815", "transcript": "x"}
    mapped = SchedulerService._to_extraction_item(item, "youtube")
    assert mapped["ts"] == "2026-08-15T00:00:00Z"
    assert mapped["ts_precision"] == "date"


def test_youtube_item_without_date_has_no_ts():
    mapped = SchedulerService._to_extraction_item({"video_id": "abc", "upload_date": ""}, "youtube")
    assert mapped["ts"] is None
