"""
Publish-timestamp normalization for items pushed to trender.

Connectors hand us publish times in several shapes: yt-dlp's bare
``YYYYMMDD`` ``upload_date``, epoch seconds, datetimes (telegram), or ISO
strings. trender's pydantic models parse a bare numeric string like
``"20260815"`` as a Unix epoch (-> 1970-08-23), so everything is normalized
here to an ISO-8601 UTC string with a trailing ``Z`` plus its precision.
"""
import re
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

_YYYYMMDD = re.compile(r"^\d{8}$")
_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _fmt(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_published(value: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Return ``(iso_utc, precision)`` where precision is ``"datetime"`` or
    ``"date"``; ``(None, None)`` when the value is missing or unparseable.

    Date-only inputs are anchored at 00:00 UTC of that day.
    """
    if value is None or isinstance(value, bool):
        return None, None
    try:
        if isinstance(value, datetime):
            return _fmt(value), "datetime"
        if isinstance(value, (int, float)):
            return _fmt(datetime.fromtimestamp(value, tz=timezone.utc)), "datetime"
        if not isinstance(value, str):
            return None, None

        text = value.strip()
        if _YYYYMMDD.match(text):
            return _fmt(datetime.strptime(text, "%Y%m%d")), "date"
        if _DATE_ONLY.match(text):
            return _fmt(datetime.strptime(text, "%Y-%m-%d")), "date"
        if text and not text.isdigit():
            return _fmt(datetime.fromisoformat(text.replace("Z", "+00:00"))), "datetime"
    except (ValueError, OverflowError, OSError):
        pass
    return None, None
