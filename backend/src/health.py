"""
Deep health for the zivalx-ops watchdog (``GET /health/deep``).

    {"status": "ok" | "degraded" | "down", "checks": {name: {"ok": bool, "detail": str}}}

The watchdog polls this over loopback every few minutes. ``down`` pages after one run and
``degraded`` after about 30 minutes, so only the database is ``down``; the feeder checks
are ``degraded`` and tolerate a missed 6-hourly tick.

Checks:
  database    SELECT 1 works.
  feeder      a feeder cycle finished within 2 x interval + 30 min (or the process only just
              started, the feeder is switched off, or no finance source is configured).
  extraction  the latest finished feeder cycle did not lose every new item to LLM errors
              (expired key, no credit).
  push        no mention has been waiting for trender longer than the feeder limit
              (wrong or unset token, trender down).
Details never include error text or other data: the route is reachable without credentials.
"""
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from sqlalchemy import func, select, text

from .config import FeederSettings, get_settings

STARTED_AT = datetime.now(timezone.utc)
SLACK = timedelta(minutes=30)
# Matches scheduler_service.BACKLOG_MAX_AGE_DAYS: older rows were given up on.
BACKLOG_MAX_AGE = timedelta(days=14)


def _aware(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _check(ok: bool, detail: str) -> Dict[str, Any]:
    return {"ok": ok, "detail": detail}


def _hours(delta: timedelta) -> str:
    return f"{delta.total_seconds() / 3600:.1f} h"


async def deep_health(
    session_factory=None,
    settings: Optional[FeederSettings] = None,
    now: Optional[datetime] = None,
    started_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    if session_factory is None:
        from .database import AsyncSessionLocal
        session_factory = AsyncSessionLocal
    settings = settings or get_settings()
    now = now or datetime.now(timezone.utc)
    started_at = started_at or STARTED_AT

    checks: Dict[str, Any] = {}
    try:
        async with session_factory() as db:
            await db.execute(text("SELECT 1"))
            checks["database"] = _check(True, "connected")
            checks["feeder"] = await _feeder_check(db, settings, now, started_at)
            checks["extraction"] = await _extraction_check(db)
            checks["push"] = await _push_check(db, settings, now)
    except Exception:  # noqa: BLE001 - report, never raise from a health route
        checks["database"] = _check(False, "unreachable")
        for name in ("feeder", "extraction", "push"):
            checks.setdefault(name, _check(True, "skipped, database down"))
        return {"status": "down", "checks": checks}

    status = "ok" if all(c["ok"] for c in checks.values()) else "degraded"
    return {"status": status, "checks": checks}


def status_code_for(result: Dict[str, Any]) -> int:
    return 503 if result["status"] == "down" else 200


def _limit(settings: FeederSettings) -> timedelta:
    return timedelta(seconds=2 * settings.feeder_interval_seconds) + SLACK


async def _feeder_check(db, settings: FeederSettings, now: datetime, started_at: datetime):
    from .models import Cycle, SourceConfig
    from .models.cycle import CycleStatus

    if not settings.feeder_enabled:
        return _check(True, "feeder disabled")

    sources = (await db.execute(
        select(func.count()).select_from(SourceConfig).where(
            SourceConfig.enabled == True,  # noqa: E712
            SourceConfig.purpose == "finance",
        )
    )).scalar_one()
    if not sources:
        return _check(True, "no finance sources configured")

    last = (await db.execute(
        select(func.max(Cycle.completed_at)).where(
            Cycle.status == CycleStatus.COMPLETED,
            Cycle.name.like("finance-feeder:%"),
        )
    )).scalar_one()
    last = _aware(last)

    # A restart gets a full limit to finish its first cycle, however old the last one is.
    reference = max(filter(None, [last, _aware(started_at)]))
    age = now - reference
    limit = _limit(settings)
    if age > limit:
        return _check(False, f"no finished feeder cycle for {_hours(age)} (limit {_hours(limit)})")
    if last is None:
        return _check(True, "starting, first cycle not finished yet")
    return _check(True, f"last cycle finished {_hours(now - last)} ago")


async def _extraction_check(db):
    from .models import Cycle
    from .models.cycle import CycleStatus

    row = (await db.execute(
        select(Cycle.config_snapshot)
        .where(Cycle.status == CycleStatus.COMPLETED, Cycle.name.like("finance-feeder:%"))
        .order_by(Cycle.completed_at.desc())
        .limit(1)
    )).scalar_one_or_none()
    stats = (row or {}).get("stats") if isinstance(row, dict) else None
    if not stats or not stats.get("new"):
        return _check(True, "nothing to extract in the last cycle")
    failed, new = stats.get("extraction_failed", 0), stats["new"]
    if failed >= new:
        return _check(False, f"the last cycle lost all {new} new item(s) to LLM errors")
    return _check(True, f"{failed} of {new} item(s) failed in the last cycle")


async def _push_check(db, settings: FeederSettings, now: datetime):
    from .models import ExtractedMention
    from .services.scheduler_service import not_permanently_refused

    waiting_since = now - _limit(settings)
    given_up = now - BACKLOG_MAX_AGE
    count = (await db.execute(
        select(func.count()).select_from(ExtractedMention).where(
            ExtractedMention.pushed_to_trender == False,  # noqa: E712
            ExtractedMention.created_at < waiting_since,
            ExtractedMention.created_at >= given_up,
            not_permanently_refused(),
        )
    )).scalar_one()
    if count:
        return _check(False, f"{count} mention(s) have been waiting for trender for over {_hours(_limit(settings))}")
    return _check(True, "no mentions waiting for trender")
