"""
One-shot backfill for the finance feeder.

    docker exec ai-digest-app python -m src.backfill --days 30
    docker exec ai-digest-app python -m src.backfill --days 30 --source "Finance YT"
    docker exec ai-digest-app python -m src.backfill --days 30 --dry-run

The regular feeder looks at the newest few items of each source every 6 hours. After a gap
(the box was down, or influencer content was not being captured yet) this runs the same
cycle once with a longer look-back, for every enabled ``finance`` source (or one, with
``--source``), and prints a summary.

What it does and does not do:

* It reuses the feeder cycle, so the processed-item ledger applies: anything already seen
  is skipped, an item is only recorded after its mentions are stored, and mentions are
  pushed to trender in batches of 50 (earlier failed pushes are retried too). Running it
  again is safe and finishes whatever was left.
* Items older than the window, or without a publish date, are dropped before extraction
  and are NOT recorded as seen, so a later, longer backfill still picks them up.
* The look-back is capped (``MAX_BACKFILL_DAYS``) and so is the item count per channel
  (``--max-items``). The YouTube collector takes the newest N videos of a channel and does
  not enforce a day window itself, so the real reach is N videos per channel: with the
  default 40 a channel that posts daily is covered for about 40 days, one that posts 5 a
  day for about 8. Raise ``--max-items`` (up to 200) for busy channels.
* trender rejects a ``mentioned_at`` before the year 2000 or more than a day ahead.
  Publish dates come from the collectors and are normalised to ISO UTC, so this does not
  happen with real data; trender's answer is counted as "rejected" in the summary.

Exit code: 0 done, 1 a source failed, 2 bad arguments, 3 finished but some items were not
processed or some mentions were not pushed (run the same command again).

LLM calls cost money: use ``--dry-run`` first to see how many items would be extracted.
Memory: this process runs inside the app container and counts toward its memory limit.
"""
import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select

MAX_BACKFILL_DAYS = 60
MAX_ITEMS_CAP = 200
DEFAULT_MAX_ITEMS = {"youtube": 40, "telegram": 200}
FALLBACK_MAX_ITEMS = 40
# A cycle that started less than this long ago and has not finished blocks the backfill.
BUSY_WINDOW = timedelta(hours=3)
PREVIEW_LIMIT = 15

logger = logging.getLogger("backfill")


class BackfillError(Exception):
    """A problem the operator can fix (unknown source, nothing to do, a cycle running)."""


@dataclass
class SourceResult:
    name: str
    source_type: str
    ok: bool = True
    error: Optional[str] = None
    stats: Dict[str, int] = field(default_factory=dict)
    preview: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BackfillReport:
    days: int
    max_items: Optional[int]
    window_start: datetime
    dry_run: bool = False
    sources: List[SourceResult] = field(default_factory=list)

    @property
    def totals(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for src in self.sources:
            for key, value in src.stats.items():
                out[key] = out.get(key, 0) + value
        return out

    def exit_code(self) -> int:
        if any(not s.ok for s in self.sources):
            return 1
        totals = self.totals
        if not self.dry_run and (totals.get("push_failed", 0) or totals.get("extraction_failed", 0)):
            return 3
        return 0


# ------------------------------------------------------------------ arguments

def _days(value: str) -> int:
    try:
        days = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number of days")
    if days < 1:
        raise argparse.ArgumentTypeError("days must be at least 1")
    if days > MAX_BACKFILL_DAYS:
        raise argparse.ArgumentTypeError(f"days is capped at {MAX_BACKFILL_DAYS}")
    return days


def _max_items(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number")
    if n < 1 or n > MAX_ITEMS_CAP:
        raise argparse.ArgumentTypeError(f"max-items must be between 1 and {MAX_ITEMS_CAP}")
    return n


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.backfill",
        description="Run the finance feeder once with a longer look-back (see module docstring).",
    )
    parser.add_argument("--days", type=_days, required=True,
                        help=f"look-back in days, 1-{MAX_BACKFILL_DAYS}")
    parser.add_argument("--source", metavar="NAME",
                        help="only this source config (name as shown on the Sources page)")
    parser.add_argument("--max-items", type=_max_items, metavar="N",
                        help=f"items per channel to collect, up to {MAX_ITEMS_CAP} "
                             f"(default: {DEFAULT_MAX_ITEMS['youtube']} videos, "
                             f"{DEFAULT_MAX_ITEMS['telegram']} telegram messages)")
    parser.add_argument("--dry-run", action="store_true",
                        help="collect and filter, but extract, write and push nothing")
    parser.add_argument("--force", action="store_true",
                        help="run even if a feeder cycle looks like it is still running")
    return parser.parse_args(argv)


def window_start(now: datetime, days: int) -> datetime:
    """Midnight UTC, ``days`` days before ``now`` (video dates have day precision)."""
    return (now - timedelta(days=days)).astimezone(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0)


def spec_overrides_for(source_type: str, days: int, max_items: Optional[int]) -> Dict[str, Any]:
    """Collector settings for the wider look-back; they win over the stored collect_spec."""
    if source_type == "youtube":
        return {"days_back": days,
                "max_videos_per_channel": max_items or DEFAULT_MAX_ITEMS["youtube"]}
    if source_type == "telegram":
        return {"max_messages": max_items or DEFAULT_MAX_ITEMS["telegram"]}
    return {}


# ------------------------------------------------------------------ the run

async def _finance_configs(session_factory, source: Optional[str]):
    from .models import SourceConfig

    async with session_factory() as db:
        rows = (await db.execute(
            select(SourceConfig).where(
                SourceConfig.enabled == True,  # noqa: E712
                SourceConfig.purpose == "finance",
            ).order_by(SourceConfig.name)
        )).scalars().all()

    if not rows:
        raise BackfillError("no enabled finance source configs found (Sources page, purpose 'finance')")
    if source is None:
        return rows
    matches = [c for c in rows if c.name.strip().lower() == source.strip().lower()]
    if not matches:
        names = ", ".join(sorted(c.name for c in rows))
        raise BackfillError(f"no enabled finance source named {source!r}; available: {names}")
    return matches


async def _check_not_busy(session_factory, now: datetime) -> None:
    from .models import Cycle
    from .models.cycle import CycleStatus

    async with session_factory() as db:
        rows = (await db.execute(
            select(Cycle).where(Cycle.status.in_([CycleStatus.COLLECTING, CycleStatus.SUMMARIZING]))
        )).scalars().all()
    for cycle in rows:
        started = cycle.started_at
        if started is None:
            continue
        started = started if started.tzinfo else started.replace(tzinfo=timezone.utc)
        if now - started < BUSY_WINDOW:
            raise BackfillError(
                f"cycle {cycle.name!r} started {started:%Y-%m-%d %H:%M} UTC has not finished. "
                f"Wait for it, or pass --force if it is stuck"
            )


async def run_backfill(
    *,
    days: int,
    source: Optional[str] = None,
    max_items: Optional[int] = None,
    dry_run: bool = False,
    force: bool = False,
    scheduler=None,
    session_factory=None,
    now: Optional[datetime] = None,
) -> BackfillReport:
    """Run the backfill; ``scheduler`` and ``session_factory`` are injectable for tests."""
    if not 1 <= days <= MAX_BACKFILL_DAYS:
        raise BackfillError(f"days must be between 1 and {MAX_BACKFILL_DAYS}")
    if max_items is not None and not 1 <= max_items <= MAX_ITEMS_CAP:
        raise BackfillError(f"max-items must be between 1 and {MAX_ITEMS_CAP}")

    if scheduler is None:
        from .services.scheduler_service import SchedulerService
        scheduler = SchedulerService()
    session_factory = session_factory or scheduler.session_factory
    now = now or datetime.now(timezone.utc)
    cutoff = window_start(now, days)

    configs = await _finance_configs(session_factory, source)
    if not dry_run and not force:
        await _check_not_busy(session_factory, now)

    report = BackfillReport(days=days, max_items=max_items, window_start=cutoff, dry_run=dry_run)
    for config in configs:
        result = SourceResult(name=config.name, source_type=config.source_type)
        report.sources.append(result)
        overrides = spec_overrides_for(config.source_type, days, max_items)
        try:
            if dry_run:
                result.stats, new_items = await scheduler.dry_run(
                    config, timeframe_days=days, spec_overrides=overrides, cutoff=cutoff)
                result.preview = new_items[:PREVIEW_LIMIT]
            else:
                _, result.stats = await scheduler.run_cycle_with_stats(
                    config, timeframe_days=days, spec_overrides=overrides, cutoff=cutoff,
                    cycle_prefix="finance-backfill")
        except Exception as exc:  # noqa: BLE001 - report it, carry on with the next source
            logger.error("Backfill of %s failed: %s", config.name, exc, exc_info=True)
            result.ok = False
            result.error = f"{type(exc).__name__}: {exc}"
    return report


# ------------------------------------------------------------------ output

def format_report(report: BackfillReport) -> str:
    mode = "DRY RUN (nothing extracted, stored or pushed)" if report.dry_run else "backfill"
    lines = [
        f"{mode}: {report.days} days, from {report.window_start:%Y-%m-%d} 00:00 UTC "
        f"(cap {MAX_BACKFILL_DAYS} days)",
    ]
    for src in report.sources:
        lines.append(f"== {src.name} ({src.source_type}) ==")
        if not src.ok:
            lines.append(f"  FAILED: {src.error}")
            continue
        s = src.stats
        lines.append(
            f"  collected {s['collected']}, collector failed {s['collector_failed']}, "
            f"outside window or undated {s['out_of_window']}, already seen {s['already_seen']}, "
            f"new {s['new']}"
        )
        if report.dry_run:
            for item in src.preview:
                lines.append(f"    would process: {item.get('video_id') or item.get('message_id')} "
                             f"{(item.get('title') or '')[:70]}")
            if s["new"] > len(src.preview):
                lines.append(f"    ... and {s['new'] - len(src.preview)} more")
            continue
        duplicates = s["mentions_found"] - s["mentions_stored"]
        lines.append(
            f"  extraction failed {s['extraction_failed']}, mentions found {s['mentions_found']}, "
            f"stored {s['mentions_stored']} ({duplicates} duplicates)"
        )
        lines.append(
            f"  pushed {s['pushed']} (accepted {s['accepted']}, rejected {s['rejected']}), "
            f"not pushed {s['push_failed']}"
        )
    if not report.dry_run and report.exit_code() == 3:
        lines.append("Some items or pushes are unfinished (nothing is lost): run the same command "
                     "again, finished items are skipped.")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        report = asyncio.run(run_backfill(
            days=args.days, source=args.source, max_items=args.max_items,
            dry_run=args.dry_run, force=args.force))
    except BackfillError as exc:
        print(f"backfill: {exc}", file=sys.stderr)
        return 1
    print(format_report(report))
    return report.exit_code()


if __name__ == "__main__":
    sys.exit(main())
