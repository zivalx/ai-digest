"""
Backfill Service - one-off track-record extraction over a channel's back catalog.

The scheduled feeder only sees NEW items. This collects a long window, filters
out items already processed for the CURRENT ``EXTRACTION_VERSION`` (ledger:
``track_record_processed``), prints an LLM cost estimate, asks for confirmation,
extracts, then pushes mentions to discovery and calls to the track record with
``origin="backfill"``. Bumping the extraction version makes it re-runnable.

Items whose collection failed, that have no transcript/text, or no publish time
are reported as unavailable and never ledgered, so a later run retries them.
``since`` is applied here on the normalized publish time (the collectors'
channel listing does not filter by date itself).

Calls are pushed for every extracted recommendation, including picks whose
mention row already existed; unpushed calls can be re-sent with ``repush()``.
Mapping/storing/pushing is reused from ``SchedulerService``.
"""
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, List, Tuple

from sqlalchemy import select

from ..models import Cycle, SourceConfig, TrackRecordProcessed
from ..models.cycle import CycleStatus
from .extraction_service import EXTRACTION_VERSION
from .scheduler_service import CallPushResult, SchedulerService
from .timestamps import normalize_published

logger = logging.getLogger(__name__)


@dataclass
class BackfillReport:
    items_found: int
    items_new: int
    est_input_tokens: int
    est_cost_usd: float
    calls_pushed: int  # accepted + duplicate (flagged pushed)
    dry_run: bool
    mentions_failed: int = 0
    calls_failed: int = 0  # transport / HTTP failures
    calls_accepted: int = 0
    calls_duplicate: int = 0
    calls_rejected: Dict[str, int] = field(default_factory=dict)  # reason -> count
    items: List[Dict[str, Any]] = field(default_factory=list)
    unavailable: List[Dict[str, Any]] = field(default_factory=list)  # id, title, reason


def estimate_cost(items: List[Dict[str, Any]], usd_per_mtok: float) -> Tuple[int, float]:
    """Rough input-cost estimate: tokens = total transcript chars // 4."""
    chars = sum(len(it.get("transcript") or it.get("text") or "") for it in items)
    tokens = chars // 4
    return tokens, tokens / 1_000_000 * usd_per_mtok


def _published(item: Dict[str, Any]) -> Any:
    return item.get("published_at") or item.get("upload_date") or item.get("date")


def _unavailable_reason(item: Dict[str, Any], source_type: str, since: date) -> Tuple[str, bool]:
    """
    ``(reason, skip)`` for an item: reason is non-empty if it is unavailable
    (reported, retried next run); ``skip`` is True if it is simply older than
    ``since``.
    """
    if item.get("status") == "failed":
        return f"collection failed: {item.get('error') or 'unknown error'}", False
    if not (item.get("transcript") or item.get("text") or "").strip():
        return ("no text" if source_type == "telegram" else "no transcript"), False
    ts, _ = normalize_published(_published(item))
    if ts is None:
        return "no publish time", False
    return "", ts[:10] < since.isoformat()


class BackfillService:
    def __init__(self, orchestrator, extraction_service, trender_client, session_factory,
                 usd_per_mtok: float = 3.0):
        self.orchestrator = orchestrator
        self.extraction_service = extraction_service
        self.trender_client = trender_client
        self.session_factory = session_factory
        self.usd_per_mtok = usd_per_mtok
        # Reuse the feeder's mapping / store / push helpers.
        self._feeder = SchedulerService(
            session_factory=session_factory,
            orchestrator=orchestrator,
            extraction_service=extraction_service,
            trender_client=trender_client,
            interval_seconds=1,
        )

    async def run(
        self,
        source_config_name: str,
        since: date,
        limit: int,
        dry_run: bool,
        confirm: Callable[[str], bool],
    ) -> BackfillReport:
        async with self.session_factory() as db:
            config = (await db.execute(
                select(SourceConfig).where(SourceConfig.name == source_config_name)
            )).scalars().first()
            if config is None:
                raise ValueError(f"No source config named {source_config_name!r}")
            source_type = config.source_type

            # Explicit overrides: the orchestrator only setdefault()s the timeframe.
            days_back = max((date.today() - since).days, 1)
            spec = dict(config.collect_spec or {})
            spec["days_back"] = days_back
            spec["max_videos_per_channel"] = limit
            spec["max_messages"] = limit  # telegram's equivalent (no date filter there)
            result = await self.orchestrator.collect(
                source_type=source_type,
                credential_ref=config.credential_ref,
                collect_spec=spec,
                timeframe_days=days_back,
            )
            items = result.get("data", []) or []

            done = set((await db.execute(
                select(TrackRecordProcessed.item_id).where(
                    TrackRecordProcessed.source_type == source_type,
                    TrackRecordProcessed.extraction_version == EXTRACTION_VERSION,
                )
            )).scalars().all())
            new_items = []
            unavailable = []
            queued: set = set()
            for it in items:
                item_id = SchedulerService._item_id(it, source_type)
                if item_id is None or item_id in done or item_id in queued:
                    continue
                queued.add(item_id)
                reason, too_old = _unavailable_reason(it, source_type, since)
                if reason:
                    unavailable.append({"id": item_id, "title": it.get("title"), "reason": reason})
                elif not too_old:
                    new_items.append(it)

            tokens, cost = estimate_cost(new_items, self.usd_per_mtok)
            listing = [
                {
                    "id": SchedulerService._item_id(it, source_type),
                    "title": it.get("title"),
                    "published": _published(it),
                    "transcript_chars": len(it.get("transcript") or it.get("text") or ""),
                }
                for it in new_items
            ]
            report = BackfillReport(
                len(items), len(new_items), tokens, cost, 0, dry_run,
                items=listing, unavailable=unavailable,
            )
            summary = (
                f"{len(items)} item(s) found, {len(new_items)} new for extraction "
                f"{EXTRACTION_VERSION} ({len(unavailable)} unavailable); "
                f"est. {tokens} input tokens (~${cost:.2f})"
            )
            logger.info("Backfill %s: %s", source_config_name, summary)
            if dry_run or not new_items:
                return report
            if not confirm(f"{summary}. Proceed?"):
                logger.info("Backfill declined; nothing extracted")
                return report

            cycle = Cycle(
                name=f"backfill:{source_config_name}",
                status=CycleStatus.SUMMARIZING,
                config_snapshot={"source_type": source_type, "collect_spec": spec,
                                 "purpose": "backfill"},
                started_at=datetime.now(timezone.utc),
            )
            db.add(cycle)
            await db.commit()
            await db.refresh(cycle)

            cycle_id = cycle.id
            try:
                ext_items = [self._feeder._to_extraction_item(it, source_type) for it in new_items]
                raw_by_ext = {id(e): it for e, it in zip(ext_items, new_items)}
                failed_ids: set = set()

                def on_error(ext_item, exc):
                    raw = raw_by_ext.get(id(ext_item))
                    if raw is not None:
                        failed_ids.add(SchedulerService._item_id(raw, source_type))

                mentions = await self.extraction_service.extract(ext_items, on_error=on_error)
                stored = await self._feeder._store_mentions(db, cycle.id, mentions)
                await self._feeder._push(db, stored)
                report.mentions_failed = sum(1 for row, _ in stored if not row.pushed_to_trender)
                # Calls for ALL extracted recommendations, incl. pre-existing rows.
                pending = await self._feeder._attach_calls(db, mentions, origin="backfill")
                self._apply_calls(report, await self._feeder._push_calls(db, pending))

                # Failed items get no ledger row so the next run retries them.
                for listed in listing:
                    if listed["id"] in failed_ids:
                        continue
                    db.add(TrackRecordProcessed(
                        source_type=source_type,
                        item_id=listed["id"],
                        extraction_version=EXTRACTION_VERSION,
                    ))
                cycle.status = CycleStatus.COMPLETED
                cycle.completed_at = datetime.now(timezone.utc)
                await db.commit()
            except Exception as e:  # noqa: BLE001
                await db.rollback()
                fresh = await db.execute(select(Cycle).where(Cycle.id == cycle_id))
                cycle = fresh.scalar_one()
                cycle.status = CycleStatus.FAILED
                cycle.error_message = str(e)
                cycle.completed_at = datetime.now(timezone.utc)
                await db.commit()
                logger.error("Backfill cycle %s failed: %s", cycle_id, e, exc_info=True)
                raise
            return report

    async def repush(self) -> CallPushResult:
        """Re-send stored, not-yet-accepted calls only: no collection, no LLM."""
        async with self.session_factory() as db:
            return await self._feeder.repush_unpushed_calls(db)

    @staticmethod
    def _apply_calls(report: BackfillReport, calls: CallPushResult) -> None:
        report.calls_pushed = calls.pushed
        report.calls_accepted = calls.accepted
        report.calls_duplicate = calls.duplicate
        report.calls_rejected = dict(calls.rejected)
        report.calls_failed = calls.failed
