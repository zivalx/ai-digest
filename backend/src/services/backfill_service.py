"""
Backfill Service - one-off track-record extraction over a channel's back catalog.

The scheduled feeder only sees NEW items. This collects a long window, filters
out items already processed for the CURRENT ``EXTRACTION_VERSION`` (ledger:
``track_record_processed``), prints an LLM cost estimate, asks for confirmation,
extracts, then pushes mentions to discovery and calls to the track record with
``origin="backfill"``. Bumping the extraction version makes it re-runnable.

Mapping/storing/pushing is reused from ``SchedulerService``.
"""
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, List, Tuple

from sqlalchemy import select

from ..models import Cycle, SourceConfig, TrackRecordProcessed
from ..models.cycle import CycleStatus
from .extraction_service import EXTRACTION_VERSION
from .scheduler_service import SchedulerService

logger = logging.getLogger(__name__)


@dataclass
class BackfillReport:
    items_found: int
    items_new: int
    est_input_tokens: int
    est_cost_usd: float
    calls_pushed: int
    dry_run: bool


def estimate_cost(items: List[Dict[str, Any]], usd_per_mtok: float) -> Tuple[int, float]:
    """Rough input-cost estimate: tokens = total transcript chars // 4."""
    chars = sum(len(it.get("transcript") or it.get("text") or "") for it in items)
    tokens = chars // 4
    return tokens, tokens / 1_000_000 * usd_per_mtok


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
            for it in items:
                item_id = SchedulerService._item_id(it, source_type)
                if item_id is not None and item_id not in done:
                    new_items.append(it)

            tokens, cost = estimate_cost(new_items, self.usd_per_mtok)
            report = BackfillReport(len(items), len(new_items), tokens, cost, 0, dry_run)
            summary = (
                f"{len(items)} item(s) found, {len(new_items)} new for extraction "
                f"{EXTRACTION_VERSION}; est. {tokens} input tokens (~${cost:.2f})"
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

            mentions = await self.extraction_service.extract(
                [self._feeder._to_extraction_item(it, source_type) for it in new_items]
            )
            stored = await self._feeder._store_mentions(db, cycle.id, mentions)
            await self._feeder._push(db, stored)
            await self._feeder._push_calls(db, stored, origin="backfill")
            report.calls_pushed = sum(1 for row, _ in stored if row.pushed_calls_to_trender)

            for it in new_items:
                db.add(TrackRecordProcessed(
                    source_type=source_type,
                    item_id=SchedulerService._item_id(it, source_type),
                    extraction_version=EXTRACTION_VERSION,
                ))
            cycle.status = CycleStatus.COMPLETED
            cycle.completed_at = datetime.now(timezone.utc)
            await db.commit()
            return report
