"""
Scheduler Service - the channel-intelligence feeder loop.

On a single global interval (env ``FEEDER_INTERVAL_SECONDS``, default 6h), for
each enabled ``SourceConfig`` with ``purpose == "finance"`` it runs a Cycle:

    collecting -> (new-item filter) -> extracting -> pushing -> completed/failed

A processed-item ledger (prior ``CollectedData``) means each run only extracts
NEW videos/messages, so LLM cost is bounded and reruns are cheap. Mentions are
deduped by ``dedupe_key`` before insert and gated by ``pushed_to_trender`` before
send, so retries never double-count.

Wiring into app startup is guarded by ``FEEDER_ENABLED`` (default false).
"""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from ..config import get_settings
from ..database import AsyncSessionLocal
from ..models import Cycle, CollectedData, ExtractedMention, SourceConfig
from ..models.cycle import CycleStatus
from .collection_orchestrator import CollectionOrchestrator
from .extraction_service import ExtractionService
from .trender_client import TrenderClient

logger = logging.getLogger(__name__)


def _iso(value: Any) -> Any:
    """Recursively coerce datetimes to ISO strings for JSON storage."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _iso(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_iso(v) for v in value]
    return value


class SchedulerService:
    """APScheduler-driven finance feeder. Dependencies are injectable for tests."""

    def __init__(
        self,
        session_factory=None,
        orchestrator: Optional[CollectionOrchestrator] = None,
        extraction_service: Optional[ExtractionService] = None,
        trender_client: Optional[TrenderClient] = None,
        interval_seconds: Optional[int] = None,
        timeframe_days: int = 7,
    ):
        self.session_factory = session_factory or AsyncSessionLocal
        self.orchestrator = orchestrator or CollectionOrchestrator()
        self._extraction_service = extraction_service
        self._trender_client = trender_client
        self.interval_seconds = interval_seconds or get_settings().feeder_interval_seconds
        self.timeframe_days = timeframe_days
        self.scheduler = None

    # ------------------------------------------------------------------ #
    # Lazily-built dependencies (avoid needing API keys at import time)
    # ------------------------------------------------------------------ #
    @property
    def extraction_service(self) -> ExtractionService:
        if self._extraction_service is None:
            self._extraction_service = ExtractionService()
        return self._extraction_service

    @property
    def trender_client(self) -> TrenderClient:
        if self._trender_client is None:
            self._trender_client = TrenderClient()
        return self._trender_client

    # ------------------------------------------------------------------ #
    # APScheduler lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start the interval job. Called from app startup when FEEDER_ENABLED."""
        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        self.scheduler = AsyncIOScheduler()
        self.scheduler.add_job(
            self.run_all,
            trigger="interval",
            seconds=self.interval_seconds,
            id="finance_feeder",
            replace_existing=True,
            next_run_time=datetime.now(timezone.utc),
        )
        self.scheduler.start()
        logger.info("Finance feeder scheduler started (interval=%ss)", self.interval_seconds)

    def shutdown(self) -> None:
        if self.scheduler is not None:
            self.scheduler.shutdown(wait=False)
            self.scheduler = None
            logger.info("Finance feeder scheduler stopped")

    # ------------------------------------------------------------------ #
    # Run logic
    # ------------------------------------------------------------------ #
    async def run_all(self) -> None:
        """Run one feeder cycle for every enabled finance SourceConfig."""
        async with self.session_factory() as db:
            result = await db.execute(
                select(SourceConfig).where(
                    SourceConfig.enabled == True,  # noqa: E712
                    SourceConfig.purpose == "finance",
                )
            )
            configs = result.scalars().all()

        logger.info("Feeder tick: %d finance source config(s)", len(configs))
        for config in configs:
            try:
                await self.run_finance_cycle(config)
            except Exception as e:  # noqa: BLE001 - one source must not kill the tick
                logger.error("Finance cycle failed for config %s: %s", config.name, e, exc_info=True)

    async def run_finance_cycle(self, source_config: SourceConfig) -> Cycle:
        """
        Run a full feeder cycle for one finance source config.

        Returns the persisted Cycle. Uses its own DB session so it is safe to run
        concurrently / from the scheduler.
        """
        async with self.session_factory() as db:
            cycle = Cycle(
                name=f"finance-feeder:{source_config.name}",
                status=CycleStatus.COLLECTING,
                config_snapshot={
                    "source_type": source_config.source_type,
                    "collect_spec": source_config.collect_spec,
                    "purpose": "finance",
                },
                started_at=datetime.now(timezone.utc),
            )
            db.add(cycle)
            await db.commit()
            await db.refresh(cycle)

            try:
                # 1. Collect from the source.
                result = await self.orchestrator.collect(
                    source_type=source_config.source_type,
                    credential_ref=source_config.credential_ref,
                    collect_spec=source_config.collect_spec,
                    timeframe_days=self.timeframe_days,
                )
                raw_items = result.get("data", []) or []

                # 2. New-item filter against the processed-item ledger.
                seen = await self._seen_ids(db, source_config.source_type, exclude_cycle_id=cycle.id)
                new_items = [
                    it for it in raw_items
                    if self._item_id(it, source_config.source_type) not in seen
                ]
                logger.info(
                    "Cycle %s: %d collected, %d new after ledger filter",
                    cycle.id, len(raw_items), len(new_items),
                )

                # Persist the raw collection so future runs see these ids.
                db.add(CollectedData(
                    cycle_id=cycle.id,
                    source_type=source_config.source_type,
                    source_name=self._source_name(result),
                    data=_iso(raw_items),
                    data_size_bytes=len(str(raw_items)),
                    item_count=len(raw_items),
                    collection_time_ms=result.get("metadata", {}).get("collection_time_ms"),
                ))
                await db.commit()

                # 3. Extract picks from new items.
                cycle.status = CycleStatus.SUMMARIZING  # reused lifecycle state = "extracting/pushing"
                await db.commit()

                extraction_items = [
                    self._to_extraction_item(it, source_config.source_type) for it in new_items
                ]
                mentions = await self.extraction_service.extract(extraction_items)

                # 4. Build ExtractedMention rows (dedupe before insert).
                stored = await self._store_mentions(db, cycle.id, mentions)

                # 5. Push to trender, then flag pushed rows.
                await self._push(db, stored)

                cycle.status = CycleStatus.COMPLETED
                cycle.completed_at = datetime.now(timezone.utc)
                await db.commit()
                logger.info("Cycle %s completed: %d mention(s) stored", cycle.id, len(stored))

            except Exception as e:  # noqa: BLE001
                await db.rollback()
                fresh = await db.execute(select(Cycle).where(Cycle.id == cycle.id))
                cycle = fresh.scalar_one()
                cycle.status = CycleStatus.FAILED
                cycle.error_message = str(e)
                cycle.completed_at = datetime.now(timezone.utc)
                await db.commit()
                logger.error("Cycle %s failed: %s", cycle.id, e, exc_info=True)
                raise

            await db.refresh(cycle)
            return cycle

    # ------------------------------------------------------------------ #
    # Ledger + mapping helpers
    # ------------------------------------------------------------------ #
    async def _seen_ids(self, db, source_type: str, exclude_cycle_id=None) -> set:
        """Collect all previously-seen item ids for this source from CollectedData."""
        query = select(CollectedData).where(CollectedData.source_type == source_type)
        if exclude_cycle_id is not None:
            query = query.where(CollectedData.cycle_id != exclude_cycle_id)
        rows = (await db.execute(query)).scalars().all()

        seen: set = set()
        for row in rows:
            data = row.data or []
            if not isinstance(data, list):
                continue
            for item in data:
                item_id = self._item_id(item, source_type)
                if item_id is not None:
                    seen.add(item_id)
        return seen

    @staticmethod
    def _item_id(item: Dict[str, Any], source_type: str) -> Optional[str]:
        """Stable per-item id: youtube video_id, telegram message_id."""
        if not isinstance(item, dict):
            return None
        if source_type == "telegram":
            mid = item.get("message_id") or item.get("id")
            return str(mid) if mid is not None else None
        # youtube (and default)
        vid = item.get("video_id") or item.get("id")
        return str(vid) if vid is not None else None

    @staticmethod
    def _source_name(result: Dict[str, Any]) -> str:
        channels = (result.get("source_info", {}) or {}).get("channels")
        if isinstance(channels, list):
            return ", ".join(str(c) for c in channels)
        return str(channels) if channels else "unknown"

    @staticmethod
    def _to_extraction_item(item: Dict[str, Any], source_type: str) -> Dict[str, Any]:
        """Map a raw connector item to the ExtractionService input shape."""
        if source_type == "telegram":
            return {
                "source": "telegram",
                "transcript": item.get("text"),
                "message_link": item.get("message_link") or item.get("link"),
                "channel": item.get("channel"),
                "ts": item.get("date") or item.get("ts"),
            }
        # youtube (and default)
        return {
            "source": "youtube",
            "transcript": item.get("transcript"),
            "transcript_segments": item.get("transcript_segments"),
            "video_id": item.get("video_id") or item.get("id"),
            "url": item.get("url"),
            "channel": item.get("channel"),
            "title": item.get("title"),
            "ts": item.get("published_at") or item.get("upload_date"),
        }

    async def _store_mentions(self, db, cycle_id, mentions: List[Dict[str, Any]]):
        """
        Insert mentions, skipping any dedupe_key already present in the DB.

        Returns a list of (row, mention_dict) pairs so the push step can send the
        exact frozen-contract payload (preserving ``mentioned_at`` etc.).
        """
        existing = set(
            (await db.execute(select(ExtractedMention.dedupe_key))).scalars().all()
        )
        stored = []
        batch_keys: set = set()
        for mention in mentions:
            key = ExtractionService.dedupe_key_for(mention)
            if key in existing or key in batch_keys:
                logger.debug("Skipping duplicate mention dedupe_key=%s", key)
                continue
            batch_keys.add(key)
            row = ExtractedMention(
                cycle_id=cycle_id,
                symbol=mention["symbol"],
                company_name=mention.get("company_name"),
                thesis=mention.get("context"),
                stance=mention.get("stance"),
                sentiment_score=mention.get("sentiment_score"),
                confidence=mention.get("confidence"),
                citation_json=mention.get("citation", {}) or {},
                dedupe_key=key,
                pushed_to_trender=False,
            )
            db.add(row)
            stored.append((row, mention))
        await db.commit()
        for row, _ in stored:
            await db.refresh(row)
        return stored

    async def _push(self, db, stored) -> None:
        """Push not-yet-pushed mentions to trender and record the response."""
        to_push = [(row, mention) for row, mention in stored if not row.pushed_to_trender]
        if not to_push:
            return

        payload = [mention for _, mention in to_push]
        try:
            response = await self.trender_client.push_mentions(payload)
        except Exception as e:  # noqa: BLE001 - record failure, leave pushed=False
            logger.error("trender push failed for %d mention(s): %s", len(to_push), e)
            for row, _ in to_push:
                row.trender_response = f"error: {e}"
            await db.commit()
            return

        response_text = str(response)
        for row, _ in to_push:
            row.pushed_to_trender = True
            row.trender_response = response_text
        await db.commit()
