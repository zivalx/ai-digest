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
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select

from ..config import get_settings
from ..database import AsyncSessionLocal
from ..models import Cycle, CollectedData, ExtractedMention, SourceConfig
from ..models.cycle import CycleStatus
from .collection_orchestrator import CollectionOrchestrator
from .extraction_service import ExtractionService
from .timestamps import normalize_published
from .trender_client import TrenderClient, TrenderPushError

logger = logging.getLogger(__name__)

# Mentions per request to trender (it re-scores candidates after every request).
PUSH_BATCH_SIZE = 50
# Mentions whose push failed are retried on later cycles, but only this long.
BACKLOG_MAX_AGE_DAYS = 14
BACKLOG_MAX_ROWS = 500
# Dropped from the stored ledger copy of an item: the ledger only needs ids, and a
# year of transcripts would be loaded into memory on every cycle.
_LEDGER_HEAVY_FIELDS = ("transcript", "transcript_segments", "description")


def _iso(value: Any) -> Any:
    """Recursively coerce datetimes to ISO strings for JSON storage."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _iso(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_iso(v) for v in value]
    return value


def _slim(item: Any) -> Any:
    """Ledger copy of a collected item without the bulky text fields."""
    if not isinstance(item, dict):
        return item
    return {k: v for k, v in item.items() if k not in _LEDGER_HEAVY_FIELDS}


def _parse_iso(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 string (trailing Z allowed) to an aware UTC datetime."""
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _chunks(seq: List[Any], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


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
        cycle, _ = await self.run_cycle_with_stats(source_config)
        return cycle

    async def run_cycle_with_stats(
        self,
        source_config: SourceConfig,
        *,
        timeframe_days: Optional[int] = None,
        spec_overrides: Optional[Dict[str, Any]] = None,
        cutoff: Optional[datetime] = None,
        cycle_prefix: str = "finance-feeder",
    ) -> Tuple[Cycle, Dict[str, int]]:
        """
        One feeder cycle, returning ``(cycle, stats)``.

        ``timeframe_days`` / ``spec_overrides`` widen what the collector is asked for (the
        backfill uses them; ``spec_overrides`` win over the stored collect_spec). ``cutoff``
        drops collected items published before it, or with no usable date, *before* they
        are extracted or written to the ledger: the collectors ask for the newest N items
        and do not enforce a day window, so they may return older ones.

        Ledger rule: an item is written to the processed-item ledger only after its
        mentions were stored, and never when the collector failed on it or the LLM call
        for it failed. Those items are picked up again by the next run.
        """
        days = timeframe_days or self.timeframe_days
        collect_spec = {**(source_config.collect_spec or {}), **(spec_overrides or {})}
        stats: Dict[str, int] = {
            "collected": 0, "collector_failed": 0, "out_of_window": 0, "already_seen": 0,
            "new": 0, "extraction_failed": 0, "mentions_found": 0, "mentions_stored": 0,
            "pushed": 0, "accepted": 0, "rejected": 0, "push_failed": 0,
        }

        async with self.session_factory() as db:
            cycle = Cycle(
                name=f"{cycle_prefix}:{source_config.name}",
                status=CycleStatus.COLLECTING,
                config_snapshot={
                    "source_type": source_config.source_type,
                    "collect_spec": collect_spec,
                    "purpose": "finance",
                    "timeframe_days": days,
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
                    collect_spec=collect_spec,
                    timeframe_days=days,
                )
                raw_items = result.get("data", []) or []
                stats["collected"] = len(raw_items)
                source_type = source_config.source_type

                # 2. Drop items the collector failed on, and (backfill) items outside the window.
                usable = []
                for it in raw_items:
                    if isinstance(it, dict) and it.get("status") == "failed":
                        stats["collector_failed"] += 1
                        continue
                    if cutoff is not None and not self._published_on_or_after(it, source_type, cutoff):
                        stats["out_of_window"] += 1
                        continue
                    usable.append(it)

                # 3. New-item filter against the processed-item ledger.
                seen = await self._seen_ids(db, source_type, exclude_cycle_id=cycle.id)
                new_items = [it for it in usable if self._item_id(it, source_type) not in seen]
                stats["already_seen"] = len(usable) - len(new_items)
                stats["new"] = len(new_items)
                logger.info(
                    "Cycle %s: %d collected, %d new after filters",
                    cycle.id, len(raw_items), len(new_items),
                )

                # 4. Extract picks from new items.
                cycle.status = CycleStatus.SUMMARIZING  # reused lifecycle state = "extracting/pushing"
                await db.commit()

                extraction_items = [self._to_extraction_item(it, source_type) for it in new_items]
                mentions = await self.extraction_service.extract(extraction_items)
                stats["mentions_found"] = len(mentions)
                # Positions (in the list handed to extract) whose LLM call failed.
                failed_idx = {
                    i for i in (getattr(self.extraction_service, "last_failed_indexes", None) or [])
                    if 0 <= i < len(new_items)
                }
                stats["extraction_failed"] = len(failed_idx)

                # 5. Build ExtractedMention rows (dedupe before insert).
                stored = await self._store_mentions(db, cycle.id, mentions)
                stats["mentions_stored"] = len(stored)

                # 6. Record the ledger (ids only, after the mentions are safe in the DB).
                ledger = [_slim(it) for i, it in enumerate(new_items) if i not in failed_idx]
                db.add(CollectedData(
                    cycle_id=cycle.id,
                    source_type=source_type,
                    source_name=self._source_name(result),
                    data=_iso(ledger),
                    data_size_bytes=len(str(ledger)),
                    item_count=len(ledger),
                    collection_time_ms=result.get("metadata", {}).get("collection_time_ms"),
                ))
                await db.commit()

                # 7. Push to trender (this cycle's mentions plus any earlier failed pushes).
                await self._push(db, stored, stats)

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
            return cycle, stats

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
        """Stable per-item id: youtube video_id, telegram channel:message_id."""
        if not isinstance(item, dict):
            return None
        if source_type == "telegram":
            mid = item.get("message_id") or item.get("id")
            if mid is None:
                return None
            # Message ids are only unique inside one channel.
            channel = item.get("channel")
            return f"{channel}:{mid}" if channel else str(mid)
        # youtube (and default)
        vid = item.get("video_id") or item.get("id")
        return str(vid) if vid is not None else None

    def _published_on_or_after(self, item: Dict[str, Any], source_type: str, cutoff: datetime) -> bool:
        """True when the item has a usable publish time at or after ``cutoff``."""
        mapped = self._to_extraction_item(item, source_type) if isinstance(item, dict) else {}
        published = _parse_iso(mapped.get("ts"))
        return published is not None and published >= cutoff

    @staticmethod
    def _source_name(result: Dict[str, Any]) -> str:
        channels = (result.get("source_info", {}) or {}).get("channels")
        if isinstance(channels, list):
            return ", ".join(str(c) for c in channels)
        return str(channels) if channels else "unknown"

    @staticmethod
    def _to_extraction_item(item: Dict[str, Any], source_type: str) -> Dict[str, Any]:
        """Map a raw connector item to the ExtractionService input shape.

        ``ts`` is always ISO-8601 UTC (or None) — see ``normalize_published``.
        """
        if source_type == "telegram":
            ts, precision = normalize_published(item.get("date") or item.get("ts"))
            return {
                "source": "telegram",
                "transcript": item.get("text"),
                "message_link": item.get("message_link") or item.get("link"),
                "channel": item.get("channel"),
                "ts": ts,
                "ts_precision": precision,
            }
        # youtube (and default)
        ts, precision = normalize_published(item.get("published_at") or item.get("upload_date"))
        return {
            "source": "youtube",
            "transcript": item.get("transcript"),
            "transcript_segments": item.get("transcript_segments"),
            "video_id": item.get("video_id") or item.get("id"),
            "url": item.get("url"),
            "channel": item.get("channel"),
            "title": item.get("title"),
            "ts": ts,
            "ts_precision": precision,
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
                mentioned_at=_parse_iso(mention.get("mentioned_at")),
                dedupe_key=key,
                pushed_to_trender=False,
            )
            db.add(row)
            stored.append((row, mention))
        await db.commit()
        for row, _ in stored:
            await db.refresh(row)
        return stored

    async def _push(self, db, stored, stats: Optional[Dict[str, int]] = None) -> None:
        """
        Push mentions to trender in batches and record the response on each row.

        Sends this cycle's new mentions plus any earlier ones that never made it (trender
        was down or restarting). A transport failure stops the loop and leaves the rest
        unpushed for the next cycle; an HTTP error for a batch retries that batch one
        mention at a time so a single bad mention cannot block the others.
        """
        stats = stats if stats is not None else {}
        current = [(row, mention) for row, mention in stored if not row.pushed_to_trender]
        backlog = await self._unpushed_backlog(db, exclude_ids={row.id for row, _ in stored})
        pairs = backlog + current
        if not pairs:
            return
        if backlog:
            logger.info("Retrying %d mention(s) that were not pushed earlier", len(backlog))

        for chunk in _chunks(pairs, PUSH_BATCH_SIZE):
            if not await self._push_chunk(db, chunk, stats):
                logger.error("trender unreachable; %d mention(s) stay unpushed", len(pairs))
                break
        stats["push_failed"] = sum(1 for row, _ in pairs if not row.pushed_to_trender)

    async def _push_chunk(self, db, chunk, stats: Dict[str, int]) -> bool:
        """Push one batch. Returns False when trender could not be reached at all."""
        payload = [mention for _, mention in chunk]
        try:
            response = await self.trender_client.push_mentions(payload)
        except Exception as e:  # noqa: BLE001 - record failure, leave pushed=False
            logger.error("trender push failed for %d mention(s): %s", len(chunk), e)
            for row, _ in chunk:
                row.trender_response = f"error: {e}"
            await db.commit()
            status = getattr(e, "status_code", None)
            if isinstance(e, TrenderPushError) and status is not None and len(chunk) > 1:
                for single in chunk:
                    await self._push_chunk(db, [single], stats)
                return True
            return status is not None  # transport failure (no status): stop

        response_text = str(response)
        for row, _ in chunk:
            row.pushed_to_trender = True
            row.trender_response = response_text
        await db.commit()

        stats["pushed"] = stats.get("pushed", 0) + len(chunk)
        if isinstance(response, dict):
            accepted = response.get("accepted")
            rejected = response.get("rejected")
            stats["accepted"] = stats.get("accepted", 0) + (
                accepted if isinstance(accepted, int) else len(chunk)
            )
            stats["rejected"] = stats.get("rejected", 0) + (
                len(rejected) if isinstance(rejected, list)
                else rejected if isinstance(rejected, int) else 0
            )
        return True

    async def _unpushed_backlog(self, db, exclude_ids) -> List[Tuple[ExtractedMention, Dict[str, Any]]]:
        """Earlier mentions still waiting for a push, rebuilt as trender payloads."""
        since = datetime.now(timezone.utc) - timedelta(days=BACKLOG_MAX_AGE_DAYS)
        rows = (await db.execute(
            select(ExtractedMention)
            .where(ExtractedMention.pushed_to_trender == False)  # noqa: E712
            .order_by(ExtractedMention.created_at)
            .limit(BACKLOG_MAX_ROWS)
        )).scalars().all()
        out = []
        for row in rows:
            if row.id in exclude_ids:
                continue
            created = row.created_at
            if created is not None:
                created = created if created.tzinfo else created.replace(tzinfo=timezone.utc)
                if created < since:
                    continue
            out.append((row, self._mention_from_row(row)))
        return out

    @staticmethod
    def _mention_from_row(row: ExtractedMention) -> Dict[str, Any]:
        """Rebuild the trender payload from a stored row (the frozen output contract)."""
        citation = row.citation_json or {}
        when = row.mentioned_at or row.created_at or datetime.now(timezone.utc)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return {
            "symbol": row.symbol,
            "company_name": row.company_name,
            "source": citation.get("kind") or "youtube",
            "stance": row.stance,
            "sentiment_score": float(row.sentiment_score) if row.sentiment_score is not None else None,
            "confidence": float(row.confidence) if row.confidence is not None else None,
            "context": row.thesis,
            "citation": citation,
            "mentioned_at": when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
