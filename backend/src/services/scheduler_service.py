"""
Scheduler Service - the channel-intelligence feeder loop.

On a single global interval (env ``FEEDER_INTERVAL_SECONDS``, default 6h), for
each enabled ``SourceConfig`` with ``purpose == "finance"`` it runs a Cycle:

    collecting -> (new-item filter) -> extracting -> pushing -> completed/failed

A processed-item ledger (prior ``CollectedData``) means each run only extracts
NEW videos/messages, so LLM cost is bounded and reruns are cheap. Mentions are
deduped by ``dedupe_key`` before insert and gated by ``pushed_to_trender`` before
send, so retries never double-count.

Track-record calls: every extracted recommendation's call payload is stored on
its mention row (``call_payload_json``, also for rows that already existed) before
the ledger is written, then pushed to ``/api/track-record/calls`` with the same
rules as mentions: batches of at most ``TRENDER_PUSH_BATCH_SIZE`` (default ``PUSH_BATCH_SIZE``), a call is flagged pushed
only once trender accepted it (or already had it), a batch trender refuses as
invalid (400/413/422) is split down to single calls, any other failure (401, 503,
5xx, no connection) stops the loop until the next cycle, and unpushed calls
(failed, or rejected e.g. for an unmapped channel) are retried by later
track-record cycles for ``BACKLOG_MAX_AGE_DAYS``. A single call trender refuses
as invalid is not retried.

Wiring into app startup is guarded by ``FEEDER_ENABLED`` (default false).
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import or_, select

from .. import run_log
from ..config import get_settings
from ..database import AsyncSessionLocal
from ..models import Cycle, CollectedData, ExtractedMention, SourceConfig
from ..models.cycle import CycleStatus
from .collection_orchestrator import CollectionOrchestrator
from .extraction_service import ExtractionService
from .timestamps import normalize_published
from .trender_client import TrenderClient, TrenderPushError

logger = logging.getLogger(__name__)

# Default mentions/calls per request to trender (it re-scores candidates after every
# request). TRENDER_PUSH_BATCH_SIZE overrides it; read it with push_batch_size().
PUSH_BATCH_SIZE = 50
# Mentions whose push failed are retried on later cycles, but only this long.
BACKLOG_MAX_AGE_DAYS = 14
BACKLOG_MAX_ROWS = 500
# trender answers these when one mention in the batch is invalid: split the batch so the
# good ones still go through. Any other error (401 wrong token, 503 not configured, 5xx,
# no connection) is not about the content, so the loop stops and tries again next cycle.
SPLITTABLE_HTTP_STATUS = (400, 413, 422)


def not_permanently_refused(column=None):
    """SQL filter: rows trender did not refuse as invalid (those are not retried or counted).

    ``column`` is the response column to check: ``trender_response`` (mentions, default)
    or ``trender_calls_response`` (track-record calls).
    """
    column = ExtractedMention.trender_response if column is None else column
    refused = or_(*[
        column.like(f"error: trender returned {code}%")
        for code in SPLITTABLE_HTTP_STATUS
    ])
    return or_(column.is_(None), ~refused)
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


def push_batch_size() -> int:
    """Configured batch size (TRENDER_PUSH_BATCH_SIZE, default PUSH_BATCH_SIZE)."""
    return get_settings().trender_push_batch_size


def _chunks(seq: List[Any], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


@dataclass
class CallPushResult:
    """Outcome of one call push, from trender's ``{accepted, duplicates, rejected}``."""

    sent: int = 0
    accepted: int = 0
    duplicate: int = 0
    rejected: Dict[str, int] = field(default_factory=dict)  # reason -> count
    failed: int = 0  # transport / HTTP failures (left unpushed)

    @property
    def pushed(self) -> int:
        return self.accepted + self.duplicate


def _symbol_unique_batches(
    rows: List[ExtractedMention], max_size: Optional[int] = None
) -> List[List[ExtractedMention]]:
    """Split rows into batches of at most ``max_size`` with no symbol repeated in one.

    trender identifies rejected calls only by symbol, so this keeps the mapping
    from a rejection back to its row exact.
    """
    max_size = max_size or push_batch_size()
    batches: List[List[ExtractedMention]] = []
    symbols: List[set] = []
    for row in rows:
        symbol = (row.call_payload_json or {}).get("symbol")
        for batch, seen in zip(batches, symbols):
            if symbol not in seen and len(batch) < max_size:
                batch.append(row)
                seen.add(symbol)
                break
        else:
            batches.append([row])
            symbols.append({symbol})
    return batches


def _call_channel(row: ExtractedMention) -> str:
    payload = row.call_payload_json or {}
    return str((payload.get("citation") or {}).get("channel") or "-")


def log_call(
    row: ExtractedMention, outcome: str, reason: Optional[str] = None, detail: Optional[str] = None
) -> None:
    """One log line per call: channel, source item, symbol, outcome and reason.

    Outcomes: accepted, duplicate, accepted_or_duplicate (trender reports only counts),
    rejected, failed to send. Rejected and failed ones are warnings.
    """
    payload = row.call_payload_json or {}
    bad = outcome in ("rejected", "failed to send")
    logger.log(
        logging.WARNING if bad else logging.INFO,
        "call outcome=%s channel=%s item=%s symbol=%s reason=%s%s",
        outcome, _call_channel(row), payload.get("source_item_id"), payload.get("symbol"),
        reason or "-", f" ({detail})" if detail else "",
    )
    run_log.record_call("failed" if outcome == "failed to send" else outcome, reason)


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
        stats = self._new_stats()

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
            cycle_id = cycle.id  # read now: after a rollback the instance is expired

            try:
                # 1-3. Collect, drop failed / out-of-window items, filter against the ledger.
                result, new_items = await self._collect_new(
                    db, source_config, collect_spec, days, cutoff, stats, exclude_cycle_id=cycle.id
                )
                source_type = source_config.source_type
                logger.info(
                    "Cycle %s: %d collected, %d new after filters",
                    cycle.id, stats["collected"], len(new_items),
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
                # 5b. Track record: store each recommendation's call payload on its row
                # (before the ledger, so a crash after it still leaves the call to push).
                pending_calls: List[ExtractedMention] = []
                if source_config.track_record:
                    pending_calls = await self._attach_calls(db, mentions, origin="feeder")

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
                if source_config.track_record:
                    # 8. Push calls: this cycle's plus earlier ones still unpushed (same rules).
                    self._apply_call_stats(stats, await self._push_calls(db, pending_calls))

                cycle.status = CycleStatus.COMPLETED
                cycle.completed_at = datetime.now(timezone.utc)
                # Kept for /health/deep (e.g. "every new item failed extraction").
                cycle.config_snapshot = {**(cycle.config_snapshot or {}), "stats": dict(stats)}
                await db.commit()
                logger.info("Cycle %s completed: %d mention(s) stored", cycle.id, len(stored))

            except Exception as e:  # noqa: BLE001
                await db.rollback()
                fresh = await db.execute(select(Cycle).where(Cycle.id == cycle_id))
                cycle = fresh.scalar_one()
                cycle.status = CycleStatus.FAILED
                cycle.error_message = str(e)
                cycle.completed_at = datetime.now(timezone.utc)
                await db.commit()
                logger.error("Cycle %s failed: %s", cycle_id, e, exc_info=True)
                raise

            await db.refresh(cycle)
            return cycle, stats

    async def _collect_new(self, db, source_config, collect_spec, days, cutoff, stats, exclude_cycle_id=None):
        """Collect, then keep only items worth extracting. Returns ``(result, new_items)``."""
        source_type = source_config.source_type
        result = await self.orchestrator.collect(
            source_type=source_type,
            credential_ref=source_config.credential_ref,
            collect_spec=collect_spec,
            timeframe_days=days,
        )
        raw_items = result.get("data", []) or []
        stats["collected"] = len(raw_items)

        usable = []
        for it in raw_items:
            if isinstance(it, dict) and it.get("status") == "failed":
                stats["collector_failed"] += 1
                continue
            if cutoff is not None and not self._published_on_or_after(it, source_type, cutoff):
                stats["out_of_window"] += 1
                continue
            usable.append(it)

        seen = await self._seen_ids(db, source_type, exclude_cycle_id=exclude_cycle_id)
        new_items = [it for it in usable if self._item_id(it, source_type) not in seen]
        stats["already_seen"] = len(usable) - len(new_items)
        stats["new"] = len(new_items)
        return result, new_items

    async def dry_run(
        self,
        source_config: SourceConfig,
        *,
        timeframe_days: Optional[int] = None,
        spec_overrides: Optional[Dict[str, Any]] = None,
        cutoff: Optional[datetime] = None,
    ) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        """Collect and filter exactly like a cycle, but extract and push nothing, write nothing."""
        days = timeframe_days or self.timeframe_days
        collect_spec = {**(source_config.collect_spec or {}), **(spec_overrides or {})}
        stats = self._new_stats()
        async with self.session_factory() as db:
            _, new_items = await self._collect_new(db, source_config, collect_spec, days, cutoff, stats)
        return stats, new_items

    @staticmethod
    def _new_stats() -> Dict[str, int]:
        return {
            "collected": 0, "collector_failed": 0, "out_of_window": 0, "already_seen": 0,
            "new": 0, "extraction_failed": 0, "mentions_found": 0, "mentions_stored": 0,
            "pushed": 0, "accepted": 0, "rejected": 0, "push_failed": 0,
        }

    # ------------------------------------------------------------------ #
    # Ledger + mapping helpers
    # ------------------------------------------------------------------ #
    async def _seen_ids(self, db, source_type: str, exclude_cycle_id=None) -> set:
        """Collect all previously-seen item ids for this source from CollectedData."""
        # Backfill rows are archives (possibly of declined/dry runs), not proof an
        # item was processed, so they don't count as "seen" for the feeder.
        query = (
            select(CollectedData)
            .join(Cycle, Cycle.id == CollectedData.cycle_id)
            .where(CollectedData.source_type == source_type)
            .where(Cycle.name.notlike("backfill:%"))
        )
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
                is_recommendation=bool(mention.get("is_recommendation")),
                action=mention.get("action"),
                horizon=mention.get("horizon"),
                entry_price=mention.get("entry_price"),
                target_price=mention.get("target_price"),
                stop_price=mention.get("stop_price"),
                trigger_price=mention.get("trigger_price"),
                quote=mention.get("quote"),
                said_at=mention.get("said_at"),
                said_at_precision=mention.get("said_at_precision"),
                citation_json=mention.get("citation", {}) or {},
                mentioned_at=_parse_iso(mention.get("mentioned_at")),
                dedupe_key=key,
                pushed_to_trender=False,
                pushed_calls_to_trender=False,
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

        for chunk in _chunks(pairs, push_batch_size()):
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
            if status not in SPLITTABLE_HTTP_STATUS:
                return False
            if len(chunk) > 1:
                for single in chunk:
                    await self._push_chunk(db, [single], stats)
            return True

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
            .where(not_permanently_refused())
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

    async def _attach_calls(self, db, mentions: List[Dict[str, Any]], origin: str) -> List[ExtractedMention]:
        """
        Store the call payload of every extracted recommendation on its mention
        row, including rows that already existed (dedupe_key collision), and
        return the rows whose call still needs pushing.

        An existing row keeps its discovery fields and any payload it already
        has; only a missing payload is filled in.
        """
        model = self.extraction_service.model
        payloads: Dict[str, Dict[str, Any]] = {}
        for mention in mentions:
            call = ExtractionService.to_call_payload(mention, origin, model)
            if call is not None:
                payloads.setdefault(ExtractionService.dedupe_key_for(mention), call)
        if not payloads:
            return []

        keys = list(payloads)
        rows: Dict[str, ExtractedMention] = {}
        for i in range(0, len(keys), 500):  # stay under DB bind-parameter limits
            chunk = keys[i:i + 500]
            found = await db.execute(
                select(ExtractedMention).where(ExtractedMention.dedupe_key.in_(chunk))
            )
            rows.update({row.dedupe_key: row for row in found.scalars().all()})

        pending = []
        for key in keys:  # keep extraction order
            row = rows.get(key)
            if row is None:
                continue
            if row.call_payload_json is None:
                row.call_payload_json = payloads[key]
            if not row.pushed_calls_to_trender:
                pending.append(row)
        await db.commit()
        return pending

    async def repush_unpushed_calls(
        self, db, *, max_age_days: Optional[int] = BACKLOG_MAX_AGE_DAYS
    ) -> CallPushResult:
        """
        Re-send stored call payloads trender has not accepted yet (failed push, or
        rejected e.g. for an unmapped channel). Not scoped to a source config.

        Default: the cycle's window and row cap. ``max_age_days=None`` (the manual
        ``backfill_track_record.py --repush``) takes every eligible call, any age.
        Calls trender refused as invalid are never re-sent.
        """
        return await self._push_calls(db, [], max_age_days=max_age_days)

    async def _unpushed_calls_backlog(
        self, db, exclude_ids, max_age_days: Optional[int] = BACKLOG_MAX_AGE_DAYS
    ) -> List[ExtractedMention]:
        """Earlier calls still waiting for trender, oldest first (window and cap unless None)."""
        query = (
            select(ExtractedMention)
            .where(
                ExtractedMention.call_payload_json.is_not(None),
                ExtractedMention.pushed_calls_to_trender == False,  # noqa: E712
            )
            .where(not_permanently_refused(ExtractedMention.trender_calls_response))
            .order_by(ExtractedMention.created_at)
        )
        if max_age_days is not None:
            query = query.limit(BACKLOG_MAX_ROWS)
        rows = (await db.execute(query)).scalars().all()
        since = (
            None if max_age_days is None
            else datetime.now(timezone.utc) - timedelta(days=max_age_days)
        )
        out = []
        for row in rows:
            if row.id in exclude_ids:
                continue
            created = row.created_at
            if since is not None and created is not None:
                created = created if created.tzinfo else created.replace(tzinfo=timezone.utc)
                if created < since:
                    continue
            out.append(row)
        return out

    async def _push_calls(
        self, db, rows: List[ExtractedMention], *, max_age_days: Optional[int] = BACKLOG_MAX_AGE_DAYS
    ) -> CallPushResult:
        """
        Push ``rows``' stored call payloads, plus earlier unpushed calls, to trender's
        track-record endpoint (see the module docstring for the rules).

        Accepted and duplicate calls are flagged pushed; rejected calls (matched back
        by symbol, unique per request) and failures stay unpushed with the response
        recorded, so a later cycle retries them.
        """
        result = CallPushResult()
        current = [r for r in rows if r.call_payload_json and not r.pushed_calls_to_trender]
        backlog = await self._unpushed_calls_backlog(
            db, exclude_ids={r.id for r in current}, max_age_days=max_age_days
        )
        todo = backlog + current
        if not todo:
            return result
        if backlog:
            logger.info("Retrying %d track-record call(s) that were not pushed earlier", len(backlog))

        attempted: set = set()
        for batch in _symbol_unique_batches(todo):
            if not await self._push_call_batch(db, batch, result, attempted):
                logger.error("trender unreachable; %d call(s) stay unpushed",
                             sum(1 for r in todo if not r.pushed_calls_to_trender))
                break

        for r in todo:
            if id(r) not in attempted:  # the loop stopped before this call's batch
                log_call(r, "failed to send", "not attempted: trender unreachable earlier in this run")
        result.sent = len(attempted)
        result.failed = sum(
            1 for r in todo
            if not r.pushed_calls_to_trender
            and not (r.trender_calls_response or "").startswith("rejected:")
        )
        if result.rejected:
            logger.warning("trender rejected track-record call(s): %s", result.rejected)
        return result

    async def _push_call_batch(self, db, batch, result: CallPushResult, attempted: set) -> bool:
        """Push one batch of calls. Returns False when trender could not be reached at all."""
        attempted.update(id(row) for row in batch)
        try:
            response = await self.trender_client.push_calls(
                [row.call_payload_json for row in batch]
            )
        except Exception as e:  # noqa: BLE001 - record failure, leave flag False
            logger.error(
                "trender calls push failed for %d call(s): %s: %s", len(batch), type(e).__name__, e
            )
            for row in batch:
                row.trender_calls_response = f"error: {e}"
            await db.commit()
            splits = getattr(e, "status_code", None) in SPLITTABLE_HTTP_STATUS and len(batch) > 1
            if not splits:  # a split batch is logged call by call when its singles are sent
                reason = getattr(e, "reason", None) or f"{type(e).__name__}: {e}"
                for row in batch:
                    log_call(row, "failed to send", reason, str(e)[:200] if reason != str(e) else None)
            if getattr(e, "status_code", None) not in SPLITTABLE_HTTP_STATUS:
                return False
            if len(batch) > 1:  # one invalid call must not block the others
                for single in batch:
                    await self._push_call_batch(db, [single], result, attempted)
            return True

        response = response if isinstance(response, dict) else {}
        rejected: Dict[str, str] = {}
        for entry in response.get("rejected") or []:
            reason = entry.get("reason") or "unknown"
            rejected[entry.get("symbol")] = reason
            result.rejected[reason] = result.rejected.get(reason, 0) + 1
        duplicates = int(response.get("duplicates") or 0)
        result.duplicate += duplicates
        result.accepted += int(response.get("accepted", len(batch) - duplicates - len(rejected)))

        text = str(response)
        # trender reports duplicates only as a count, so name the outcome exactly when
        # every pushed call in the batch was new or every one was already there.
        pushed_in_batch = len(batch) - len(rejected)
        ok_outcome = (
            "accepted_or_duplicate" if 0 < duplicates < pushed_in_batch
            else "duplicate" if duplicates and duplicates >= pushed_in_batch
            else "accepted"
        )
        for row in batch:
            reason = rejected.get(row.call_payload_json.get("symbol"))
            if reason is None:
                row.pushed_calls_to_trender = True
                row.trender_calls_response = text
                log_call(row, ok_outcome)
            else:
                row.trender_calls_response = f"rejected: {reason}; {text}"
                log_call(row, "rejected", reason)
        await db.commit()
        return True

    @staticmethod
    def _apply_call_stats(stats: Dict[str, int], calls: CallPushResult) -> None:
        stats["calls_sent"] = calls.sent
        stats["calls_pushed"] = calls.pushed
        stats["calls_rejected"] = sum(calls.rejected.values())
        stats["calls_push_failed"] = calls.failed
