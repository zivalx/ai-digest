#!/usr/bin/env python3
"""
Backfill the influencer track record from a channel's back catalog.

Note: Whisper transcription fallback is not supported by the collectors
library yet, so there is no --whisper flag.

Usage (from backend/):
    python scripts/backfill_track_record.py --source-config "Finance YT" --dry-run
    python scripts/backfill_track_record.py --source-config "Finance YT" --since 2025-01-01 --limit 100
    python scripts/backfill_track_record.py --repush   # re-send unpushed calls only
"""
import argparse
import asyncio
import os
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import AsyncSessionLocal  # noqa: E402
from src.services.backfill_service import BackfillService  # noqa: E402
from src.services.collection_orchestrator import CollectionOrchestrator  # noqa: E402
from src.services.extraction_service import ExtractionService  # noqa: E402
from src.services.trender_client import TrenderClient  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--source-config", help="SourceConfig name (required unless --repush)")
    p.add_argument("--since", type=date.fromisoformat,
                   default=date.today() - timedelta(days=730), help="YYYY-MM-DD (default: today - 730d)")
    p.add_argument("--limit", type=int, default=200, help="max items per channel (default 200)")
    p.add_argument("--dry-run", action="store_true", help="estimate only; no LLM, no pushes")
    p.add_argument("--usd-per-mtok", type=float,
                   default=float(os.getenv("BACKFILL_USD_PER_MTOK", "3.0")))
    p.add_argument("--repush", action="store_true",
                   help="only re-send stored calls trender has not accepted (no collection, no LLM)")
    args = p.parse_args(argv)
    if not args.repush and not args.source_config:
        p.error("--source-config is required unless --repush")
    return args


def confirm(message: str) -> bool:
    return input(f"{message} [y/N] ").strip().lower() == "y"


UNMAPPED_HINT = (
    "HINT: 'unmapped channel' means trender has no influencer for that channel - "
    "seed influencer channels with the exact YouTube channel name, then run --repush."
)


def print_calls(accepted: int, duplicate: int, rejected: dict, failed: int) -> None:
    print(f"calls: {accepted} accepted, {duplicate} duplicate, "
          f"{sum(rejected.values())} rejected, {failed} failed to send")
    for reason, count in sorted(rejected.items()):
        print(f"  rejected ({reason}): {count}")
    if "unmapped channel" in rejected:
        print(UNMAPPED_HINT)
    if rejected or failed:
        print("Rejected/failed calls are kept and retried by the next feeder cycle or --repush.")


async def main(args) -> None:
    service = BackfillService(
        CollectionOrchestrator(), ExtractionService(), TrenderClient(),
        AsyncSessionLocal, usd_per_mtok=args.usd_per_mtok,
    )
    if args.repush:
        result = await service.repush()
        print(f"{result.sent} unpushed call(s) re-sent")
        print_calls(result.accepted, result.duplicate, result.rejected, result.failed)
        return

    report = await service.run(
        args.source_config, args.since, args.limit, args.dry_run, confirm
    )
    for it in report.items:
        print(f"  {it['id']}  {it['published'] or '-'}  {it['transcript_chars']} chars  {it['title'] or ''}")
    if report.unavailable:
        print(f"{len(report.unavailable)} unavailable item(s) (not processed; retried next run):")
        for it in report.unavailable:
            print(f"  {it['id']}  {it['reason']}  {it['title'] or ''}")
    print(
        f"{report.items_new} new items, ~{report.est_input_tokens} input tokens, "
        f"~${report.est_cost_usd:.2f} (lower bound: transcript tokens only)"
    )
    if report.mentions_failed:
        print(
            f"WARNING: {report.mentions_failed} mention(s) failed to push to trender; "
            "they are stored locally but NOT delivered."
        )
    if not report.dry_run:
        print_calls(report.calls_accepted, report.calls_duplicate,
                    report.calls_rejected, report.calls_failed)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
