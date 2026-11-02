#!/usr/bin/env python3
"""
Backfill the influencer track record from a channel's back catalog.

Note: Whisper transcription fallback is not supported by the collectors
library yet, so there is no --whisper flag.

Usage (from backend/):
    python scripts/backfill_track_record.py --source-config "Finance YT" --dry-run
    python scripts/backfill_track_record.py --source-config "Finance YT" --since 2025-01-01 --limit 100
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
    p.add_argument("--source-config", required=True, help="SourceConfig name")
    p.add_argument("--since", type=date.fromisoformat,
                   default=date.today() - timedelta(days=730), help="YYYY-MM-DD (default: today - 730d)")
    p.add_argument("--limit", type=int, default=200, help="max items per channel (default 200)")
    p.add_argument("--dry-run", action="store_true", help="estimate only; no LLM, no pushes")
    p.add_argument("--usd-per-mtok", type=float,
                   default=float(os.getenv("BACKFILL_USD_PER_MTOK", "3.0")))
    return p.parse_args(argv)


def confirm(message: str) -> bool:
    return input(f"{message} [y/N] ").strip().lower() == "y"


async def main(args) -> None:
    service = BackfillService(
        CollectionOrchestrator(), ExtractionService(), TrenderClient(),
        AsyncSessionLocal, usd_per_mtok=args.usd_per_mtok,
    )
    report = await service.run(
        args.source_config, args.since, args.limit, args.dry_run, confirm
    )
    for it in report.items:
        print(f"  {it['id']}  {it['published'] or '-'}  {it['transcript_chars']} chars  {it['title'] or ''}")
    print(
        f"{report.items_new} new items, ~{report.est_input_tokens} input tokens, "
        f"~${report.est_cost_usd:.2f} (lower bound: transcript tokens only)"
    )
    if report.mentions_failed or report.calls_failed:
        print(
            f"WARNING: {report.mentions_failed} mention(s) and {report.calls_failed} call(s) "
            "failed to push to trender; they are stored locally but NOT delivered."
        )
    if not report.dry_run:
        print(f"{report.calls_pushed} call(s) pushed")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
