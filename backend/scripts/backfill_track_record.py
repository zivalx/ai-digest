#!/usr/bin/env python3
"""
Backfill the influencer track record from a channel's back catalog.

Note: Whisper transcription fallback is not supported by the collectors
library yet, so there is no --whisper flag.

Usage (from backend/):
    python scripts/backfill_track_record.py --source-config "Finance YT" --dry-run
    python scripts/backfill_track_record.py --source-config "Finance YT" --since 2025-01-01 --limit 100
    python scripts/backfill_track_record.py --source-config "Finance YT" --from-collected  # no re-collection
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
from src.run_log import run_log  # noqa: E402
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
    p.add_argument("--dry-run", action="store_true", help="collect + store raw transcripts and estimate only; no LLM, no pushes")
    p.add_argument("--usd-per-mtok", type=float,
                   default=float(os.getenv("BACKFILL_USD_PER_MTOK", "3.0")))
    p.add_argument("--from-collected", action="store_true",
                   help="reuse items stored by a previous backfill of this source config "
                        "instead of collecting again")
    p.add_argument("--concurrency", type=int, default=None,
                   help="max videos fetched concurrently (overrides collect_spec; default 1)")
    p.add_argument("--delay", type=float, default=None,
                   help="seconds to pause before each video fetch, randomized 0.5x-1.5x "
                        "(overrides collect_spec; default 6)")
    p.add_argument("--cookies", metavar="FILE", default=None,
                   help="Netscape cookies.txt for YouTube (overrides collect_spec cookies_file)")
    p.add_argument("--no-whisper", action="store_true",
                   help="YouTube: captions only, no local Whisper fallback (missing ones retry next run)")
    p.add_argument("--repush", action="store_true",
                   help="only re-send stored calls trender has not accepted (no collection, no LLM)")
    args = p.parse_args(argv)
    if not args.repush and not args.source_config:
        p.error("--source-config is required unless --repush")
    if args.concurrency is not None and args.concurrency < 1:
        p.error("--concurrency must be >= 1")
    if args.delay is not None and args.delay < 0:
        p.error("--delay must be >= 0")
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


def summarize_report(report) -> dict:
    """Counts only (no item lists) for the run's summary file."""
    return {k: v for k, v in vars(report).items() if k not in ("items", "unavailable")} | {
        "unavailable_items": len(report.unavailable)}


async def main(args) -> None:
    label = "repush" if args.repush else args.source_config
    with run_log("backfill-track-record", label) as run:
        if run.log_path:
            print(f"run log: {run.log_path}", file=sys.stderr)
        await _main(args, run)


async def _main(args, run) -> None:
    service = BackfillService(
        CollectionOrchestrator(), ExtractionService(), TrenderClient(),
        AsyncSessionLocal, usd_per_mtok=args.usd_per_mtok,
    )
    if args.repush:
        result = await service.repush()
        run.summary["stats"] = {"sent": result.sent}
        print(f"{result.sent} unpushed call(s) re-sent")
        print_calls(result.accepted, result.duplicate, result.rejected, result.failed)
        return

    report = await service.run(
        args.source_config, args.since, args.limit, args.dry_run, confirm,
        reuse_collected=args.from_collected,
        concurrency=args.concurrency, delay=args.delay, cookies_file=args.cookies,
        no_whisper=args.no_whisper,
    )
    run.summary["stats"] = summarize_report(report)
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
    if report.aborted_reason:
        print(f"STOPPED: analysis {report.aborted_reason}. "
              "Nothing else was sent; unprocessed items are retried on the next run.")
    if report.mentions_failed:
        print(
            f"WARNING: {report.mentions_failed} mention(s) failed to push to trender; "
            "they are stored locally but NOT delivered."
        )
    if not report.dry_run:
        print_calls(report.calls_accepted, report.calls_duplicate,
                    report.calls_rejected, report.calls_failed)


if __name__ == "__main__":
    try:
        asyncio.run(main(parse_args()))
    except ValueError as exc:  # e.g. unknown config, no stored data for --from-collected
        sys.exit(f"error: {exc}")
