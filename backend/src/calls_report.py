"""
Which track-record calls has trender NOT taken, per channel, and why. Read-only.

    docker exec ai-digest-app python -m src.calls_report
    docker exec ai-digest-app python -m src.calls_report --csv /data/logs/calls.csv
    python -m src.calls_report --channel "Noam Stocks" --status rejected failed

Calls are listed per channel, newest first, with the link to the source post/video:

* pending  - extracted, not sent yet
* rejected - trender refused the call (reason from trender, e.g. "unknown symbol")
* failed   - the push itself failed (timeout, HTTP error); the error is shown

Pending, rejected and failed calls are retried by later feeder cycles or
``scripts/backfill_track_record.py --repush``, except an HTTP 400/413/422 refusal.
This command never changes anything. Accepted/duplicate calls are not listed.
"""
import argparse
import asyncio
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Set

from sqlalchemy import select

from .database import AsyncSessionLocal
from .models import ExtractedMention

STATUSES = ("pending", "rejected", "failed")
CSV_FIELDS = ["channel", "status", "symbol", "action", "said_at", "reason", "link",
              "source_item_id", "created_at"]
NO_CHANNEL = "(no channel)"


@dataclass
class CallRow:
    channel: str
    status: str
    symbol: str
    action: str
    said_at: str
    reason: str
    link: str
    source_item_id: str
    created_at: Optional[datetime]


def classify(response: Optional[str]) -> tuple:
    """``(status, reason)`` from the stored trender response of an unpushed call."""
    text = (response or "").strip()
    if text.startswith("rejected:"):
        return "rejected", text[len("rejected:"):].split(";", 1)[0].strip()
    if text.startswith("error:"):
        return "failed", text[len("error:"):].strip()
    return "pending", ""


async def collect(
    session_factory=None,
    *,
    channel: Optional[str] = None,
    statuses: Optional[Iterable[str]] = None,
) -> List[CallRow]:
    """Calls trender has not accepted, grouped by channel (A-Z), newest first within each."""
    session_factory = session_factory or AsyncSessionLocal
    wanted: Set[str] = set(statuses or STATUSES)
    async with session_factory() as db:
        result = await db.execute(
            select(ExtractedMention).where(
                ExtractedMention.call_payload_json.is_not(None),
                ExtractedMention.pushed_calls_to_trender == False,  # noqa: E712
            )
        )
        mentions = result.scalars().all()

    rows: List[CallRow] = []
    for m in mentions:
        payload = m.call_payload_json or {}
        citation = payload.get("citation") or {}
        name = str(citation.get("channel") or NO_CHANNEL)
        status, reason = classify(m.trender_calls_response)
        if status not in wanted or (channel and name != channel):
            continue
        rows.append(CallRow(
            channel=name, status=status, symbol=str(payload.get("symbol") or m.symbol),
            action=str(payload.get("action") or ""), said_at=str(payload.get("said_at") or ""),
            reason=reason, link=str(citation.get("url") or citation.get("message_link") or ""),
            source_item_id=str(payload.get("source_item_id") or ""), created_at=m.created_at,
        ))
    newest = sorted(rows, key=lambda r: (r.created_at is not None, r.created_at, r.said_at), reverse=True)
    return sorted(newest, key=lambda r: r.channel.lower())  # stable: keeps newest-first per channel


def format_report(rows: List[CallRow]) -> str:
    if not rows:
        return "No pending, rejected or failed calls."
    lines: List[str] = []
    channels = list(dict.fromkeys(r.channel for r in rows))
    for name in channels:
        mine = [r for r in rows if r.channel == name]
        counts = {s: sum(1 for r in mine if r.status == s) for s in STATUSES}
        lines.append(f"== {name}: " + ", ".join(f"{s} {counts[s]}" for s in STATUSES) + " ==")
        for r in mine:
            when = r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "-"
            reason = f" [{r.reason}]" if r.reason else ""
            lines.append(f"  {when}  {r.status:<8} {r.symbol:<10} {r.action:<6}{reason}  {r.link}")
    totals = {s: sum(1 for r in rows if r.status == s) for s in STATUSES}
    lines.append("")
    lines.append("total: " + ", ".join(f"{s} {totals[s]}" for s in STATUSES))
    return "\n".join(lines)


def write_csv(rows: List[CallRow], path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8-sig") as fh:  # BOM: Excel reads Hebrew
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "channel": r.channel, "status": r.status, "symbol": r.symbol, "action": r.action,
                "said_at": r.said_at, "reason": r.reason, "link": r.link,
                "source_item_id": r.source_item_id,
                "created_at": r.created_at.isoformat() if r.created_at else "",
            })


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip() if __doc__ else None)
    p.add_argument("--csv", metavar="PATH", help="also write the same rows as CSV")
    p.add_argument("--channel", help="only this channel (exact name as in the citation)")
    p.add_argument("--status", nargs="+", choices=STATUSES, help="only these statuses")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    rows = asyncio.run(collect(channel=args.channel, statuses=args.status))
    print(format_report(rows))
    if args.csv:
        write_csv(rows, args.csv)
        print(f"csv: {args.csv} ({len(rows)} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
