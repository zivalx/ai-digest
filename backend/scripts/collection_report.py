#!/usr/bin/env python3
"""
Per-item status log of everything the backfills collected for a source config.

One CSV row per video/post (newest collection attempt wins): whether it has a
transcript, where it came from (YouTube captions or Whisper), its length, and the
exact error when it failed. Writes DIR/<config-slug>_status.csv and prints totals.

Usage (from backend/):
    python scripts/collection_report.py --out ../logs --source-config "TR English" [--source-config ...]
"""
import argparse
import asyncio
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from src.database import AsyncSessionLocal  # noqa: E402
from src.models import CollectedData, Cycle, SourceConfig  # noqa: E402
from src.services.backfill_service import backfill_cycle_name  # noqa: E402
from src.services.scheduler_service import SchedulerService  # noqa: E402

FIELDS = ["item_id", "url", "title", "published", "status", "transcript_source",
          "transcript_chars", "error", "attempted_at"]


def _text(it: Dict[str, Any]) -> str:
    return it.get("transcript") or it.get("text") or ""


def status_rows(rows: List[Any], source_type: str) -> List[Dict[str, Any]]:
    """rows: (CollectedData, cycle started_at) newest first. Newest attempt per item wins,
    but a stored success beats any later failure."""
    out: Dict[str, Dict[str, Any]] = {}
    for data, started_at in rows:
        for it in data.data if isinstance(data.data, list) else []:
            item_id = SchedulerService._item_id(it, source_type)
            if item_id is None:
                continue
            text = _text(it)
            failed = it.get("status") == "failed"
            seen = out.get(item_id)
            # Newest attempt wins, except that a success is never hidden by a later
            # failure (backfills reuse the stored success).
            if seen is not None and (failed or seen["status"] != "failed"):
                continue
            out[item_id] = {
                "item_id": item_id,
                "url": it.get("url") or it.get("webpage_url") or it.get("message_link") or "",
                "title": (it.get("title") or "").replace("\n", " ")[:200],
                "published": it.get("published_at") or it.get("upload_date") or it.get("date") or "",
                "status": "failed" if failed else ("ok" if text.strip() else "no_text"),
                "transcript_source": "" if failed else (it.get("transcript_source") or source_type),
                "transcript_chars": len(text),
                "error": (it.get("error") or "").replace("\n", " ")[:500] if failed else "",
                "attempted_at": started_at.isoformat() if started_at else "",
            }
    return sorted(out.values(), key=lambda r: str(r["published"]), reverse=True)


async def report(names: List[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    async with AsyncSessionLocal() as db:
        for name in names:
            config = (await db.execute(
                select(SourceConfig).where(SourceConfig.name == name))).scalars().first()
            if config is None:
                print(f"{name}: no such source config")
                continue
            rows = (await db.execute(
                select(CollectedData, Cycle.started_at)
                .join(Cycle, Cycle.id == CollectedData.cycle_id)
                .where(Cycle.name == backfill_cycle_name(name))
                .order_by(Cycle.started_at.desc(), CollectedData.collected_at.desc())
            )).all()
            items = status_rows(rows, config.source_type)
            path = out_dir / f"{name.lower().replace(' ', '-')}_status.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(items)
            counts = Counter(r["status"] for r in items)
            sources = Counter(r["transcript_source"] for r in items if r["status"] == "ok")
            print(f"{name}: {len(items)} items, "
                  + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
                  + (f" (sources: {dict(sources)})" if sources else "") + f" -> {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True, type=Path, help="directory for the CSV files")
    p.add_argument("--source-config", action="append", required=True, dest="names")
    args = p.parse_args()
    asyncio.run(report(args.names, args.out))


if __name__ == "__main__":
    main()
