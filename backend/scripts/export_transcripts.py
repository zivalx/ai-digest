#!/usr/bin/env python3
"""
Export every stored collected item (transcripts) to files on disk.

Layout: DIR/<channel-slug>/<YYYY-MM>/<YYYY-MM-DD>_<item_id>.json, items without
a publish time under DIR/<channel-slug>/undated/, plus DIR/<channel-slug>/index.csv.
Items are deduped by id across all CollectedData rows (latest wins). Idempotent.

Usage (from backend/):
    python scripts/export_transcripts.py --out ./transcripts [--source-type youtube|telegram]
"""
import argparse
import asyncio
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from src.database import AsyncSessionLocal  # noqa: E402
from src.models import CollectedData  # noqa: E402
from src.services.scheduler_service import SchedulerService  # noqa: E402
from src.services.timestamps import normalize_published  # noqa: E402

_NON_ALNUM = re.compile(r"[\W_]+", re.UNICODE)  # unicode-aware: Hebrew letters survive


def slugify(name: Any) -> str:
    slug = _NON_ALNUM.sub("-", str(name or "").lower()).strip("-")
    return slug or "unknown"


def _safe_id(item_id: str) -> str:
    return re.sub(r"[^\w.-]+", "-", item_id, flags=re.UNICODE).strip("-") or "item"


def _channel(item: Dict[str, Any]) -> str:
    return (item.get("channel") or item.get("channel_title") or item.get("channel_name")
            or item.get("chat") or item.get("chat_title") or item.get("source") or "unknown")


def collect_items(rows: List[CollectedData]) -> Dict[tuple, Dict[str, Any]]:
    """{(source_type, item_id): item}, rows ordered oldest -> newest so latest wins."""
    out: Dict[tuple, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row.data, list):
            continue
        for it in row.data:
            item_id = SchedulerService._item_id(it, row.source_type)
            if item_id is not None:
                out[(row.source_type, item_id)] = it
    return out


def build_record(item: Dict[str, Any], source_type: str, item_id: str) -> Dict[str, Any]:
    raw = item.get("published_at") or item.get("upload_date") or item.get("date")
    ts, precision = normalize_published(raw)
    rec: Dict[str, Any] = {
        "item_id": item_id,
        "source_type": source_type,
        "channel": _channel(item),
        "title": item.get("title"),
        "url": item.get("url") or item.get("webpage_url"),
        "published_at": ts,
        "ts_precision": precision,
    }
    if item.get("transcript_source"):
        rec["transcript_source"] = item["transcript_source"]
    rec["transcript"] = item.get("transcript") or item.get("text") or ""
    if item.get("transcript_segments"):
        rec["transcript_segments"] = item["transcript_segments"]
    return rec


def export(rows: List[CollectedData], out_dir: Path, source_type: Optional[str] = None) -> int:
    items = collect_items([r for r in rows if source_type in (None, r.source_type)])
    index: Dict[str, List[Dict[str, Any]]] = {}
    for (stype, item_id), item in items.items():
        rec = build_record(item, stype, item_id)
        slug = slugify(rec["channel"])
        day = rec["published_at"][:10] if rec["published_at"] else None
        folder = Path(slug) / (day[:7] if day else "undated")
        name = f"{day}_{_safe_id(item_id)}.json" if day else f"{_safe_id(item_id)}.json"
        path = out_dir / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        index.setdefault(slug, []).append({
            "date": day or "", "item_id": item_id, "title": rec["title"] or "",
            "url": rec["url"] or "", "transcript_chars": len(rec["transcript"]),
            "file": str(folder / name),
        })
    for slug, rows_ in index.items():
        rows_.sort(key=lambda r: (r["date"], r["item_id"]))
        with open(out_dir / slug / "index.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["date", "item_id", "title", "url",
                                              "transcript_chars", "file"])
            w.writeheader()
            w.writerows(rows_)
    return len(items)


async def load_rows(session_factory) -> List[CollectedData]:
    async with session_factory() as db:
        return list((await db.execute(
            select(CollectedData).order_by(CollectedData.collected_at.asc())
        )).scalars().all())


async def main(args) -> None:
    rows = await load_rows(AsyncSessionLocal)
    n = export(rows, Path(args.out), args.source_type)
    print(f"exported {n} item(s) to {args.out}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--source-type", choices=["youtube", "telegram"])
    return p.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
