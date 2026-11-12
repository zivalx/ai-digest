"""CLI flag parsing for the backfill script."""
import importlib.util
from pathlib import Path

import pytest

_p = Path(__file__).resolve().parent.parent / "scripts" / "backfill_track_record.py"
_spec = importlib.util.spec_from_file_location("backfill_cli", _p)
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def test_flags_default_to_none():
    a = cli.parse_args(["--source-config", "X"])
    assert (a.concurrency, a.delay, a.cookies) == (None, None, None)


def test_flags_parsed():
    a = cli.parse_args(["--source-config", "X", "--concurrency", "2", "--delay", "1.5",
                        "--cookies", "/tmp/c.txt"])
    assert (a.concurrency, a.delay, a.cookies) == (2, 1.5, "/tmp/c.txt")


@pytest.mark.parametrize("bad", [["--concurrency", "0"], ["--delay", "-1"]])
def test_invalid_values_rejected(bad):
    with pytest.raises(SystemExit):
        cli.parse_args(["--source-config", "X", *bad])


@pytest.mark.asyncio
async def test_main_passes_overrides_to_service(monkeypatch):
    seen = {}

    class FakeSvc:
        def __init__(self, *a, **k): pass

        async def run(self, *a, **kw):
            seen.update(kw)
            from src.services.backfill_service import BackfillReport
            return BackfillReport(0, 0, 0, 0.0, 0, True)

    monkeypatch.setattr(cli, "BackfillService", FakeSvc)
    monkeypatch.setattr(cli, "CollectionOrchestrator", lambda: None)
    monkeypatch.setattr(cli, "ExtractionService", lambda: None)
    monkeypatch.setattr(cli, "TrenderClient", lambda: None)
    args = cli.parse_args(["--source-config", "X", "--dry-run", "--concurrency", "2",
                           "--delay", "1", "--cookies", "/c.txt", "--no-whisper"])
    await cli.main(args)
    assert (seen["concurrency"], seen["delay"], seen["cookies_file"]) == (2, 1.0, "/c.txt")
    assert seen["no_whisper"] is True
