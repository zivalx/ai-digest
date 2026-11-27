"""Run logs: every feeder cycle / backfill writes <time>_<source>.log and a .summary.json
into LOG_DIR by itself (no shell redirection)."""
import json
import logging
from datetime import datetime, timezone

import pytest

from src import run_log as rl


@pytest.fixture(autouse=True)
def _clean_root_handlers():
    before = list(logging.getLogger().handlers)
    yield
    for h in list(logging.getLogger().handlers):
        if h not in before:
            logging.getLogger().removeHandler(h)


NOW = datetime(2026, 11, 27, 9, 30, 5, tzinfo=timezone.utc)


def test_writes_log_and_summary_named_with_time_and_source(tmp_path):
    with rl.run_log("feeder", "Finance TG / Hebrew", log_dir=tmp_path, now=NOW) as run:
        logging.getLogger("x").info("hello from the run")
        rl.record_call("accepted")
        rl.record_call("rejected", "unknown symbol")
        rl.record_call("rejected", "unknown symbol")
        rl.record_call("failed", "ReadTimeout after 120s")
        run.summary["stats"] = {"new": 3}

    log = tmp_path / "20261127-093005_feeder_Finance-TG-Hebrew.log"
    assert run.log_path == log and log.exists()
    assert "hello from the run" in log.read_text()
    summary = json.loads(run.summary_path.read_text())
    assert run.summary_path.name == log.stem + ".summary.json"
    assert summary["kind"] == "feeder" and summary["source_config"] == "Finance TG / Hebrew"
    assert summary["status"] == "ok" and summary["log_file"] == log.name
    assert summary["calls"] == {
        "accepted": 1, "duplicate": 0, "accepted_or_duplicate": 0,
        "rejected": {"unknown symbol": 2}, "failed": {"ReadTimeout after 120s": 1},
    }
    assert summary["stats"] == {"new": 3}


def test_logging_stops_after_the_run_and_errors_are_recorded(tmp_path):
    with pytest.raises(RuntimeError):
        with rl.run_log("backfill", None, log_dir=tmp_path, now=NOW) as run:
            raise RuntimeError("boom")
    logging.getLogger("x").warning("after the run")
    assert "after the run" not in run.log_path.read_text()
    summary = json.loads(run.summary_path.read_text())
    assert summary["status"] == "error" and "RuntimeError: boom" in summary["error"]
    assert "boom" in run.log_path.read_text()  # traceback is in the log


def test_same_second_runs_do_not_overwrite_each_other(tmp_path):
    paths = []
    for _ in range(2):
        with rl.run_log("feeder", "a", log_dir=tmp_path, now=NOW) as run:
            paths.append(run.log_path)
    assert paths[0] != paths[1] and all(p.exists() for p in paths)


def test_unwritable_log_dir_never_breaks_the_run(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with rl.run_log("feeder", "a", log_dir=blocker / "sub", now=NOW) as run:
        rl.record_call("accepted")
    assert run.log_path is None and run.summary["calls"]["accepted"] == 1


def test_record_call_without_a_run_is_a_noop():
    rl.record_call("accepted")  # must not raise


def test_log_dir_comes_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    with rl.run_log("feeder", "a", now=NOW) as run:
        pass
    assert run.log_path.parent == tmp_path / "logs"


# --- the three entry points write a run log by themselves ----------------------------------

def _written(tmp_path):
    return sorted(p.name for p in tmp_path.iterdir())


@pytest.mark.asyncio
async def test_feeder_cycle_writes_log_and_summary_with_call_counts(session_factory, tmp_path, monkeypatch):
    from tests.test_scheduler_service import FakeExtraction, FakeOrchestrator, FakeTrender, _make_finance_config
    from src.services.scheduler_service import SchedulerService

    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    cfg = await _make_finance_config(session_factory)
    svc = SchedulerService(
        session_factory=session_factory,
        orchestrator=FakeOrchestrator([{"video_id": "AAA", "channel": "@finance", "title": "One",
                                        "transcript": "buy nvda"}]),
        extraction_service=FakeExtraction(), trender_client=FakeTrender())
    await svc.run_finance_cycle(cfg)

    names = _written(tmp_path)
    assert len(names) == 2
    assert names[0].endswith("_feeder_Finance-YT.log") and names[1].endswith("_feeder_Finance-YT.summary.json")
    summary = json.loads((tmp_path / names[1]).read_text())
    assert summary["status"] == "ok" and summary["source_config"] == "Finance YT"
    assert "new" in summary["stats"] and "calls" in summary
    assert "Cycle " in (tmp_path / names[0]).read_text()


def test_backfill_cli_writes_log_and_summary(session_factory, tmp_path, monkeypatch):
    from src import backfill

    monkeypatch.setenv("LOG_DIR", str(tmp_path))

    async def fake_run_backfill(**kwargs):
        logging.getLogger("backfill").info("fake backfill ran")
        return backfill.BackfillReport(days=3, max_items=None, window_start=NOW, dry_run=True)

    monkeypatch.setattr(backfill, "run_backfill", fake_run_backfill)
    assert backfill.main(["--days", "3", "--source", "Finance YT", "--dry-run"]) == 0
    log, summary = sorted(tmp_path.iterdir())
    assert log.name.endswith("_backfill_Finance-YT.log") and "fake backfill ran" in log.read_text()
    assert json.loads(summary.read_text())["stats"]["days"] == 3


@pytest.mark.asyncio
async def test_track_record_script_writes_log_and_summary(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    from src.services.backfill_service import BackfillReport

    path = Path(__file__).resolve().parent.parent / "scripts" / "backfill_track_record.py"
    spec = importlib.util.spec_from_file_location("backfill_cli_log", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    class FakeSvc:
        def __init__(self, *a, **k): pass

        async def run(self, *a, **kw):
            return BackfillReport(items_found=2, items_new=2, est_input_tokens=0, est_cost_usd=0.0,
                                  calls_pushed=1, dry_run=False, calls_accepted=1,
                                  items=[{"id": "V", "published": None, "transcript_chars": 1, "title": "t"}])

    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "BackfillService", FakeSvc)
    for name in ("CollectionOrchestrator", "ExtractionService", "TrenderClient"):
        monkeypatch.setattr(cli, name, lambda: None)
    await cli.main(cli.parse_args(["--source-config", "TR English"]))

    names = _written(tmp_path)
    assert names[0].endswith("_backfill-track-record_TR-English.log")
    summary = json.loads((tmp_path / names[1]).read_text())
    assert summary["stats"]["calls_accepted"] == 1 and "items" not in summary["stats"]
