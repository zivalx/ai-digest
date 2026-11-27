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
