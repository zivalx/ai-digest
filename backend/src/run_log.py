"""
Run logs: the feeder and the backfills write their own log file.

``run_log(kind, source_config)`` is a context manager. While it is open, everything logged
by the process also goes to ``LOG_DIR/<UTC time>_<kind>_<source>.log``; on exit a short
``<same name>.summary.json`` is written next to it with the counts per outcome and reason
of the calls pushed to trender. The operator never needs shell redirection.

Failing to write the files (bad LOG_DIR, full disk) is logged as a warning and never stops
the run. ``LOG_DIR`` defaults to ``backend/logs/`` (gitignored); on the server it is on the
data volume (see ``.env.example``).
"""
import json
import logging
import re
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from .config import get_settings

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
CALL_OUTCOMES = ("accepted", "duplicate", "accepted_or_duplicate", "rejected", "failed")

_current: ContextVar[Optional["RunLog"]] = ContextVar("current_run_log", default=None)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")[:60] or "all"


def _empty_calls() -> Dict[str, Any]:
    return {"accepted": 0, "duplicate": 0, "accepted_or_duplicate": 0, "rejected": {}, "failed": {}}


class RunLog:
    """One run's log file, summary file and call counters."""

    def __init__(self, kind: str, source_config: Optional[str], started_at: datetime):
        self.kind = kind
        self.source_config = source_config
        self.started_at = started_at
        self.log_path: Optional[Path] = None
        self.summary_path: Optional[Path] = None
        self.summary: Dict[str, Any] = {"calls": _empty_calls()}
        self._handler: Optional[logging.Handler] = None

    # -- counters ---------------------------------------------------------------------------
    def record_call(self, outcome: str, reason: Optional[str] = None) -> None:
        calls = self.summary["calls"]
        if outcome in ("rejected", "failed"):
            key = reason or "unknown"
            calls[outcome][key] = calls[outcome].get(key, 0) + 1
        else:
            calls[outcome] = calls.get(outcome, 0) + 1

    # -- files ------------------------------------------------------------------------------
    def _open(self, log_dir: Path) -> None:
        stamp = self.started_at.astimezone(timezone.utc).strftime("%Y%m%d-%H%M%S")
        base = f"{stamp}_{_slug(self.kind)}" + (f"_{_slug(self.source_config)}" if self.source_config else "")
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            path, n = log_dir / f"{base}.log", 1
            while path.exists():  # two runs in the same second must not share a file
                n += 1
                path = log_dir / f"{base}-{n}.log"
            handler = logging.FileHandler(path, encoding="utf-8")
        except OSError as exc:
            logger.warning("Run log disabled: cannot write to %s (%s)", log_dir, exc)
            return
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        handler.setLevel(logging.INFO)
        root = logging.getLogger()
        if root.getEffectiveLevel() > logging.INFO:
            root.setLevel(logging.INFO)
        root.addHandler(handler)
        self._handler, self.log_path = handler, path
        self.summary_path = path.with_suffix(".summary.json")

    def _close(self, status: str, error: Optional[str]) -> None:
        if self._handler is not None:
            logging.getLogger().removeHandler(self._handler)
            self._handler.close()
        if self.summary_path is None:
            return
        summary = {
            "kind": self.kind,
            "source_config": self.source_config,
            "status": status,
            "started_at": self.started_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "log_file": self.log_path.name,
            **({"error": error} if error else {}),
            **self.summary,
        }
        try:
            self.summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not write run summary %s (%s)", self.summary_path, exc)


@contextmanager
def run_log(
    kind: str,
    source_config: Optional[str] = None,
    *,
    log_dir: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> Iterator[RunLog]:
    """Log this run to a file in ``log_dir`` (default ``LOG_DIR``) and summarise it on exit."""
    run = RunLog(kind, source_config, now or datetime.now(timezone.utc))
    run._open(Path(log_dir or get_settings().log_dir))
    token = _current.set(run)
    try:
        yield run
    except BaseException as exc:
        logger.error("Run aborted: %s", "".join(traceback.format_exception(exc)).rstrip())
        _current.reset(token)
        run._close("error", f"{type(exc).__name__}: {exc}")
        raise
    else:
        _current.reset(token)
        run._close("ok", None)


def record_call(outcome: str, reason: Optional[str] = None) -> None:
    """Count one call outcome in the active run's summary (no-op outside a run)."""
    run = _current.get()
    if run is not None:
        run.record_call(outcome, reason)
