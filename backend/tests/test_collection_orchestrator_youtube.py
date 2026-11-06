"""_collect_youtube passes throttling options through, guarding older connectors."""
import sys
import types
from unittest.mock import AsyncMock

import pytest

from src.services.collection_orchestrator import CollectionOrchestrator


def _install(monkeypatch, new_fields: bool):
    captured = {}

    class Cfg:
        def __init__(self, **kw):
            captured["cfg"] = kw
            if not new_fields and ({"max_concurrency", "delay_seconds", "cookies_file"} & kw.keys()):
                raise ValueError("extra fields not permitted")
            self.__dict__.update(kw)

    class Spec:
        def __init__(self, **kw):
            captured["spec"] = kw
            if not new_fields and "skip_video_ids" in kw:
                raise ValueError("extra fields not permitted")
            self.__dict__.update(kw)

    class Collector:
        def __init__(self, cfg): pass
        fetch = AsyncMock(return_value=[])

    mod = types.ModuleType("connectors.youtube")
    mod.YouTubeCollector, mod.YouTubeClientConfig, mod.YouTubeCollectSpec = Collector, Cfg, Spec
    monkeypatch.setitem(sys.modules, "connectors.youtube", mod)
    return captured


@pytest.mark.asyncio
async def test_passes_present_keys_only(monkeypatch):
    cap = _install(monkeypatch, new_fields=True)
    await CollectionOrchestrator()._collect_youtube("YT", {"channels": ["@a"]})
    assert not {"max_concurrency", "delay_seconds", "cookies_file"} & cap["cfg"].keys()
    assert "skip_video_ids" not in cap["spec"]


@pytest.mark.asyncio
async def test_passes_throttle_and_skip_ids(monkeypatch):
    cap = _install(monkeypatch, new_fields=True)
    await CollectionOrchestrator()._collect_youtube("YT", {
        "channels": ["@a"], "max_concurrency": 1, "delay_seconds": 6,
        "cookies_file": "/c.txt", "skip_video_ids": ["A", "B"],
    })
    assert (cap["cfg"]["max_concurrency"], cap["cfg"]["delay_seconds"],
            cap["cfg"]["cookies_file"]) == (1, 6, "/c.txt")
    assert cap["spec"]["skip_video_ids"] == ["A", "B"]


@pytest.mark.asyncio
async def test_falls_back_when_connectors_reject_new_fields(monkeypatch, caplog):
    _install(monkeypatch, new_fields=False)
    with caplog.at_level("WARNING"):
        result = await CollectionOrchestrator()._collect_youtube("YT", {
            "channels": ["@a"], "max_concurrency": 1, "skip_video_ids": ["A"],
        })
    assert result["data"] == []
    assert "falling back" in caplog.text
