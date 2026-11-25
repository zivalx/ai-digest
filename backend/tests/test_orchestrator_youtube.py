"""YouTube collection settings that matter on a small server (no network, collector faked)."""
import pytest

pytest.importorskip("connectors.youtube")

import connectors.youtube as yt  # noqa: E402

from src.services.collection_orchestrator import CollectionOrchestrator  # noqa: E402


class FakeCollector:
    last = None

    def __init__(self, config):
        FakeCollector.config = config

    async def fetch(self, spec):
        FakeCollector.spec = spec
        return []


@pytest.fixture(autouse=True)
def _fake_collector(monkeypatch):
    monkeypatch.setattr(yt, "YouTubeCollector", FakeCollector)


async def _collect(spec):
    await CollectionOrchestrator()._collect_youtube("YOUTUBE", {"channels": ["@a"], **spec})
    return FakeCollector.config


@pytest.mark.asyncio
async def test_whisper_default_is_unchanged(monkeypatch):
    monkeypatch.delenv("YOUTUBE_WHISPER_MODEL", raising=False)
    assert (await _collect({})).whisper_model == "base"


@pytest.mark.asyncio
async def test_whisper_can_be_switched_off_by_env(monkeypatch):
    """An empty value means: no Whisper model is loaded (captions only)."""
    monkeypatch.setenv("YOUTUBE_WHISPER_MODEL", "")
    assert not (await _collect({})).whisper_model


@pytest.mark.asyncio
async def test_whisper_env_picks_a_smaller_model(monkeypatch):
    monkeypatch.setenv("YOUTUBE_WHISPER_MODEL", "tiny")
    assert (await _collect({})).whisper_model == "tiny"


@pytest.mark.asyncio
async def test_collect_spec_wins_over_env(monkeypatch):
    monkeypatch.setenv("YOUTUBE_WHISPER_MODEL", "tiny")
    assert (await _collect({"whisper_model": "small"})).whisper_model == "small"


# ---- telegram session location -------------------------------------------------------

import connectors.telegram as tg  # noqa: E402


class FakeTelegramCollector:
    def __init__(self, config):
        FakeTelegramCollector.config = config

    async def fetch(self, spec):
        return []


async def _telegram_config(monkeypatch):
    monkeypatch.setattr(tg, "TelegramCollector", FakeTelegramCollector)
    for suffix, value in (("API_ID", "123"), ("API_HASH", "hash"), ("PHONE", "+10000000")):
        monkeypatch.setenv(f"TGREF_{suffix}", value)
    await CollectionOrchestrator()._collect_telegram("TGREF", {"channels": ["@c"]})
    return FakeTelegramCollector.config


@pytest.mark.asyncio
async def test_telegram_session_defaults_to_the_library_name(monkeypatch):
    monkeypatch.delenv("TELEGRAM_SESSION_PATH", raising=False)
    assert (await _telegram_config(monkeypatch)).session_name == "connector_session"


@pytest.mark.asyncio
async def test_telegram_session_path_can_point_at_a_volume(monkeypatch):
    monkeypatch.setenv("TELEGRAM_SESSION_PATH", "/data/telegram")
    assert (await _telegram_config(monkeypatch)).session_name == "/data/telegram"
