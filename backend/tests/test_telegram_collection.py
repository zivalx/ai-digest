"""
Telegram collection tests — credentials accept the *_PHONE_NUMBER spelling
used by older .env files, the session path is configurable, and the
connector's message_link passes through to the collected items.
"""
import sys
import types

import pytest

from src.services.collection_orchestrator import CollectionOrchestrator, CollectionError


class _Msg:
    def __init__(self, mid):
        self.mid = mid

    def model_dump(self):
        return {"message_id": self.mid, "channel": "noammadar", "date": None, "text": "hi",
                "message_link": f"https://t.me/noammadar/{self.mid}"}


@pytest.fixture
def fake_telegram(monkeypatch):
    captured = {}
    mod = types.ModuleType("connectors.telegram")

    class Config:
        def __init__(self, **kw):
            captured["config"] = kw

    class Spec:
        def __init__(self, channels, max_messages_per_channel, include_replies):
            self.channels = channels

    class Collector:
        def __init__(self, config):
            pass

        async def fetch(self, spec):
            return [_Msg(1), _Msg(2)]

    mod.TelegramClientConfig, mod.TelegramCollectSpec, mod.TelegramCollector = Config, Spec, Collector
    monkeypatch.setitem(sys.modules, "connectors.telegram", mod)
    return captured


@pytest.mark.asyncio
async def test_collect_accepts_phone_number_spelling_and_keeps_links(fake_telegram, monkeypatch):
    monkeypatch.setenv("TGTEST_API_ID", "123")
    monkeypatch.setenv("TGTEST_API_HASH", "abc")
    monkeypatch.delenv("TGTEST_PHONE", raising=False)
    monkeypatch.setenv("TGTEST_PHONE_NUMBER", "+100")
    monkeypatch.setenv("TGTEST_SESSION", "/tmp/tg_test")
    result = await CollectionOrchestrator()._collect_telegram("TGTEST", {"channels": ["noammadar"]})
    assert fake_telegram["config"]["phone"] == "+100"
    assert fake_telegram["config"]["session_name"] == "/tmp/tg_test"
    assert [i["message_link"] for i in result["data"]] == [
        "https://t.me/noammadar/1", "https://t.me/noammadar/2"]


@pytest.mark.asyncio
async def test_collect_missing_credentials_raises(fake_telegram, monkeypatch):
    for k in ("API_ID", "API_HASH", "PHONE", "PHONE_NUMBER"):
        monkeypatch.delenv(f"TGTEST2_{k}", raising=False)
    with pytest.raises(CollectionError):
        await CollectionOrchestrator()._collect_telegram("TGTEST2", {"channels": ["x"]})
