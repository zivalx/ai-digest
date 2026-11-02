"""
ExtractionService tests with a stubbed LLM (no real Claude call).
"""
import pytest

from src.services.extraction_service import ExtractionService, PickList, RawPick


@pytest.fixture(autouse=True)
def _fallback_chain_api_keys(monkeypatch):
    """
    Fake API keys for every provider.

    Only needed by the fallback-chain-building tests below (the stubbed-LLM
    tests above never touch the real provider construction path), but
    applying it to the whole module is harmless and keeps the fixture list
    simple.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")
    monkeypatch.setenv("XAI_API_KEY", "fake-xai-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")


class FakeStructuredLLM:
    """Stands in for `ChatAnthropic(...).with_structured_output(PickList)`."""

    def __init__(self, picklist: PickList):
        self.picklist = picklist
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self.picklist


@pytest.mark.asyncio
async def test_youtube_pick_builds_timestamped_url():
    picks = PickList(picks=[
        RawPick(
            symbol="nvda",  # lowercase on purpose -> normalized
            company_name="NVIDIA",
            thesis="Datacenter demand re-accelerates into Q3",
            stance="bullish",
            sentiment_score=0.7,
            confidence=0.8,
            start_seconds=1234,
            quote="NVIDIA is going to keep running",
        )
    ])
    svc = ExtractionService(structured_llm=FakeStructuredLLM(picks))

    item = {
        "source": "youtube",
        "video_id": "VIDEO",
        "channel": "@handle",
        "title": "Weekly picks",
        "transcript_segments": [
            {"start": 1200.0, "text": "Intro"},
            {"start": 1234.5, "text": "NVIDIA is going to keep running"},
        ],
    }

    mentions = await svc.extract([item])

    assert len(mentions) == 1
    m = mentions[0]
    assert m["symbol"] == "NVDA"
    assert m["company_name"] == "NVIDIA"
    assert m["source"] == "youtube"
    assert m["stance"] == "bullish"
    assert m["sentiment_score"] == 0.7
    assert m["confidence"] == 0.8
    assert "Datacenter demand" in m["context"]
    assert "NVIDIA is going to keep running" in m["context"]

    cit = m["citation"]
    assert cit["kind"] == "youtube"
    assert cit["url"] == "https://youtu.be/VIDEO?t=1234"
    assert cit["video_id"] == "VIDEO"
    assert cit["channel"] == "@handle"
    assert cit["start_seconds"] == 1234
    assert cit["quote"] == "NVIDIA is going to keep running"

    # dedupe key = symbol | url | start_seconds
    assert ExtractionService.dedupe_key_for(m) == "NVDA|https://youtu.be/VIDEO?t=1234|1234"

    # Time markers were passed to the model.
    human_msg = svc._structured_llm.calls[0][1].content
    assert "[1234s]" in human_msg


@pytest.mark.asyncio
async def test_low_confidence_and_bad_stance_dropped():
    picks = PickList(picks=[
        RawPick(symbol="AAPL", thesis="ok", stance="bullish",
                sentiment_score=0.5, confidence=0.9, start_seconds=10, quote="q"),
        RawPick(symbol="TSLA", thesis="weak", stance="bullish",
                sentiment_score=0.1, confidence=0.05, start_seconds=20, quote="q"),  # low conf
        RawPick(symbol="MSFT", thesis="bad", stance="mixed",
                sentiment_score=0.1, confidence=0.9, start_seconds=30, quote="q"),  # bad stance
    ])
    svc = ExtractionService(structured_llm=FakeStructuredLLM(picks))
    item = {"source": "youtube", "video_id": "V", "transcript": "text"}

    mentions = await svc.extract([item])

    assert [m["symbol"] for m in mentions] == ["AAPL"]


@pytest.mark.asyncio
async def test_item_without_transcript_skipped_no_llm_call():
    fake = FakeStructuredLLM(PickList(picks=[]))
    svc = ExtractionService(structured_llm=fake)

    mentions = await svc.extract([{"source": "youtube", "video_id": "V"}])

    assert mentions == []
    assert fake.calls == []  # skipped before any LLM call


@pytest.mark.asyncio
async def test_telegram_citation_uses_message_link():
    picks = PickList(picks=[
        RawPick(symbol="BTC", thesis="up only", stance="bullish",
                sentiment_score=0.6, confidence=0.7, quote="btc breaking out"),
    ])
    svc = ExtractionService(structured_llm=FakeStructuredLLM(picks))
    item = {
        "source": "telegram",
        "message_link": "https://t.me/chan/42",
        "channel": "@chan",
        "transcript": "btc breaking out to new highs",
        "ts": "2026-08-25T12:00:00Z",
    }

    mentions = await svc.extract([item])

    assert len(mentions) == 1
    m = mentions[0]
    assert m["source"] == "telegram"
    assert m["mentioned_at"] == "2026-08-25T12:00:00Z"
    cit = m["citation"]
    assert cit["kind"] == "telegram"
    assert cit["message_link"] == "https://t.me/chan/42"
    assert "url" not in cit
    assert ExtractionService.dedupe_key_for(m) == "BTC|https://t.me/chan/42|"


# ---------------------------------------------------------------------- #
# Fallback-chain construction (no structured_llm injected -> exercises the
# real provider-building path, but with fake API keys so there is no
# network call: constructing a LangChain chat model and calling
# .with_structured_output(...) never makes a request on its own).
# ---------------------------------------------------------------------- #

def test_fallback_chain_default_order_and_count(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDERS", raising=False)
    svc = ExtractionService()

    chain = svc.structured_llm

    assert chain.runnable.name == "extraction_provider_anthropic"
    assert [f.name for f in chain.fallbacks] == [
        "extraction_provider_xai",
        "extraction_provider_deepseek",
    ]


def test_fallback_chain_respects_custom_llm_providers_order(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "deepseek,anthropic")
    svc = ExtractionService()

    chain = svc.structured_llm

    assert chain.runnable.name == "extraction_provider_deepseek"
    assert [f.name for f in chain.fallbacks] == ["extraction_provider_anthropic"]


def test_fallback_chain_constructor_arg_overrides_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "anthropic,xai,deepseek")
    svc = ExtractionService(llm_providers="xai,deepseek")

    chain = svc.structured_llm

    assert chain.runnable.name == "extraction_provider_xai"
    assert [f.name for f in chain.fallbacks] == ["extraction_provider_deepseek"]


def test_fallback_chain_single_provider_has_no_fallbacks(monkeypatch):
    svc = ExtractionService(llm_providers="anthropic")

    chain = svc.structured_llm

    assert chain.runnable.name == "extraction_provider_anthropic"
    assert list(chain.fallbacks) == []


def test_fallback_chain_invalid_provider_raises_immediately():
    svc = ExtractionService(llm_providers="anthropic,not-a-real-provider")

    with pytest.raises(ValueError, match="Unsupported provider in LLM_PROVIDERS"):
        _ = svc.structured_llm


def test_fallback_chain_empty_providers_raises():
    svc = ExtractionService(llm_providers="   ,  ")

    with pytest.raises(ValueError, match="LLM_PROVIDERS must contain at least one provider"):
        _ = svc.structured_llm


@pytest.mark.asyncio
async def test_fallback_chain_falls_through_on_primary_error(monkeypatch, caplog):
    """
    with_fallbacks should catch a primary-provider error and try the next
    provider; the wrapper should log which provider ultimately served the
    result.
    """
    import logging

    class _BoomLLM:
        def with_structured_output(self, schema, method=None):
            async def _boom(_messages):
                raise RuntimeError("simulated anthropic rate limit")

            class _Boom:
                ainvoke = staticmethod(_boom)

            return _Boom()

    class _OkLLM:
        def __init__(self, result):
            self._result = result

        def with_structured_output(self, schema, method=None):
            async def _ok(_messages):
                return self._result

            class _Ok:
                ainvoke = staticmethod(_ok)

            return _Ok()

    picks = PickList(picks=[
        RawPick(symbol="AAPL", thesis="ok", stance="bullish",
                sentiment_score=0.5, confidence=0.9, start_seconds=10, quote="q"),
    ])

    def fake_build_chat_model(provider, model=None, temperature=0):
        if provider == "anthropic":
            return _BoomLLM()
        return _OkLLM(picks)

    monkeypatch.setattr(
        "src.services.llm_factory.build_chat_model", fake_build_chat_model
    )
    monkeypatch.setenv("LLM_PROVIDERS", "anthropic,xai")

    svc = ExtractionService()

    with caplog.at_level(logging.INFO):
        result = await svc.structured_llm.ainvoke([])

    assert result is picks
    assert any("Extraction served by provider=xai" in rec.message for rec in caplog.records)
