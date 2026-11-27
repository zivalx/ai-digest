"""
ExtractionService tests with a stubbed LLM (no real Claude call).
"""
import pytest

from src.services.extraction_service import EXTRACTION_VERSION, ExtractionService, PickList, RawPick


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


def test_fallback_chain_skips_provider_without_api_key(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    svc = ExtractionService(llm_providers="anthropic,xai,deepseek")

    chain = svc.structured_llm

    assert chain.runnable.name == "extraction_provider_anthropic"
    assert [f.name for f in chain.fallbacks] == ["extraction_provider_deepseek"]


def test_fallback_chain_raises_when_no_provider_has_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    svc = ExtractionService(llm_providers="anthropic,xai")

    with pytest.raises(ValueError, match="No extraction provider has an API key set"):
        _ = svc.structured_llm


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


# ---------------------------------------------------------------------- #
# Recommendation fields / track-record payload
# ---------------------------------------------------------------------- #
def _rec_pick(**over):
    base = dict(
        symbol="NVDA", company_name="NVIDIA", thesis="Breakout setup",
        stance="bullish", sentiment_score=0.6, confidence=0.8,
        start_seconds=60, quote="I'm buying NVDA here",
        is_recommendation=True, action="buy", horizon="short",
        entry_price=100.0, target_price=120.0, stop_price=95.0, trigger_price=None,
    )
    base.update(over)
    return RawPick(**base)


_YT_ITEM = {
    "source": "youtube", "video_id": "VID1", "channel": "@micha", "title": "T",
    "transcript": "I'm buying NVDA here",
    "ts": "2026-10-01T12:00:00Z", "ts_precision": "datetime",
}


async def _extract_one(pick, item=None):
    svc = ExtractionService(structured_llm=FakeStructuredLLM(PickList(picks=[pick])))
    return await svc.extract([item or _YT_ITEM])


@pytest.mark.asyncio
async def test_recommendation_fields_pass_through():
    (m,) = await _extract_one(_rec_pick())
    assert m["is_recommendation"] is True
    assert m["action"] == "buy"
    assert m["horizon"] == "short"
    assert (m["entry_price"], m["target_price"], m["stop_price"], m["trigger_price"]) == (
        100.0, 120.0, 95.0, None)
    assert m["thesis"] == "Breakout setup"
    assert m["quote"] == "I'm buying NVDA here"
    assert m["said_at"] == "2026-10-01T12:00:00Z"
    assert m["said_at_precision"] == "datetime"


@pytest.mark.asyncio
async def test_said_at_has_no_now_fallback():
    item = {k: v for k, v in _YT_ITEM.items() if k not in ("ts", "ts_precision")}
    (m,) = await _extract_one(_rec_pick(), item)
    assert m["said_at"] is None
    assert m["said_at_precision"] is None
    assert m["mentioned_at"]  # discovery fallback unchanged


@pytest.mark.asyncio
async def test_invalid_action_downgrades_to_mention():
    (m,) = await _extract_one(_rec_pick(action="moon"))
    assert m["symbol"] == "NVDA"  # still goes to discovery
    assert m["is_recommendation"] is False
    assert m["action"] is None


@pytest.mark.asyncio
async def test_invalid_horizon_nulls_only_horizon():
    # trender defaults the horizon from the influencer's style.
    (m,) = await _extract_one(_rec_pick(horizon="forever"))
    assert m["horizon"] is None
    assert m["is_recommendation"] is True and m["action"] == "buy"
    assert ExtractionService.to_call_payload(m, "feeder", "x")["horizon"] is None


@pytest.mark.asyncio
async def test_action_is_normalised():
    (m,) = await _extract_one(_rec_pick(action=" BUY "))
    assert m["action"] == "buy" and m["is_recommendation"] is True


@pytest.mark.asyncio
async def test_plain_pick_defaults_to_mention():
    pick = RawPick(symbol="AAPL", thesis="t", stance="neutral", sentiment_score=0,
                   confidence=0.7, quote="q")
    (m,) = await _extract_one(pick)
    assert m["is_recommendation"] is False and m["action"] is None


@pytest.mark.asyncio
async def test_to_call_payload_none_for_mentions():
    pick = _rec_pick(is_recommendation=False)
    (m,) = await _extract_one(pick)
    assert ExtractionService.to_call_payload(m, "feeder", "claude-x") is None


@pytest.mark.asyncio
async def test_to_call_payload_none_without_said_at():
    item = {k: v for k, v in _YT_ITEM.items() if k not in ("ts", "ts_precision")}
    (m,) = await _extract_one(_rec_pick(), item)
    assert ExtractionService.to_call_payload(m, "feeder", "claude-x") is None


@pytest.mark.asyncio
async def test_to_call_payload_none_without_source_item_id_or_channel():
    (m,) = await _extract_one(_rec_pick())
    no_id = {**m, "citation": {**m["citation"], "video_id": None}}
    no_channel = {**m, "citation": {**m["citation"], "channel": None}}
    assert ExtractionService.to_call_payload(no_id, "feeder", "x") is None
    assert ExtractionService.to_call_payload(no_channel, "feeder", "x") is None


@pytest.mark.asyncio
async def test_to_call_payload_shape():
    (m,) = await _extract_one(_rec_pick())
    p = ExtractionService.to_call_payload(m, "feeder", "claude-x")
    assert set(p) == {
        "symbol", "company_name", "source", "action", "stance", "horizon", "conviction",
        "entry_price", "target_price", "stop_price", "trigger_price", "thesis", "quote",
        "citation", "source_item_id", "said_at", "said_at_precision", "origin",
        "extractor_model", "extraction_version",
    }
    assert p["extraction_version"] == EXTRACTION_VERSION == "tr-2"
    assert p["conviction"] == 0.8
    assert p["source_item_id"] == "VID1"
    assert p["citation"]["channel"] == "@micha"
    assert p["origin"] == "feeder" and p["extractor_model"] == "claude-x"
    assert p["said_at"] == "2026-10-01T12:00:00Z" and p["said_at_precision"] == "datetime"
    assert p["action"] == "buy" and p["horizon"] == "short" and p["stance"] == "bullish"


@pytest.mark.asyncio
async def test_to_call_payload_telegram_and_precision_default():
    item = {"source": "telegram", "message_link": "https://t.me/c/1", "channel": "noam",
            "text": "מוכר", "ts": "2026-10-01T00:00:00Z"}
    (m,) = await _extract_one(_rec_pick(action="sell"), item)
    p = ExtractionService.to_call_payload(m, "backfill", "m")
    assert p["source_item_id"] == "https://t.me/c/1"
    assert p["said_at_precision"] == "datetime"  # missing precision -> datetime
    assert p["source"] == "telegram"


def test_system_prompt_covers_recommendations():
    from src.services.extraction_service import SYSTEM_PROMPT
    for needle in ("is_recommendation", "watch", "אני קונה", "מוכר", "אם תפרוץ", "false"):
        assert needle in SYSTEM_PROMPT


# Contract with trender's ``CallIn`` (trender: app/api/track_record.py, enums in
# app/track_record/models.py). Hard-coded on purpose: a drift here means the
# ingest endpoint would 422 the whole batch.
CALLIN_KEYS = {
    "symbol", "company_name", "source", "action", "stance", "horizon", "conviction",
    "entry_price", "target_price", "stop_price", "trigger_price", "thesis", "quote",
    "citation", "source_item_id", "said_at", "said_at_precision", "origin",
    "extractor_model", "extraction_version",
}
CALLIN_ACTIONS = {"buy", "add", "hold", "trim", "sell", "short", "avoid", "watch"}  # CallAction
CALLIN_HORIZONS = {"short", "medium", "long"}  # Horizon
CALLIN_SOURCES = {"youtube", "telegram", "twitter"}
CALLIN_PRECISIONS = {"datetime", "date"}
CALLIN_ORIGINS = {"feeder", "backfill"}


def test_extraction_enums_are_subsets_of_trender_callin():
    from src.services.extraction_service import _ACTIONS, _HORIZONS
    assert _ACTIONS <= CALLIN_ACTIONS
    assert _HORIZONS <= CALLIN_HORIZONS


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", sorted(CALLIN_ORIGINS))
async def test_to_call_payload_matches_trender_callin_contract(origin):
    (yt,) = await _extract_one(_rec_pick())
    (tg,) = await _extract_one(_rec_pick(action="sell", horizon=None), {
        "source": "telegram", "message_link": "https://t.me/c/1", "channel": "noam",
        "text": "x", "ts": "2026-10-01T00:00:00Z", "ts_precision": "date"})
    for m in (yt, tg):
        p = ExtractionService.to_call_payload(m, origin, "claude-x")
        assert set(p) == CALLIN_KEYS
        assert p["action"] in CALLIN_ACTIONS
        assert p["horizon"] is None or p["horizon"] in CALLIN_HORIZONS
        assert p["source"] in CALLIN_SOURCES
        assert p["said_at_precision"] in CALLIN_PRECISIONS
        assert p["origin"] in CALLIN_ORIGINS
        assert 0 <= p["conviction"] <= 1
        assert p["citation"].get("channel")


class _ScriptedLLM:
    """Raises the scripted exceptions in order; returns an empty PickList for None."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else None
        if outcome is not None:
            raise outcome
        return PickList(picks=[])


class _AuthError(Exception):
    status_code = 401


def _items(n):
    return [{"source": "youtube", "video_id": f"V{i}", "transcript": "talk"} for i in range(n)]


@pytest.mark.asyncio
async def test_bad_api_key_stops_batch_after_first_call():
    llm = _ScriptedLLM([_AuthError("invalid x-api-key")] * 10)
    svc = ExtractionService(structured_llm=llm)
    failed = []

    await svc.extract(_items(10), on_error=lambda it, e: failed.append(it["video_id"]))

    assert llm.calls == 1  # no more calls once the key is known bad
    assert failed == [f"V{i}" for i in range(10)]  # all reported, so all retried later
    assert "invalid x-api-key" in svc.aborted_reason


@pytest.mark.asyncio
async def test_three_failures_in_a_row_stop_batch_but_isolated_ones_do_not():
    boom = RuntimeError("overloaded")
    llm = _ScriptedLLM([boom, None, boom, boom, boom, None, None])
    svc = ExtractionService(structured_llm=llm)
    failed = []

    await svc.extract(_items(7), on_error=lambda it, e: failed.append(it["video_id"]))

    assert llm.calls == 5  # V0 fails, V1 ok (resets), V2-V4 fail -> stop
    assert failed == ["V0", "V2", "V3", "V4", "V5", "V6"]
    assert svc.aborted_reason.startswith("stopped after 3 failed LLM call(s)")


@pytest.mark.asyncio
async def test_successful_batch_has_no_abort_reason():
    svc = ExtractionService(structured_llm=_ScriptedLLM([]))
    await svc.extract(_items(2))
    assert svc.aborted_reason is None


@pytest.mark.asyncio
async def test_failed_llm_call_is_reported_by_position():
    """The scheduler keeps failed items out of its ledger so the next run retries them."""

    class FlakyLLM:
        def __init__(self):
            self.n = 0

        async def ainvoke(self, messages):
            self.n += 1
            if self.n == 2:
                raise RuntimeError("rate limited")
            return PickList(picks=[])

    svc = ExtractionService(structured_llm=FlakyLLM())
    items = [{"source": "youtube", "video_id": f"V{i}", "transcript": "text"} for i in range(3)]
    assert await svc.extract(items) == []
    assert svc.last_failed_indexes == [1]

    # A second run starts from a clean slate.
    assert await svc.extract([]) == []
    assert svc.last_failed_indexes == []


def test_prompt_and_schema_tell_the_model_to_suffix_tel_aviv_tickers():
    """Incident 2026-11-27: Tel Aviv stocks were extracted as bare AYAL/ELAL and rejected."""
    from src.services.extraction_service import SYSTEM_PROMPT

    assert ".TA" in SYSTEM_PROMPT and "AYAL.TA" in SYSTEM_PROMPT
    assert "TEVA.TA" in SYSTEM_PROMPT  # dual listing: use the one actually discussed
    assert "no suffix" in SYSTEM_PROMPT.lower()
    assert ".TA" in RawPick.model_fields["symbol"].description


@pytest.mark.asyncio
async def test_ta_suffixed_symbol_survives_extraction_and_reaches_the_call_payload():
    item = {
        "source": "telegram", "message_link": "https://t.me/ch/9", "channel": "ch",
        "text": "קונה איילון", "ts": "2026-10-02", "ts_precision": "date",
    }

    class _LLM:
        async def ainvoke(self, messages):
            return PickList(picks=[RawPick(
                symbol="ayal.ta", thesis="t", stance="bullish", sentiment_score=0.5,
                confidence=0.8, quote="קונה איילון", is_recommendation=True, action="buy")])

    (mention,) = await ExtractionService(structured_llm=_LLM()).extract([item])
    assert mention["symbol"] == "AYAL.TA"
    assert ExtractionService.to_call_payload(mention, "feeder", "m")["symbol"] == "AYAL.TA"
