"""
ExtractionService tests with a stubbed LLM (no real Claude call).
"""
import pytest

from src.services.extraction_service import ExtractionService, PickList, RawPick


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
