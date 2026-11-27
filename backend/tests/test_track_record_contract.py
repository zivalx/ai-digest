"""
Contract with trender's ``POST /api/track-record/calls`` (trender: ``CallsBatchRequest``
/ ``CallIn`` in app/api/track_record.py).

``fixtures/track_record_calls_contract.json`` is committed byte-for-byte in both repos
(trender: tests/test_track_record/fixtures/). This side proves the feeder produces exactly
that payload; trender's contract test proves its endpoint accepts it. Change the payload
shape on both sides together, then update the file and FIXTURE_SHA256 in both repos.
"""
import hashlib
import json
from pathlib import Path

import pytest

from src.services.extraction_service import EXTRACTION_VERSION, ExtractionService, PickList, RawPick
from src.services.trender_client import TrenderClient

FIXTURE = Path(__file__).parent / "fixtures" / "track_record_calls_contract.json"
FIXTURE_SHA256 = "eb17cb698568f68d54acf211515c5bd683ff69b9084a4c5d0d8e4aea3d0b8a60"


class _OneShotLLM:
    def __init__(self, pick: RawPick):
        self.pick = pick

    async def ainvoke(self, messages):
        return PickList(picks=[self.pick])


def _pick(**over) -> RawPick:
    base = dict(
        symbol="NVDA", company_name="NVIDIA", thesis="Breakout setup", stance="bullish",
        sentiment_score=0.6, confidence=0.8, start_seconds=60, quote="I'm buying NVDA here",
        is_recommendation=True, action="buy", horizon="short",
        entry_price=100.0, target_price=120.0, stop_price=95.0, trigger_price=None,
    )
    base.update(over)
    return RawPick(**base)


YOUTUBE_ITEM = {
    "source": "youtube", "video_id": "dQw4w9WgXcQ", "channel": "Micha Stocks",
    "title": "NVDA breakout", "transcript": "I'm buying NVDA here",
    "ts": "2026-10-01T12:00:00Z", "ts_precision": "datetime",
}
TELEGRAM_ITEM = {
    "source": "telegram", "message_link": "https://t.me/noamstocks/1234", "channel": "noamstocks",
    "text": "מוכר את AAPL", "ts": "2026-10-02", "ts_precision": "date",
}
TELEGRAM_PICK = dict(
    symbol="AAPL", company_name="Apple", action="sell", stance="bearish", horizon=None,
    confidence=0.65, entry_price=None, target_price=None, stop_price=None, trigger_price=None,
    thesis="Taking profits", quote="מוכר את AAPL", start_seconds=None,
)


async def _payload(item, pick) -> dict:
    (mention,) = await ExtractionService(structured_llm=_OneShotLLM(pick)).extract([item])
    return ExtractionService.to_call_payload(mention, "feeder", "claude-sonnet-5")


def test_fixture_is_the_shared_file():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256


@pytest.mark.asyncio
async def test_feeder_builds_exactly_the_contract_payload():
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    built = [
        await _payload(YOUTUBE_ITEM, _pick()),
        await _payload(TELEGRAM_ITEM, _pick(**TELEGRAM_PICK)),
    ]
    # Round-trip through JSON, as call_payload_json stores it and httpx sends it.
    sent = json.loads(json.dumps({"calls": built}))
    # extraction_version is a free-form label that changes with the prompt (the shape is
    # what the contract pins), so the shared fixture keeps its original value.
    assert {c.pop("extraction_version") for c in sent["calls"]} == {EXTRACTION_VERSION}
    for call in sent["calls"]:
        call["extraction_version"] = "tr-1"
    assert sent == expected


@pytest.mark.asyncio
async def test_missing_confidence_sends_trenders_default_conviction():
    """trender's CallIn.conviction is a float in [0, 1] (default 0.5); null would 422 the batch."""
    mention = (await ExtractionService(structured_llm=_OneShotLLM(_pick())).extract([YOUTUBE_ITEM]))[0]
    payload = ExtractionService.to_call_payload({**mention, "confidence": None}, "feeder", "m")
    assert payload["conviction"] == 0.5


@pytest.mark.asyncio
async def test_client_posts_the_batch_to_the_track_record_endpoint_with_the_ingest_token():
    import httpx

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"accepted": 2, "duplicates": 0, "rejected": []})

    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = TrenderClient(base_url="http://trender-app:8000", ingest_token="tok", client=http)
        await client.push_calls(expected["calls"])
    assert seen["url"] == "http://trender-app:8000/api/track-record/calls"
    assert seen["auth"] == "Bearer tok"
    assert seen["body"] == expected

