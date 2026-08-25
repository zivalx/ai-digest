"""
TrenderClient tests against a mocked httpx transport (no real network).
"""
import httpx
import pytest

from src.services.trender_client import TrenderClient, TrenderPushError

MENTIONS = [{"symbol": "NVDA", "source": "youtube", "citation": {"kind": "youtube"}}]


def _client_with(handler, **kwargs):
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    return TrenderClient(base_url="http://trender.test", client=http, backoff_base=0.0, **kwargs)


@pytest.mark.asyncio
async def test_push_success():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"accepted": 1, "rejected": 0})

    client = _client_with(handler, ingest_token=None)
    resp = await client.push_mentions(MENTIONS)

    assert resp == {"accepted": 1, "rejected": 0}
    assert captured["url"] == "http://trender.test/api/discovery/mentions"
    assert captured["auth"] is None  # no token -> no auth header


@pytest.mark.asyncio
async def test_push_sends_auth_header_when_token_set():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"ok": True})

    client = _client_with(handler, ingest_token="secret-token")
    await client.push_mentions(MENTIONS)

    assert captured["auth"] == "Bearer secret-token"


@pytest.mark.asyncio
async def test_push_retries_on_network_failure_then_succeeds():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, json={"ok": True})

    client = _client_with(handler, ingest_token=None, max_retries=2)
    resp = await client.push_mentions(MENTIONS)

    assert resp == {"ok": True}
    assert attempts["n"] == 3  # 2 failures + 1 success


@pytest.mark.asyncio
async def test_push_raises_after_exhausting_retries():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    client = _client_with(handler, ingest_token=None, max_retries=1)
    with pytest.raises(TrenderPushError):
        await client.push_mentions(MENTIONS)


@pytest.mark.asyncio
async def test_push_http_error_raises_without_retry():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(422, json={"detail": "bad payload"})

    client = _client_with(handler, ingest_token=None, max_retries=3)
    with pytest.raises(TrenderPushError):
        await client.push_mentions(MENTIONS)
    assert attempts["n"] == 1  # 4xx is not retried


@pytest.mark.asyncio
async def test_empty_batch_short_circuits():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("should not POST for empty batch")

    client = _client_with(handler)
    resp = await client.push_mentions([])
    assert resp == {"pushed": 0, "response": None}
