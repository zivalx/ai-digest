"""
TrenderClient tests against a mocked httpx transport (no real network).
"""
import json

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


CALLS = [{"symbol": "NVDA", "action": "buy", "origin": "feeder"}]


@pytest.mark.asyncio
async def test_push_calls_url_and_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"accepted": 1, "duplicates": 0, "rejected": []})

    client = _client_with(handler, ingest_token="tok")
    resp = await client.push_calls(CALLS)

    assert resp == {"accepted": 1, "duplicates": 0, "rejected": []}
    assert captured["url"] == "http://trender.test/api/track-record/calls"
    assert captured["body"] == {"calls": CALLS}
    assert captured["auth"] == "Bearer tok"


@pytest.mark.asyncio
async def test_push_calls_retry_then_success():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, json={"accepted": 1, "duplicates": 0, "rejected": []})

    client = _client_with(handler, ingest_token=None, max_retries=2)
    resp = await client.push_calls(CALLS)

    assert resp["accepted"] == 1
    assert attempts["n"] == 3


@pytest.mark.asyncio
async def test_push_retries_5xx_then_succeeds():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json={"accepted": 1, "duplicates": 0, "rejected": []})

    client = _client_with(handler, ingest_token=None, max_retries=2)
    resp = await client.push_calls(CALLS)

    assert resp["accepted"] == 1
    assert attempts["n"] == 3  # 2x 503 + 1 success


@pytest.mark.asyncio
async def test_push_raises_after_persistent_5xx():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(500, text="boom")

    client = _client_with(handler, ingest_token=None, max_retries=1)
    with pytest.raises(TrenderPushError, match="500"):
        await client.push_calls(CALLS)
    assert attempts["n"] == 2


# --- errors that say what happened; configurable timeout --------------------------------------

@pytest.mark.asyncio
async def test_timeout_error_names_the_exception_the_wait_the_endpoint_and_batch_size(caplog):
    """Incident 2026-11-27: the log said 'Push to trender failed (attempt 1/3): ;' (empty)."""
    import logging

    def handler(request):
        raise httpx.ReadTimeout("", request=request)  # httpx leaves the message empty

    client = _client_with(handler, timeout=120)
    with caplog.at_level(logging.WARNING, logger="src.services.trender_client"):
        with pytest.raises(TrenderPushError) as exc:
            await client.push_calls([{"symbol": "AYAL.TA"}, {"symbol": "ELAL.TA"}])

    assert "ReadTimeout after 120s" in str(exc.value)
    assert "http://trender.test/api/track-record/calls" in str(exc.value)
    assert "2 call(s)" in str(exc.value)
    text = caplog.text
    assert "ReadTimeout after 120s" in text and "2 call(s)" in text
    assert "/api/track-record/calls" in text and "attempt 1/3" in text
    assert "failed (attempt 1/3): ;" not in text


@pytest.mark.asyncio
async def test_connect_error_with_message_keeps_type_and_message():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(TrenderPushError) as exc:
        await _client_with(handler).push_mentions(MENTIONS)
    assert "ConnectError: connection refused" in str(exc.value)
    assert "1 mention(s)" in str(exc.value)


def test_timeout_comes_from_settings_and_an_explicit_arg_wins(monkeypatch):
    monkeypatch.delenv("TRENDER_TIMEOUT_SECONDS", raising=False)
    assert TrenderClient(base_url="http://x").timeout == 120.0
    monkeypatch.setenv("TRENDER_TIMEOUT_SECONDS", "45")
    assert TrenderClient(base_url="http://x").timeout == 45.0
    assert TrenderClient(base_url="http://x", timeout=5).timeout == 5
    monkeypatch.setenv("TRENDER_TIMEOUT_SECONDS", "junk")
    assert TrenderClient(base_url="http://x").timeout == 120.0


def test_push_batch_size_and_log_dir_settings(monkeypatch):
    from src.config import get_settings

    monkeypatch.delenv("TRENDER_PUSH_BATCH_SIZE", raising=False)
    monkeypatch.delenv("LOG_DIR", raising=False)
    s = get_settings()
    assert s.trender_push_batch_size == 50
    assert s.log_dir.endswith("backend/logs")
    monkeypatch.setenv("TRENDER_PUSH_BATCH_SIZE", "10")
    monkeypatch.setenv("LOG_DIR", "/data/logs")
    s = get_settings()
    assert s.trender_push_batch_size == 10 and s.log_dir == "/data/logs"
    monkeypatch.setenv("TRENDER_PUSH_BATCH_SIZE", "0")
    assert get_settings().trender_push_batch_size == 50
