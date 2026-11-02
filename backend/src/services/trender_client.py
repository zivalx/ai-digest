"""
Trender Client - pushes extracted mentions and track-record calls to trender.

POSTs batches to ``{TRENDER_BASE_URL}/api/discovery/mentions`` and
``{TRENDER_BASE_URL}/api/track-record/calls``. Sends a bearer
token when ``TRENDER_INGEST_TOKEN`` is set (Hetzner); open locally. Retries a
couple of times on network failure or HTTP 5xx with exponential backoff.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx

from ..config import get_settings

logger = logging.getLogger(__name__)


class TrenderPushError(Exception):
    """Raised when the mentions batch cannot be pushed after retries."""


class TrenderClient:
    """Thin async httpx client for trender's discovery ingest endpoint."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        ingest_token: Optional[str] = None,
        max_retries: int = 2,
        backoff_base: float = 0.5,
        timeout: float = 30.0,
        client: Optional[httpx.AsyncClient] = None,
    ):
        settings = get_settings()
        self.base_url = (base_url or settings.trender_base_url).rstrip("/")
        # Explicit arg wins; otherwise fall back to env-derived token.
        self.ingest_token = ingest_token if ingest_token is not None else settings.trender_ingest_token
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.timeout = timeout
        self._client = client  # injectable for tests

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.ingest_token:
            headers["Authorization"] = f"Bearer {self.ingest_token}"
        return headers

    async def push_mentions(self, mentions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        POST a batch of mention dicts to trender.

        Returns the parsed JSON response. Retries on transport errors and 5xx;
        raises :class:`TrenderPushError` if all attempts fail.
        """
        if not mentions:
            logger.info("No mentions to push to trender")
            return {"pushed": 0, "response": None}

        return await self._post_with_retry(
            "/api/discovery/mentions", {"mentions": mentions}, f"{len(mentions)} mention(s)"
        )

    async def push_calls(self, calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        POST a batch of track-record call dicts to trender.

        Same retry/auth behaviour as :meth:`push_mentions`.
        """
        if not calls:
            logger.info("No calls to push to trender")
            return {"pushed": 0, "response": None}

        return await self._post_with_retry(
            "/api/track-record/calls", {"calls": calls}, f"{len(calls)} call(s)"
        )

    async def _post_with_retry(self, path: str, payload: Dict[str, Any], label: str) -> Dict[str, Any]:
        """POST ``payload`` to ``base_url + path``.

        Transport errors and HTTP 5xx are retried with exponential backoff; a 4xx
        is the server rejecting the batch and fails fast.
        """
        url = f"{self.base_url}{path}"

        last_error: Optional[str] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._post(url, payload)
                response.raise_for_status()
                data = response.json()
                logger.info("Pushed %s to trender: %s", label, url)
                return data
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_error = str(e)
            except httpx.HTTPStatusError as e:
                status, body = e.response.status_code, e.response.text[:500]
                if status < 500:
                    logger.error("trender rejected the batch: %s %s", status, body)
                    raise TrenderPushError(f"trender returned {status}: {body}") from e
                last_error = f"trender returned {status}: {body}"

            if attempt < self.max_retries:
                delay = self.backoff_base * (2 ** attempt)
                logger.warning(
                    "Push to trender failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt + 1,
                    self.max_retries + 1,
                    last_error,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error("Push to trender failed after %d attempts: %s", attempt + 1, last_error)

        raise TrenderPushError(f"Failed to push {label} after retries: {last_error}")

    async def _post(self, url: str, payload: Dict[str, Any]) -> httpx.Response:
        """Perform the POST, using an injected client if provided else a fresh one."""
        if self._client is not None:
            return await self._client.post(url, json=payload, headers=self._headers())
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await client.post(url, json=payload, headers=self._headers())
