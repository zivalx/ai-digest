"""
Trender Client - pushes extracted mentions to trender's discovery ingest endpoint.

POSTs a batch to ``{TRENDER_BASE_URL}/api/discovery/mentions``. Sends a bearer
token when ``TRENDER_INGEST_TOKEN`` is set (Hetzner); open locally. Retries a
couple of times on network failure with exponential backoff.
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

        Returns the parsed JSON response. Retries on transport/network errors;
        raises :class:`TrenderPushError` if all attempts fail.
        """
        if not mentions:
            logger.info("No mentions to push to trender")
            return {"pushed": 0, "response": None}

        url = f"{self.base_url}/api/discovery/mentions"
        payload = {"mentions": mentions}

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._post(url, payload)
                response.raise_for_status()
                data = response.json()
                logger.info("Pushed %d mention(s) to trender: %s", len(mentions), url)
                return data
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self.backoff_base * (2 ** attempt)
                    logger.warning(
                        "Push to trender failed (attempt %d/%d): %s; retrying in %.1fs",
                        attempt + 1,
                        self.max_retries + 1,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("Push to trender failed after %d attempts: %s", attempt + 1, e)
            except httpx.HTTPStatusError as e:
                # A 4xx/5xx is a server-side decision, not a transient network fault.
                logger.error(
                    "trender rejected the batch: %s %s",
                    e.response.status_code,
                    e.response.text[:500],
                )
                raise TrenderPushError(
                    f"trender returned {e.response.status_code}: {e.response.text[:500]}"
                ) from e

        raise TrenderPushError(f"Failed to push mentions after retries: {last_error}")

    async def _post(self, url: str, payload: Dict[str, Any]) -> httpx.Response:
        """Perform the POST, using an injected client if provided else a fresh one."""
        if self._client is not None:
            return await self._client.post(url, json=payload, headers=self._headers())
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await client.post(url, json=payload, headers=self._headers())
