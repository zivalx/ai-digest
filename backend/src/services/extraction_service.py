"""
Extraction Service - turns collected items into ticker/asset picks via an LLM.

Uses ai-digest's existing LangChain stack with ``.with_structured_output(...)``
so the model returns a constrained list of picks. Transcripts are passed WITH
time markers (derived from ``transcript_segments``) so the model can anchor
each pick to a ``start_seconds``, which we turn into a
``https://youtu.be/<id>?t=<seconds>`` deep-link.

Provider-agnostic fallback chain: rather than hardcoding Claude, the
structured-output runnable is built as a quality-first fallback chain across
providers configured via ``LLM_PROVIDERS`` (default ``"anthropic,xai,deepseek"``,
see ``src.config``/``src.services.llm_factory``). LangChain's
``Runnable.with_fallbacks`` is used to fall through to the next provider on
a rate limit, API error, or a failed/malformed structured-output attempt.

The service never invents symbols; malformed or low-confidence picks are dropped
and the dropped count is logged. Authoritative symbol validation happens
consumer-side in trender.
"""
import os
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Picks below this confidence are dropped before push.
MIN_CONFIDENCE = 0.3

_STANCES = {"bullish", "bearish", "neutral"}

# Track-record (recommendation) extraction contract version, sent with each call.
EXTRACTION_VERSION = "tr-1"

# Must match trender's CallIn enums; an invalid value would 422 the whole batch.
_ACTIONS = {"buy", "add", "hold", "trim", "sell", "short", "avoid", "watch"}
_HORIZONS = {"short", "medium", "long"}
_CALL_SOURCES = {"youtube", "telegram", "twitter"}

# Stop the whole batch after this many LLM failures in a row: something
# systemic (outage, quota, bad config) is wrong, retrying every item is waste.
MAX_CONSECUTIVE_FAILURES = 3


def is_fatal_llm_error(exc: Exception) -> bool:
    """True for errors that will fail every item: bad/missing key, no access, no credit."""
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(s in text for s in (
        "authenticationerror", "permissiondeniederror", "invalid x-api-key",
        "invalid api key", "credit balance is too low",
        # Claude Code CLI (LLM_PROVIDERS=claude_code)
        "not logged in", "please run /login", "usage limit", "limit reached",
    ))


# Default quality-first provider order for the extraction fallback chain.
DEFAULT_LLM_PROVIDERS = "anthropic,xai,deepseek"

# Providers whose LangChain chat model is OpenAI-compatible under the hood
# (xai/deepseek/openai all subclass langchain_openai's BaseChatOpenAI) --
# these should force ``method="function_calling"`` on
# ``with_structured_output`` since their default flipped to ``"json_schema"``,
# which isn't reliably supported by every OpenAI-compatible endpoint.
# Anthropic's ``with_structured_output`` already defaults to
# ``"function_calling"`` in the installed langchain-anthropic version, so
# passing it explicitly there too is a no-op that keeps all providers
# consistent and explicit.
STRUCTURED_OUTPUT_METHOD = "function_calling"


class RawPick(BaseModel):
    """One ticker/asset pick as returned by the LLM for a single item."""

    symbol: str = Field(description="Ticker symbol, uppercase, e.g. NVDA. No $ prefix.")
    company_name: Optional[str] = Field(
        default=None, description="Full company/asset name if stated, else null."
    )
    thesis: str = Field(description="One-sentence summary of the argument for this pick.")
    stance: str = Field(description="One of: bullish, bearish, neutral.")
    sentiment_score: float = Field(
        description="Sentiment from -1.0 (very bearish) to 1.0 (very bullish)."
    )
    confidence: float = Field(
        description="Your confidence this is a genuine, actionable pick, 0.0 to 1.0."
    )
    start_seconds: Optional[int] = Field(
        default=None,
        description="For video transcripts, the [Ns] time marker where the pick is "
        "discussed. Null if unknown or not a video.",
    )
    quote: str = Field(description="The exact sentence/quote supporting the pick.")
    is_recommendation: bool = Field(
        default=False,
        description="True only if the author makes an actionable call (see action). "
        "False for a mere mention/discussion. If unsure, false.",
    )
    action: Optional[str] = Field(
        default=None,
        description="One of: buy, add, hold, trim, sell, short, avoid, watch. "
        "Null unless is_recommendation is true.",
    )
    horizon: Optional[str] = Field(
        default=None,
        description="One of: short, medium, long -- only if the author states or "
        "clearly implies one, else null.",
    )
    entry_price: Optional[float] = Field(
        default=None, description="Entry/buy price the author STATED, else null.")
    target_price: Optional[float] = Field(
        default=None, description="Price target the author STATED, else null.")
    stop_price: Optional[float] = Field(
        default=None, description="Stop-loss price the author STATED, else null.")
    trigger_price: Optional[float] = Field(
        default=None,
        description="For action=watch: the level that triggers the call "
        "(e.g. breakout above X), else null.",
    )


class PickList(BaseModel):
    """Structured-output container: all picks extracted from one item."""

    picks: List[RawPick] = Field(default_factory=list)


SYSTEM_PROMPT = (
    "You are a financial-content analyst. You are given the transcript or message "
    "text from a finance-focused channel. Extract every concrete stock/ETF/crypto "
    "ticker pick the author expresses an opinion on.\n\n"
    "Rules:\n"
    "- Only include picks that resolve to a real, plausible ticker symbol "
    "(uppercase letters, optionally with a market suffix). Never invent a symbol.\n"
    "- If the author only mentions a company by name, put the name in company_name "
    "and your best-guess ticker in symbol; if you cannot map it, omit the pick.\n"
    "- stance must be exactly one of: bullish, bearish, neutral.\n"
    "- sentiment_score in [-1, 1]; confidence in [0, 1].\n"
    "- Each transcript line may be prefixed with a [Ns] time marker. Set "
    "start_seconds to the marker nearest where the pick is discussed.\n"
    "- quote must be a verbatim snippet from the text.\n"
    "- If there are no genuine picks, return an empty list.\n\n"
    "Recommendations vs mentions:\n"
    "- Set is_recommendation=true ONLY when the author makes an actionable call on "
    "the ticker: says they buy/bought/hold/sell/sold it, or recommends doing so, or "
    "gives a conditional trade plan. Merely discussing a stock (its earnings, why it "
    "moved today, a chart description, news) is a mention: is_recommendation=false "
    "and action=null.\n"
    "- If unsure, is_recommendation=false.\n"
    "- action meanings: buy = recommends buying or says they bought (opens a long); "
    "add = adding to an existing position; hold = still holding / still likes it; "
    "trim = partially selling; sell = exits or recommends selling; short = "
    "recommends shorting / expects a drop; avoid = explicitly says stay away; "
    "watch = conditional call, e.g. \"if it breaks above X it's a buy\" "
    "(common in technical analysis; put X in trigger_price).\n"
    "- Hebrew examples: \"אני קונה\" / \"נכנסתי לפוזיציה\" = buy; \"מוכר\" / "
    "\"יצאתי\" = sell; \"אם תפרוץ מעל X\" = watch (trigger_price=X). A sentence "
    "like \"מניית אנבידיה עלתה היום ב-5%\" (the stock rose 5% today) is only a "
    "mention.\n"
    "- horizon (short, medium or long) only if stated or clearly implied, else null.\n"
    "- entry_price, target_price, stop_price, trigger_price: ONLY prices the author "
    "explicitly states; never estimate or infer them. Otherwise null.\n"
    "- Return one pick per symbol per item: the author's most decisive statement "
    "about it. The quote must be that statement."
)


class ExtractionService:
    """
    Extract ticker picks from collected items using an LLM fallback chain.

    Args:
        model: Explicit Anthropic model id override, used only for the
            ``anthropic`` provider in the chain (default from env
            ``ANTHROPIC_MODEL`` / ``claude-sonnet-5``). Other providers use
            their own env-configured defaults -- see
            ``src.services.llm_factory``.
        structured_llm: Optional pre-built runnable exposing ``ainvoke(messages)``
            returning a :class:`PickList`. Injected in tests to avoid a real call;
            built lazily as a multi-provider fallback chain on first use when
            not provided.
        min_confidence: Picks below this confidence are dropped.
        llm_providers: Comma-separated provider order for the fallback chain.
            Defaults to env ``LLM_PROVIDERS`` / ``"anthropic,xai,deepseek"``.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        structured_llm: Any = None,
        min_confidence: Optional[float] = None,
        temperature: Optional[float] = None,
        llm_providers: Optional[str] = None,
    ):
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
        self.min_confidence = (
            min_confidence
            if min_confidence is not None
            else float(os.getenv("EXTRACTION_MIN_CONFIDENCE", str(MIN_CONFIDENCE)))
        )
        self.temperature = temperature
        self.llm_providers = llm_providers or os.getenv("LLM_PROVIDERS", DEFAULT_LLM_PROVIDERS)
        self._structured_llm = structured_llm
        self.aborted_reason: Optional[str] = None

    @property
    def structured_llm(self):
        """Lazily build the provider fallback chain's structured-output runnable."""
        if self._structured_llm is None:
            self._structured_llm = self._build_fallback_chain()
        return self._structured_llm

    def _parse_providers(self) -> List[str]:
        """Parse and validate the ordered provider list from ``self.llm_providers``."""
        from src.services.claude_code_llm import PROVIDER as CLAUDE_CODE
        from src.services.llm_factory import SUPPORTED_PROVIDERS as API_PROVIDERS

        SUPPORTED_PROVIDERS = API_PROVIDERS + (CLAUDE_CODE,)
        providers = [p.strip().lower() for p in self.llm_providers.split(",") if p.strip()]
        if not providers:
            raise ValueError(
                f"LLM_PROVIDERS must contain at least one provider, got {self.llm_providers!r}"
            )
        for provider in providers:
            if provider not in SUPPORTED_PROVIDERS:
                raise ValueError(
                    f"Unsupported provider in LLM_PROVIDERS: {provider!r}. "
                    f"Expected one of: {SUPPORTED_PROVIDERS}"
                )
        return providers

    def _build_provider_runnable(self, provider: str):
        """Build one provider's structured-output runnable, tagged for attribution logging."""
        from langchain_core.runnables import RunnableLambda
        from src.services.llm_factory import build_chat_model

        # Only the anthropic leg respects an explicit `model=` override passed
        # to the service constructor; other providers use their own env-based
        # defaults (XAI_MODEL / DEEPSEEK_MODEL) via the factory.
        from src.services.claude_code_llm import PROVIDER as CLAUDE_CODE, ClaudeCodeStructured

        if provider == CLAUDE_CODE:
            # Local `claude -p` on the signed-in Claude Code account; no API key.
            structured = ClaudeCodeStructured(PickList)
        else:
            model_override = self.model if provider == "anthropic" else None
            llm = build_chat_model(provider, model=model_override, temperature=self.temperature)
            structured = llm.with_structured_output(PickList, method=STRUCTURED_OUTPUT_METHOD)

        async def _invoke(messages):
            result = await structured.ainvoke(messages)
            logger.info("Extraction served by provider=%s", provider)
            return result

        return RunnableLambda(_invoke, name=f"extraction_provider_{provider}")

    def _build_fallback_chain(self):
        """
        Compose the quality-first fallback chain: primary.with_fallbacks([...]).

        LangChain's ``with_fallbacks`` catches an error from the primary
        runnable (rate limit, API error, or a failed/malformed structured
        output) and retries the next runnable in order. Each per-provider
        runnable is wrapped so a successful call logs which provider served
        the result -- ``with_fallbacks`` itself doesn't expose that.
        """
        from src.services.claude_code_llm import PROVIDER as CLAUDE_CODE, claude_code_available
        from src.services.llm_factory import _API_KEY_ENV

        def usable(p: str) -> bool:
            return claude_code_available() if p == CLAUDE_CODE else bool(os.getenv(_API_KEY_ENV[p]))

        providers = self._parse_providers()
        # A provider without an API key (or, for claude_code, without the CLI)
        # is skipped, so a missing optional key never blocks the primary one.
        configured = [p for p in providers if usable(p)]
        skipped = [p for p in providers if p not in configured]
        if skipped:
            logger.warning("Extraction providers skipped (no API key / CLI): %s", skipped)
        if not configured:
            raise ValueError(
                "No extraction provider has an API key set; checked "
                + ", ".join(_API_KEY_ENV.get(p, "claude CLI") for p in providers)
            )
        providers = configured
        runnables = [self._build_provider_runnable(p) for p in providers]
        primary, *fallbacks = runnables
        logger.info("Extraction fallback chain built: %s", providers)
        return primary.with_fallbacks(fallbacks)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def extract(
        self,
        items: List[Dict[str, Any]],
        on_error: Optional[Callable[[Dict[str, Any], Exception], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract picks from a list of collected items.

        Each item is a dict with:
            - ``source``: "youtube" | "telegram" (defaults to "youtube")
            - ``transcript`` (or ``text``): the text to analyse
            - ``transcript_segments``: optional [{"start": float, "text": str}]
            - youtube metadata: ``video_id``, ``url``, ``channel``, ``title``
            - telegram metadata: ``message_link``, ``channel``, ``ts``

        ``on_error(item, exc)`` is called for each item whose LLM call fails.
        One failure does not stop the batch, but a fatal error (bad API key,
        no access, no credit) or ``MAX_CONSECUTIVE_FAILURES`` failures in a row
        stop it at once: the remaining items are reported through ``on_error``
        without being sent, and ``self.aborted_reason`` says why.

        Returns a list of mention dicts matching the frozen output contract.
        """
        mentions: List[Dict[str, Any]] = []
        dropped = 0
        consecutive_failures = 0
        self.aborted_reason = None

        for index, item in enumerate(items):
            transcript_text = self._build_transcript(item)
            if not transcript_text:
                logger.info(
                    "Skipping item with no transcript/text (source=%s, id=%s)",
                    item.get("source"),
                    item.get("video_id") or item.get("message_link"),
                )
                continue

            try:
                result = await self._invoke(item, transcript_text)
            except Exception as e:  # noqa: BLE001 - one item must not kill the batch
                logger.error("Extraction failed for item %s: %s", item.get("video_id"), e)
                if on_error is not None:
                    on_error(item, e)
                consecutive_failures += 1
                if is_fatal_llm_error(e) or consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    self.aborted_reason = (
                        f"stopped after {consecutive_failures} failed LLM call(s): {e}"
                    )
                    logger.error("Extraction aborted, %d item(s) not sent: %s",
                                 len(items) - index - 1, self.aborted_reason)
                    if on_error is not None:
                        for rest in items[index + 1:]:
                            on_error(rest, e)
                    break
                continue
            consecutive_failures = 0

            for raw in result.picks:
                mention = self._to_mention(raw, item)
                if mention is None:
                    dropped += 1
                    continue
                mentions.append(mention)

        if dropped:
            logger.info("Dropped %d malformed/low-confidence pick(s)", dropped)
        logger.info("Extraction produced %d mention(s) from %d item(s)", len(mentions), len(items))
        return mentions

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    async def _invoke(self, item: Dict[str, Any], transcript_text: str) -> PickList:
        """Call the structured LLM for a single item."""
        from langchain_core.messages import SystemMessage, HumanMessage

        source = item.get("source", "youtube")
        header = f"Source: {source}\nChannel: {item.get('channel', 'unknown')}"
        title = item.get("title")
        if title:
            header += f"\nTitle: {title}"

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"{header}\n\nTranscript:\n{transcript_text}"),
        ]
        result = await self.structured_llm.ainvoke(messages)
        # Be tolerant if a stub returns a plain dict.
        if isinstance(result, dict):
            result = PickList(**result)
        return result

    @staticmethod
    def _build_transcript(item: Dict[str, Any]) -> str:
        """Build transcript text with [Ns] time markers when segments are present."""
        segments = item.get("transcript_segments")
        if segments:
            lines = []
            for seg in segments:
                start = seg.get("start")
                text = (seg.get("text") or "").strip()
                if not text:
                    continue
                if start is None:
                    lines.append(text)
                else:
                    lines.append(f"[{int(float(start))}s] {text}")
            if lines:
                return "\n".join(lines)

        text = item.get("transcript") or item.get("text")
        return text.strip() if text else ""

    def _to_mention(self, raw: RawPick, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate a raw pick and shape it into the output-contract mention dict."""
        symbol = (raw.symbol or "").strip().upper().lstrip("$")
        if not symbol or not symbol.replace(".", "").replace("-", "").isalnum():
            logger.debug("Dropping pick with implausible symbol: %r", raw.symbol)
            return None

        stance = (raw.stance or "").strip().lower()
        if stance not in _STANCES:
            logger.debug("Dropping pick %s with bad stance: %r", symbol, raw.stance)
            return None

        confidence = self._clamp(raw.confidence, 0.0, 1.0)
        if confidence is None or confidence < self.min_confidence:
            logger.debug("Dropping low-confidence pick %s (conf=%s)", symbol, raw.confidence)
            return None

        sentiment = self._clamp(raw.sentiment_score, -1.0, 1.0)
        if sentiment is None:
            logger.debug("Dropping pick %s with bad sentiment: %r", symbol, raw.sentiment_score)
            return None

        source = item.get("source", "youtube")
        citation = self._build_citation(raw, item, source)
        if citation is None:
            logger.debug("Dropping pick %s: could not build citation", symbol)
            return None

        action = (raw.action or "").strip().lower() or None
        if action not in _ACTIONS:
            # Invalid/missing action: keep the pick for discovery, but never as a call.
            action = None
        horizon = (raw.horizon or "").strip().lower() or None
        if horizon not in _HORIZONS:
            # Invalid horizon drops only the horizon; trender defaults it from the
            # influencer's style.
            horizon = None
        is_recommendation = bool(raw.is_recommendation) and action is not None
        if not is_recommendation:
            action = None

        context = raw.thesis or ""
        if raw.quote:
            context = f"{context}\n\nQuote: {raw.quote}".strip()

        return {
            "symbol": symbol,
            "company_name": raw.company_name,
            "source": source,
            "stance": stance,
            "sentiment_score": sentiment,
            "confidence": confidence,
            "context": context,
            "citation": citation,
            "mentioned_at": self._mentioned_at(item),
            "is_recommendation": is_recommendation,
            "action": action,
            "horizon": horizon,
            "entry_price": raw.entry_price,
            "target_price": raw.target_price,
            "stop_price": raw.stop_price,
            "trigger_price": raw.trigger_price,
            "thesis": raw.thesis,
            "quote": raw.quote,
            # No now() fallback: a call without a real timestamp can't be scored.
            "said_at": item.get("ts") or None,
            "said_at_precision": item.get("ts_precision") or None,
        }

    @staticmethod
    def to_call_payload(
        mention: Dict[str, Any], origin: str, extractor_model: str
    ) -> Optional[Dict[str, Any]]:
        """
        Build trender's track-record ``CallIn`` dict from a mention, or None if the
        mention is not a scoreable call (not a recommendation, or missing
        action / said_at / source_item_id / citation.channel / valid source).
        """
        if not mention.get("is_recommendation") or not mention.get("action"):
            return None
        said_at = mention.get("said_at")
        if not said_at:
            return None
        source = mention.get("source")
        if source not in _CALL_SOURCES:
            return None
        citation = mention.get("citation") or {}
        source_item_id = (
            citation.get("video_id") if source == "youtube" else citation.get("message_link")
        )
        if not source_item_id or not citation.get("channel"):
            return None
        return {
            "symbol": mention["symbol"],
            "company_name": mention.get("company_name"),
            "source": source,
            "action": mention["action"],
            "stance": mention.get("stance"),
            "horizon": mention.get("horizon"),
            "conviction": mention.get("confidence"),  # pure rename
            "entry_price": mention.get("entry_price"),
            "target_price": mention.get("target_price"),
            "stop_price": mention.get("stop_price"),
            "trigger_price": mention.get("trigger_price"),
            "thesis": mention.get("thesis"),
            "quote": mention.get("quote"),
            "citation": citation,
            "source_item_id": source_item_id,
            "said_at": said_at,
            "said_at_precision": mention.get("said_at_precision") or "datetime",
            "origin": origin,
            "extractor_model": extractor_model,
            "extraction_version": EXTRACTION_VERSION,
        }

    @staticmethod
    def _build_citation(raw: RawPick, item: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """Assemble the citation payload from item metadata + the model's anchor."""
        if source == "youtube":
            video_id = item.get("video_id")
            if not video_id:
                return None
            start = raw.start_seconds if raw.start_seconds and raw.start_seconds > 0 else 0
            url = f"https://youtu.be/{video_id}"
            if start:
                url = f"{url}?t={start}"
            return {
                "kind": "youtube",
                "url": url,
                "video_id": video_id,
                "channel": item.get("channel"),
                "title": item.get("title"),
                "start_seconds": start,
                "quote": raw.quote,
            }

        if source == "telegram":
            message_link = item.get("message_link")
            if not message_link:
                return None
            return {
                "kind": "telegram",
                "message_link": message_link,
                "channel": item.get("channel"),
                "quote": raw.quote,
            }

        # Unknown source kind - keep a generic citation if a url is available.
        url = item.get("url") or item.get("message_link")
        if not url:
            return None
        return {"kind": source, "url": url, "channel": item.get("channel"), "quote": raw.quote}

    @staticmethod
    def _mentioned_at(item: Dict[str, Any]) -> str:
        """Best-effort mention timestamp as ISO-8601 UTC with trailing Z."""
        ts = item.get("ts") or item.get("published_at")
        if isinstance(ts, str) and ts:
            return ts
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _clamp(value: Any, low: float, high: float) -> Optional[float]:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        return max(low, min(high, v))

    @staticmethod
    def dedupe_key_for(mention: Dict[str, Any]) -> str:
        """dedupe_key = symbol | citation.url (or message_link) | start_seconds."""
        citation = mention.get("citation", {}) or {}
        url = citation.get("url") or citation.get("message_link") or ""
        start = citation.get("start_seconds")
        start_part = "" if start is None else str(start)
        return f"{mention.get('symbol', '')}|{url}|{start_part}"
