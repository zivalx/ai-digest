"""
Collection Orchestrator Service

Orchestrates data collection from multiple sources using the connectors library.
This service DOES NOT collect data itself - it delegates to the connectors library.

Architecture:
    UI Config → CollectionOrchestrator → Connectors Library → Raw Data
"""
import os
import logging
from typing import List, Dict, Any
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


class UnsupportedSourceError(Exception):
    """Raised when attempting to collect from an unsupported source type."""
    pass


class CollectionError(Exception):
    """Raised when collection fails."""
    pass


class CollectionOrchestrator:
    """
    Orchestrates collection from multiple sources.

    This class maps UI configurations to connector library calls.
    It does NOT implement collection logic - that's in the connectors library.
    """

    SUPPORTED_SOURCES = ["reddit", "youtube", "telegram", "twitter", "gnews", "pytrends"]

    def __init__(self):
        """Initialize orchestrator."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _apply_timeframe(
        self,
        source_type: str,
        collect_spec: Dict[str, Any],
        timeframe_days: int
    ) -> Dict[str, Any]:
        """
        Apply timeframe to collect_spec based on source type.

        Args:
            source_type: Type of source
            collect_spec: Original collect spec
            timeframe_days: Number of days back to collect

        Returns:
            Updated collect_spec with timeframe applied
        """
        spec = collect_spec.copy()

        if source_type == "pytrends":
            # PyTrends uses format: "today 1-d", "today 3-d", "today 7-d"
            spec.setdefault("timeframe", f"today {timeframe_days}-d")
        elif source_type == "youtube":
            # YouTube uses days_back parameter
            spec.setdefault("days_back", timeframe_days)
        elif source_type == "gnews":
            # GNews might support from/to parameters (check connector docs)
            # For now, we'll rely on the connector's default behavior
            pass
        # Reddit, Twitter, Telegram don't have built-in time filtering in basic API
        # They will return recent posts by default

        return spec

    async def collect(
        self,
        source_type: str,
        credential_ref: str,
        collect_spec: Dict[str, Any],
        timeframe_days: int = 1
    ) -> Dict[str, Any]:
        """
        Collect data from a single source.

        Args:
            source_type: Type of source ("reddit", "youtube", etc.)
            credential_ref: Reference to environment variable containing credentials
            collect_spec: Collection specification (what to collect)
            timeframe_days: Number of days back to collect (1-7)

        Returns:
            Dict containing:
                - data: List of collected items (Pydantic models as dicts)
                - metadata: Collection metadata
                - source_info: Source identification

        Raises:
            UnsupportedSourceError: If source type is not supported
            CollectionError: If collection fails
        """
        # Apply timeframe to collect_spec based on source type
        collect_spec = self._apply_timeframe(source_type, collect_spec, timeframe_days)
        if source_type not in self.SUPPORTED_SOURCES:
            raise UnsupportedSourceError(f"Source type '{source_type}' not supported")

        self.logger.info(f"Starting collection from {source_type}")
        start_time = datetime.now()

        try:
            # Dispatch to appropriate collector
            if source_type == "reddit":
                result = await self._collect_reddit(credential_ref, collect_spec)
            elif source_type == "youtube":
                result = await self._collect_youtube(credential_ref, collect_spec)
            elif source_type == "telegram":
                result = await self._collect_telegram(credential_ref, collect_spec)
            elif source_type == "twitter":
                result = await self._collect_twitter(credential_ref, collect_spec)
            elif source_type == "gnews":
                result = await self._collect_gnews(credential_ref, collect_spec)
            elif source_type == "pytrends":
                result = await self._collect_pytrends(credential_ref, collect_spec)
            else:
                raise UnsupportedSourceError(f"Handler for {source_type} not implemented")

            # Add metadata
            end_time = datetime.now()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            return {
                **result,
                "metadata": {
                    "collection_time_ms": duration_ms,
                    "collected_at": end_time.isoformat(),
                    "source_type": source_type,
                }
            }

        except Exception as e:
            self.logger.error(f"Collection from {source_type} failed: {e}", exc_info=True)
            raise CollectionError(f"Failed to collect from {source_type}: {str(e)}") from e

    async def _collect_reddit(
        self,
        credential_ref: str,
        collect_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collect from Reddit using connectors library."""
        from connectors.reddit import RedditCollector, RedditClientConfig, RedditCollectSpec

        # Build client config from environment variables
        client_id = os.getenv(f"{credential_ref}_CLIENT_ID")
        client_secret = os.getenv(f"{credential_ref}_CLIENT_SECRET")
        user_agent = os.getenv(f"{credential_ref}_USER_AGENT", "Daigest/2.0")

        if not client_id or not client_secret:
            raise CollectionError(f"Missing Reddit credentials for {credential_ref}")

        config = RedditClientConfig(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )

        # Build collect spec from UI config
        spec = RedditCollectSpec(
            subreddits=collect_spec.get("subreddits", []),
            sort=collect_spec.get("sort", "hot"),
            max_posts_per_subreddit=collect_spec.get("max_posts", 50),
            include_comments=collect_spec.get("include_comments", False),
        )

        # Collect data
        collector = RedditCollector(config)
        posts = await collector.fetch(spec)

        # Convert Pydantic models to dicts
        return {
            "data": [post.model_dump() for post in posts],
            "item_count": len(posts),
            "source_info": {
                "subreddits": spec.subreddits,
                "sort": spec.sort,
            }
        }

    async def _collect_youtube(
        self,
        credential_ref: str,
        collect_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collect from YouTube using connectors library."""
        from connectors.youtube import YouTubeCollector, YouTubeClientConfig, YouTubeCollectSpec

        # Build client config
        config = YouTubeClientConfig(
            whisper_model=collect_spec.get("whisper_model", "base"),
            use_transcript_api=collect_spec.get("use_transcript_api", True),
            transcript_languages=collect_spec.get("transcript_languages", ["en"]),
            max_video_length=collect_spec.get("max_video_length", 3600),
        )

        # Build collect spec
        spec = YouTubeCollectSpec(
            channels=collect_spec.get("channels", []),
            max_videos_per_channel=(
                collect_spec.get("max_videos_per_channel")
                or collect_spec.get("max_videos", 5)
            ),
            days_back=collect_spec.get("days_back", 7),
        )

        # Collect data
        collector = YouTubeCollector(config)
        videos = await collector.fetch(spec)

        return {
            "data": [video.model_dump() for video in videos],
            "item_count": len(videos),
            "source_info": {
                "channels": spec.channels,
                "max_videos_per_channel": spec.max_videos_per_channel,
            }
        }

    async def _collect_telegram(
        self,
        credential_ref: str,
        collect_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collect from Telegram using connectors library."""
        from connectors.telegram import TelegramCollector, TelegramClientConfig, TelegramCollectSpec

        # Build client config
        api_id = os.getenv(f"{credential_ref}_API_ID")
        api_hash = os.getenv(f"{credential_ref}_API_HASH")
        phone = os.getenv(f"{credential_ref}_PHONE")

        if not all([api_id, api_hash, phone]):
            raise CollectionError(f"Missing Telegram credentials for {credential_ref}")

        # Auth callbacks. Telethon only invokes get_code when there's no valid
        # session (first login or an expired/revoked one). On a headless server
        # (no TTY) there's nothing to read a code from, so fail fast with a clear
        # message instead of blocking forever on input().
        import sys

        async def get_code():
            if not sys.stdin or not sys.stdin.isatty():
                raise CollectionError(
                    f"Telegram session for {credential_ref} is missing or expired and no "
                    f"interactive terminal is available to enter the login code. Generate the "
                    f".session file locally (one interactive login) and deploy it, then retry."
                )
            return input("Enter Telegram login code: ")

        async def get_password():
            return os.getenv(f"{credential_ref}_PASSWORD", "")

        config = TelegramClientConfig(
            api_id=int(api_id),
            api_hash=api_hash,
            phone=phone,
            auth_code_callback=get_code,
            auth_password_callback=get_password,
        )

        # Build collect spec
        spec = TelegramCollectSpec(
            channels=collect_spec.get("channels", []),
            max_messages_per_channel=collect_spec.get("max_messages", 200),
            include_replies=collect_spec.get("include_replies", True),
        )

        # Collect data
        collector = TelegramCollector(config)
        messages = await collector.fetch(spec)

        return {
            "data": [msg.model_dump() for msg in messages],
            "item_count": len(messages),
            "source_info": {
                "channels": spec.channels,
            }
        }

    async def _collect_twitter(
        self,
        credential_ref: str,
        collect_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collect from Twitter using connectors library."""
        from connectors.twitter import TwitterCollector, TwitterClientConfig, TwitterCollectSpec

        # Build client config
        bearer_token = os.getenv(f"{credential_ref}_BEARER_TOKEN")

        if not bearer_token:
            raise CollectionError(f"Missing Twitter credentials for {credential_ref}")

        config = TwitterClientConfig(bearer_token=bearer_token)

        # Build collect spec
        spec = TwitterCollectSpec(
            query=collect_spec.get("query", ""),
            max_results=collect_spec.get("max_results", 100),
        )

        # Collect data
        collector = TwitterCollector(config)
        tweets = await collector.fetch(spec)

        return {
            "data": [tweet.model_dump() for tweet in tweets],
            "item_count": len(tweets),
            "source_info": {
                "query": spec.query,
            }
        }

    async def _collect_gnews(
        self,
        credential_ref: str,
        collect_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collect from GNews using connectors library."""
        from connectors.gnews import GNewsCollector, GNewsClientConfig, GNewsCollectSpec

        # Build client config
        api_key = os.getenv(f"{credential_ref}_API_KEY")

        if not api_key:
            raise CollectionError(f"Missing GNews credentials for {credential_ref}")

        config = GNewsClientConfig(api_key=api_key)

        # Build collect spec
        spec = GNewsCollectSpec(
            query=collect_spec.get("query", ""),
            language=collect_spec.get("language", "en"),
            max_results=collect_spec.get("max_results", 10),
            sort_by=collect_spec.get("sort_by", "publishedAt"),
        )

        # Collect data
        collector = GNewsCollector(config)
        result = await collector.fetch(spec)

        if result.status != "success":
            raise CollectionError(f"GNews API error: {result.error}")

        return {
            "data": [article.model_dump() for article in result.articles],
            "item_count": len(result.articles),
            "source_info": {
                "query": spec.query,
                "language": spec.language,
            }
        }

    async def _collect_pytrends(
        self,
        credential_ref: str,
        collect_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collect from Google Trends using connectors library."""
        from connectors.pytrends import PyTrendsCollector, PyTrendsClientConfig, PyTrendsCollectSpec

        # Build client config (no credentials needed for PyTrends)
        config = PyTrendsClientConfig(
            timeout=30,
            retries=3,
        )

        # Build collect spec
        spec = PyTrendsCollectSpec(
            keywords=collect_spec.get("keywords", []),
            timeframe=collect_spec.get("timeframe", "today 3-m"),
            geo=collect_spec.get("geo", ""),
            include_related_queries=collect_spec.get("include_related_queries", True),
            include_interest_by_region=collect_spec.get("include_interest_by_region", False),
        )

        # Collect data
        collector = PyTrendsCollector(config)
        result = await collector.fetch(spec)

        if result.status != "success":
            raise CollectionError(f"PyTrends error: {result.error}")

        # Serialize trend data
        trends_data = {
            "interest_over_time": [trend.model_dump() for trend in result.interest_over_time],
            "related_queries_top": {k: [q.model_dump() for q in v] for k, v in result.related_queries_top.items()},
        }

        return {
            "data": trends_data,
            "item_count": len(result.interest_over_time),
            "source_info": {
                "keywords": spec.keywords,
                "timeframe": spec.timeframe,
                "geo": spec.geo,
            }
        }

    async def collect_multiple(
        self,
        sources: List[Dict[str, Any]],
        timeframe_days: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Collect from multiple sources concurrently.

        Args:
            sources: List of source configurations, each containing:
                - source_type: str
                - credential_ref: str
                - collect_spec: dict
            timeframe_days: Number of days back to collect (1-7)

        Returns:
            List of collection results (same order as input)
        """
        tasks = [
            self.collect(
                source["source_type"],
                source["credential_ref"],
                source["collect_spec"],
                timeframe_days
            )
            for source in sources
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error dicts
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Collection from {sources[i]['source_type']} failed: {result}")
                processed_results.append({
                    "error": str(result),
                    "source_type": sources[i]["source_type"],
                    "status": "failed",
                })
            else:
                processed_results.append(result)

        return processed_results
