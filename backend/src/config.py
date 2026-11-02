"""
Application settings for the channel-intelligence feeder.

Kept intentionally light: values are read from the environment on demand so
tests can set/override env vars and call `get_settings()` to pick them up.
The rest of ai-digest reads env via ``os.getenv`` directly; this module just
centralises the feeder-specific knobs.
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FeederSettings:
    """Channel-intel feeder configuration, sourced from environment variables."""

    trender_base_url: str
    trender_ingest_token: Optional[str]
    anthropic_model: str
    xai_model: str
    deepseek_model: str
    llm_providers: str
    feeder_interval_seconds: int
    feeder_enabled: bool


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def get_settings() -> FeederSettings:
    """Read feeder settings from the current environment."""
    return FeederSettings(
        trender_base_url=os.getenv("TRENDER_BASE_URL", "http://localhost:8000"),
        trender_ingest_token=os.getenv("TRENDER_INGEST_TOKEN") or None,
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"),
        xai_model=os.getenv("XAI_MODEL", "grok-4"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        llm_providers=os.getenv("LLM_PROVIDERS", "anthropic,xai,deepseek"),
        feeder_interval_seconds=int(os.getenv("FEEDER_INTERVAL_SECONDS", "21600")),
        feeder_enabled=_as_bool(os.getenv("FEEDER_ENABLED"), default=False),
    )
