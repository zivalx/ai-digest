"""
LLM Factory - provider-agnostic chat model construction.

Centralises the "which LangChain chat model class for which provider, which
env var holds the API key, which env var overrides the model id" wiring that
``SummaryService._create_llm`` already implements for ``openai``/``anthropic``/
``xai``. ``ExtractionService`` reuses this factory to build a configurable
multi-provider fallback chain (see ``LLM_PROVIDERS`` in ``src.config``)
without duplicating provider construction logic.

Supported providers: ``anthropic``, ``xai``, ``deepseek``, ``openai``.

- ``anthropic`` -> ``langchain_anthropic.ChatAnthropic``
- ``xai`` (Grok) -> ``langchain_xai.ChatXAI`` if installed, else
  ``langchain_openai.ChatOpenAI`` pointed at xAI's OpenAI-compatible
  endpoint (``base_url="https://api.x.ai/v1"``)
- ``deepseek`` -> ``langchain_deepseek.ChatDeepSeek`` if installed, else
  ``langchain_openai.ChatOpenAI`` pointed at DeepSeek's OpenAI-compatible
  endpoint (``base_url="https://api.deepseek.com"``)
- ``openai`` -> ``langchain_openai.ChatOpenAI``

Model ids are always env-overridable, with these defaults:

- ``ANTHROPIC_MODEL`` (default ``claude-sonnet-5``)
- ``XAI_MODEL`` (default ``grok-4`` -- xAI's exact current model id is not
  guaranteed to stay accurate; override via env if it changes)
- ``DEEPSEEK_MODEL`` (default ``deepseek-chat``)
- ``OPENAI_MODEL`` (default ``gpt-4o-mini``)

API keys are read from env per provider (``ANTHROPIC_API_KEY``,
``XAI_API_KEY``, ``DEEPSEEK_API_KEY``, ``OPENAI_API_KEY``). Missing keys
raise a clear, actionable ``ValueError`` immediately -- this factory always
builds the provider it is asked for; a caller composing a fallback chain
(e.g. ``ExtractionService``) is responsible for deciding whether to skip a
provider *before* calling this function, not relying on it to skip silently.
"""
import os
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# provider -> (model env var, default model id)
_DEFAULT_MODELS = {
    "anthropic": ("ANTHROPIC_MODEL", "claude-sonnet-5"),
    "xai": ("XAI_MODEL", "grok-4"),
    "deepseek": ("DEEPSEEK_MODEL", "deepseek-chat"),
    "openai": ("OPENAI_MODEL", "gpt-4o-mini"),
}

# provider -> API key env var
_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "xai": "XAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
}

SUPPORTED_PROVIDERS = tuple(_DEFAULT_MODELS.keys())


def _require_api_key(provider: str) -> str:
    """Read the provider's API key from env, raising a clear error if unset."""
    env_var = _API_KEY_ENV[provider]
    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(
            f"{env_var} environment variable not set (required for provider={provider!r})"
        )
    return api_key


def build_chat_model(
    provider: str,
    model: Optional[str] = None,
    temperature: float = 0,
) -> Any:
    """
    Build a LangChain chat model for the given provider.

    Args:
        provider: One of "anthropic", "xai", "deepseek", "openai".
        model: Explicit model id. When omitted, falls back to the
            provider's env var (``ANTHROPIC_MODEL`` / ``XAI_MODEL`` /
            ``DEEPSEEK_MODEL`` / ``OPENAI_MODEL``) and then a hardcoded
            default.
        temperature: Sampling temperature.

    Returns:
        A LangChain chat model instance for ``provider``.

    Raises:
        ValueError: If ``provider`` is not one of ``SUPPORTED_PROVIDERS``,
            or the provider's API key environment variable is not set.
    """
    normalized = (provider or "").strip().lower()
    if normalized not in _DEFAULT_MODELS:
        raise ValueError(
            f"Unsupported LLM provider: {provider!r}. "
            f"Expected one of: {SUPPORTED_PROVIDERS}"
        )

    env_var, default_model = _DEFAULT_MODELS[normalized]
    resolved_model = model or os.getenv(env_var, default_model)
    api_key = _require_api_key(normalized)

    if normalized == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=resolved_model, temperature=temperature, api_key=api_key)

    if normalized == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=resolved_model, temperature=temperature, api_key=api_key)

    if normalized == "xai":
        try:
            from langchain_xai import ChatXAI
        except ImportError:
            logger.warning(
                "langchain_xai not installed; falling back to langchain_openai.ChatOpenAI "
                "against xAI's OpenAI-compatible endpoint"
            )
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=resolved_model,
                temperature=temperature,
                api_key=api_key,
                base_url="https://api.x.ai/v1",
            )
        return ChatXAI(model=resolved_model, temperature=temperature, api_key=api_key)

    # normalized == "deepseek"
    try:
        from langchain_deepseek import ChatDeepSeek
    except ImportError:
        logger.warning(
            "langchain_deepseek not installed; falling back to langchain_openai.ChatOpenAI "
            "against DeepSeek's OpenAI-compatible endpoint"
        )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=resolved_model,
            temperature=temperature,
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
    return ChatDeepSeek(model=resolved_model, temperature=temperature, api_key=api_key)
