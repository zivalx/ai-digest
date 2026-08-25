"""
Tests for the provider-agnostic LLM factory.

No real network calls or real API keys: provider classes are constructed
with fake keys (which they accept happily -- validation happens on first
real request, not at construction), and langchain_xai/langchain_deepseek are
monkeypatched out to exercise the langchain_openai base_url fallback path
without needing to uninstall anything.
"""
import pytest

from src.services import llm_factory
from src.services.llm_factory import build_chat_model, SUPPORTED_PROVIDERS


@pytest.fixture(autouse=True)
def _api_keys(monkeypatch):
    """Fake API keys for every provider so build_chat_model never fails on a missing key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")
    monkeypatch.setenv("XAI_API_KEY", "fake-xai-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")


def test_anthropic_returns_chat_anthropic():
    from langchain_anthropic import ChatAnthropic

    llm = build_chat_model("anthropic")
    assert isinstance(llm, ChatAnthropic)
    assert llm.model == "claude-sonnet-5"


def test_openai_returns_chat_openai():
    from langchain_openai import ChatOpenAI

    llm = build_chat_model("openai")
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "gpt-4o-mini"


def test_xai_returns_chat_xai_when_installed():
    from langchain_xai import ChatXAI

    llm = build_chat_model("xai")
    assert isinstance(llm, ChatXAI)
    assert llm.model_name == "grok-4"


def test_deepseek_returns_chat_deepseek_when_installed():
    from langchain_deepseek import ChatDeepSeek

    llm = build_chat_model("deepseek")
    assert isinstance(llm, ChatDeepSeek)
    assert llm.model_name == "deepseek-chat"


def test_model_env_override(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-4-8")
    llm = build_chat_model("anthropic")
    assert llm.model == "claude-opus-4-8"


def test_explicit_model_wins_over_env(monkeypatch):
    monkeypatch.setenv("XAI_MODEL", "grok-should-not-be-used")
    llm = build_chat_model("xai", model="grok-explicit")
    assert llm.model_name == "grok-explicit"


def test_xai_falls_back_to_chat_openai_when_package_missing(monkeypatch):
    """Simulate langchain_xai not being importable -> base_url fallback."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "langchain_xai":
            raise ImportError("simulated missing package")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from langchain_openai import ChatOpenAI

    llm = build_chat_model("xai")
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "grok-4"
    assert str(llm.openai_api_base) == "https://api.x.ai/v1"


def test_deepseek_falls_back_to_chat_openai_when_package_missing(monkeypatch):
    """Simulate langchain_deepseek not being importable -> base_url fallback."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "langchain_deepseek":
            raise ImportError("simulated missing package")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from langchain_openai import ChatOpenAI

    llm = build_chat_model("deepseek")
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "deepseek-chat"
    assert str(llm.openai_api_base) == "https://api.deepseek.com"


def test_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        build_chat_model("anthropic")


def test_unsupported_provider_raises():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        build_chat_model("not-a-real-provider")


def test_supported_providers_contains_all_four():
    assert set(SUPPORTED_PROVIDERS) == {"anthropic", "xai", "deepseek", "openai"}
