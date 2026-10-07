"""Settings resolve the model mode without touching the network."""

import pytest

from tse_demo.config import load_settings


def test_auto_mode_uses_mock_when_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TSE_LLM_MODE", "auto")

    settings = load_settings()

    assert settings.resolved_llm_mode == "mock"


def test_auto_mode_uses_openai_when_api_key_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("TSE_LLM_MODE", "auto")

    settings = load_settings()

    assert settings.resolved_llm_mode == "openai"


def test_explicit_mock_mode_ignores_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("TSE_LLM_MODE", "mock")

    settings = load_settings()

    assert settings.resolved_llm_mode == "mock"


def test_tracing_is_off_unless_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

    assert load_settings().langsmith_tracing is False


def test_unknown_mode_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TSE_LLM_MODE", "somewhere")

    with pytest.raises(ValueError, match="TSE_LLM_MODE"):
        load_settings()
