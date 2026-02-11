"""Tests for LLM model factory (Phase 2.1)."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from rohan.config.llm_settings import LLMSettings
from rohan.llm.factory import (
    _OPENROUTER_BASE_URL,
    create_chat_model,
    get_analysis_model,
    get_codegen_model,
    get_judge_model,
)


def _settings(provider: str = "openrouter", **overrides) -> LLMSettings:
    """Create test settings without reading .env."""
    return LLMSettings(
        provider=provider,  # type: ignore[arg-type]  # String will be coerced to enum
        openrouter_api_key=SecretStr("sk-test"),
        openai_api_key=SecretStr("sk-test-openai"),
        _env_file=None,  # type: ignore[call-arg]
        **overrides,
    )


class TestCreateChatModel:
    @patch("rohan.llm.factory.ChatOpenAI")
    def test_openrouter_creates_chat_openai(self, mock_cls):
        mock_cls.return_value = MagicMock()
        settings = _settings("openrouter")
        create_chat_model("anthropic/claude-sonnet-4", settings)

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["model"] == "anthropic/claude-sonnet-4"
        assert call_kwargs["base_url"] == _OPENROUTER_BASE_URL
        assert call_kwargs["temperature"] == 0.2

    @patch("rohan.llm.factory.ChatOpenAI")
    def test_openai_creates_chat_openai(self, mock_cls):
        mock_cls.return_value = MagicMock()
        settings = _settings("openai")
        create_chat_model("gpt-4o", settings)

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
        # OpenAI provider should NOT set base_url
        assert "base_url" not in call_kwargs

    def test_missing_openrouter_key_raises(self):
        settings = LLMSettings(
            provider="openrouter",  # type: ignore[arg-type]
            openrouter_api_key=None,
            _env_file=None,  # type: ignore[call-arg]
        )
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            create_chat_model("model", settings)

    def test_missing_openai_key_raises(self):
        settings = LLMSettings(
            provider="openai",  # type: ignore[arg-type]
            openai_api_key=None,
            _env_file=None,  # type: ignore[call-arg]
        )
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            create_chat_model("model", settings)

    def test_missing_google_key_raises(self):
        settings = LLMSettings(
            provider="google",  # type: ignore[arg-type]
            google_api_key=None,
            _env_file=None,  # type: ignore[call-arg]
        )
        # May raise ImportError (no langchain-google-genai) or ValueError
        with pytest.raises((ValueError, ImportError)):
            create_chat_model("model", settings)

    def test_unsupported_provider_raises(self):
        settings = _settings("openrouter")
        # Force an invalid provider
        object.__setattr__(settings, "provider", "unsupported")
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            create_chat_model("model", settings)


class TestConvenienceGetters:
    @patch("rohan.llm.factory.create_chat_model")
    @patch("rohan.llm.factory._cached_settings")
    def test_get_codegen_model(self, mock_settings, mock_create):
        mock_settings.return_value = _settings()
        mock_create.return_value = MagicMock()

        get_codegen_model()
        mock_create.assert_called_once()
        _, kwargs = mock_create.call_args
        # Should use codegen model name from settings
        args = mock_create.call_args.args
        assert "claude" in args[0]

    @patch("rohan.llm.factory.create_chat_model")
    @patch("rohan.llm.factory._cached_settings")
    def test_get_analysis_model(self, mock_settings, mock_create):
        mock_settings.return_value = _settings()
        mock_create.return_value = MagicMock()

        get_analysis_model()
        mock_create.assert_called_once()
        args = mock_create.call_args.args
        assert "gemini" in args[0]

    @patch("rohan.llm.factory.create_chat_model")
    @patch("rohan.llm.factory._cached_settings")
    def test_get_judge_model(self, mock_settings, mock_create):
        mock_settings.return_value = _settings()
        mock_create.return_value = MagicMock()

        get_judge_model()
        mock_create.assert_called_once()
        args = mock_create.call_args.args
        assert "gemini" in args[0]

    @patch("rohan.llm.factory.create_chat_model")
    def test_explicit_settings_bypass_cache(self, mock_create):
        mock_create.return_value = MagicMock()
        s = _settings()
        get_codegen_model(settings=s)
        mock_create.assert_called_once_with(s.codegen_model, s)
