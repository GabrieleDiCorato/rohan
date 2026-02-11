"""Tests for LLM configuration settings (Phase 2.1)."""

import pytest
from pydantic import ValidationError

from rohan.config.llm_settings import LLMProvider, LLMSettings


class TestLLMProvider:
    def test_enum_values(self):
        assert LLMProvider.OPENROUTER == "openrouter"
        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.GOOGLE == "google"

    def test_enum_membership(self):
        assert "openrouter" in LLMProvider.__members__.values()


class TestLLMSettings:
    def test_defaults(self):
        """Settings should load with sensible defaults even without env vars."""
        settings = LLMSettings(
            _env_file=None,  # type: ignore[call-arg]  # Pydantic-settings parameter
        )
        assert settings.provider == LLMProvider.OPENROUTER
        assert "gemini" in settings.analysis_model
        assert "claude" in settings.codegen_model
        assert settings.temperature == 0.2
        assert settings.max_tokens == 4096

    def test_env_override(self, monkeypatch):
        """Environment variables should override defaults."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_TEMPERATURE", "0.8")
        monkeypatch.setenv("LLM_MAX_TOKENS", "8192")
        monkeypatch.setenv("LLM_ANALYSIS_MODEL", "gpt-4o")

        settings = LLMSettings(_env_file=None)  # type: ignore[call-arg]
        assert settings.provider == LLMProvider.OPENAI
        assert settings.temperature == 0.8
        assert settings.max_tokens == 8192
        assert settings.analysis_model == "gpt-4o"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
        settings = LLMSettings(_env_file=None)  # type: ignore[call-arg]
        assert settings.openrouter_api_key is not None
        assert settings.openrouter_api_key.get_secret_value() == "sk-test-key"

    def test_api_key_is_secret(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret-123")
        settings = LLMSettings(_env_file=None)  # type: ignore[call-arg]
        # SecretStr should not expose the key in repr/str
        assert "sk-secret-123" not in repr(settings)

    def test_temperature_bounds(self):
        with pytest.raises(ValidationError):
            LLMSettings(temperature=-0.1, _env_file=None)  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            LLMSettings(temperature=2.5, _env_file=None)  # type: ignore[call-arg]

    def test_max_tokens_bounds(self):
        with pytest.raises(ValidationError):
            LLMSettings(max_tokens=100, _env_file=None)  # type: ignore[call-arg]  # below 256

    def test_no_api_keys_by_default(self):
        settings = LLMSettings(_env_file=None)  # type: ignore[call-arg]
        assert settings.openrouter_api_key is None
        assert settings.openai_api_key is None
        assert settings.google_api_key is None
