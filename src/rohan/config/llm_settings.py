"""LLM configuration settings.

Uses ``pydantic-settings`` for environment-based configuration with a
``.env`` file or environment variables.

Environment variables:
    LLM_PROVIDER        – "openrouter" | "openai" | "google"  (default: openrouter)
    LLM_ANALYSIS_MODEL  – model for analysis / explanation     (default: google/gemini-2.0-flash-001)
    LLM_CODEGEN_MODEL   – model for code generation            (default: anthropic/claude-sonnet-4)
    LLM_JUDGE_MODEL     – model for convergence judging        (default: google/gemini-2.0-flash-001)
    LLM_TEMPERATURE     – sampling temperature                 (default: 0.2)
    LLM_MAX_TOKENS      – max tokens for generation            (default: 4096)
    OPENROUTER_API_KEY   – API key when provider=openrouter
    OPENAI_API_KEY       – API key when provider=openai
    GOOGLE_API_KEY       – API key when provider=google
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(StrEnum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    GOOGLE = "google"


class LLMSettings(BaseSettings):
    """LLM connection and model configuration."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    provider: LLMProvider = Field(
        default=LLMProvider.OPENROUTER,
        description="LLM API provider",
    )

    # --- Model names (provider-specific) ---
    analysis_model: str = Field(
        default="google/gemini-2.0-flash-001",
        description="Model for result analysis and explanation",
    )
    codegen_model: str = Field(
        default="anthropic/claude-sonnet-4",
        description="Model for strategy code generation",
    )
    judge_model: str = Field(
        default="google/gemini-2.0-flash-001",
        description="Model for convergence judging",
    )

    # --- Sampling parameters ---
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=4096, ge=256, le=32768)

    # --- API keys (loaded from env / .env) ---
    openrouter_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="OPENROUTER_API_KEY",
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
    )
    google_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="GOOGLE_API_KEY",
    )
