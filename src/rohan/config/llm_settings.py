"""LLM configuration settings.

Uses ``pydantic-settings`` for environment-based configuration with a
``.env`` file or environment variables.

Environment variables:
    LLM_PROVIDER            – "openrouter" | "openai" | "google"  (default: openrouter)
    LLM_ANALYSIS_MODEL      – model for analysis / explanation     (default: google/gemini-2.0-flash-001)
    LLM_CODEGEN_MODEL       – model for code generation            (default: anthropic/claude-sonnet-4)
    LLM_JUDGE_MODEL         – model for convergence judging        (default: google/gemini-2.0-flash-001)
    LLM_TEMPERATURE         – sampling temperature                 (default: 0.2)
    LLM_JUDGE_TEMPERATURE   – judge sampling temperature           (default: 0.0, deterministic)
    LLM_MAX_TOKENS          – max tokens for generation            (default: 4096)
    LLM_WRITER_MAX_RETRIES  – writer retries on schema miss        (default: 3)
    LLM_WRITER_RETRY_PROMPT_TRIM – trim prompt on writer retries   (default: true)
    LLM_WRITER_FALLBACK_MODEL – optional model for last writer try (default: none)
    OPENROUTER_API_KEY       – API key when provider=openrouter
    OPENAI_API_KEY           – API key when provider=openai
    GOOGLE_API_KEY           – API key when provider=google
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

    # --- Refinement loop defaults ---
    max_validation_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum validation retries before giving up",
    )
    default_max_iterations: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Default maximum refinement iterations",
    )
    default_convergence_threshold: float = Field(
        default=7.0,
        ge=0,
        le=10,
        description="Judge score at/above which the strategy is considered converged",
    )
    default_recursion_limit: int = Field(
        default=80,
        ge=10,
        le=500,
        description="Maximum LangGraph steps before raising a recursion error",
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
    judge_temperature: float = Field(
        default=0.0,
        ge=0,
        le=2,
        description="Sampling temperature for the judge model (0.0 = deterministic)",
    )
    max_tokens: int = Field(default=4096, ge=256, le=32768)
    writer_max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Writer retries when structured output parsing returns no object",
    )
    writer_retry_prompt_trim: bool = Field(
        default=True,
        description="Trim writer retry prompts to reduce token pressure after schema misses",
    )
    writer_fallback_model: str | None = Field(
        default=None,
        description="Optional model name used only on the final writer retry attempt",
    )

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
