"""Feature flags for staged rollout of refinement-loop capabilities."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FeatureFlags(BaseSettings):
    """Environment-driven feature toggles for refinement rollout."""

    model_config = SettingsConfigDict(
        env_prefix="FEATURE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_explainer_tiers_v1: bool = Field(default=True)
    explicit_terminal_reasons_v1: bool = Field(default=True)
    baseline_cache_v1: bool = Field(default=True)
    llm_telemetry_v1: bool = Field(default=True)


@lru_cache(maxsize=1)
def get_feature_flags() -> FeatureFlags:
    """Get cached feature flags."""
    return FeatureFlags()


def feature_flags_dict() -> dict[str, bool]:
    """Return feature flags as a plain dict for graph state propagation."""
    flags = get_feature_flags()
    return {
        "llm_explainer_tiers_v1": flags.llm_explainer_tiers_v1,
        "explicit_terminal_reasons_v1": flags.explicit_terminal_reasons_v1,
        "baseline_cache_v1": flags.baseline_cache_v1,
        "llm_telemetry_v1": flags.llm_telemetry_v1,
    }
