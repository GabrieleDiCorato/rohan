"""Feature flags for staged rollout of refinement-loop capabilities."""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field
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


class FeatureFlagSnapshot(BaseModel):
    """Serializable snapshot of feature flags carried through graph state."""

    llm_explainer_tiers_v1: bool = True
    explicit_terminal_reasons_v1: bool = True
    baseline_cache_v1: bool = True
    llm_telemetry_v1: bool = True

    @classmethod
    def from_settings(cls, flags: FeatureFlags) -> FeatureFlagSnapshot:
        """Create a snapshot from environment-backed settings."""
        return cls(
            llm_explainer_tiers_v1=flags.llm_explainer_tiers_v1,
            explicit_terminal_reasons_v1=flags.explicit_terminal_reasons_v1,
            baseline_cache_v1=flags.baseline_cache_v1,
            llm_telemetry_v1=flags.llm_telemetry_v1,
        )


@lru_cache(maxsize=1)
def get_feature_flags() -> FeatureFlags:
    """Get cached feature flags."""
    return FeatureFlags()


def feature_flags_snapshot() -> FeatureFlagSnapshot:
    """Return a typed snapshot of rollout flags for graph state."""
    return FeatureFlagSnapshot.from_settings(get_feature_flags())


def feature_flags_dict() -> dict[str, bool]:
    """Return feature flags as a plain dict for graph state propagation."""
    return feature_flags_snapshot().model_dump()
