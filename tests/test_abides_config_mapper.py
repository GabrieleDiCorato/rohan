"""Unit tests for config_builder (replaces old AbidesConfigMapper tests).

Tests agent count correctness, oracle parameter passthrough, latency model
type switching, and configuration structure via the hasufel pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest
from abides_core import LatencyModel
from abides_markets.agents import (
    AdaptiveMarketMakerAgent,
    ExchangeAgent,
    MomentumAgent,
    NoiseAgent,
    ValueAgent,
)
from abides_markets.config_system import compile as compile_config
from abides_markets.oracles import SparseMeanRevertingOracle

from rohan.config import SimulationSettings
from rohan.simulation.abides_impl.config_builder import build_simulation_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_settings(**overrides) -> SimulationSettings:
    return SimulationSettings(seed=42, **overrides)


def _build_and_compile(settings: SimulationSettings) -> dict:
    """Build + compile in one step (convenience for tests)."""
    config, oracle = build_simulation_config(settings)
    return compile_config(config, oracle_instance=oracle)


# ---------------------------------------------------------------------------
# Agent count correctness
# ---------------------------------------------------------------------------


class TestAgentCounts:
    """Verify compile produces the correct number and types of agents."""

    def test_default_agent_count(self):
        settings = _default_settings()
        runtime = _build_and_compile(settings)
        agents = runtime["agents"]

        expected_count = (
            1  # exchange
            + settings.agents.noise.num_agents
            + settings.agents.value.num_agents
            + settings.agents.adaptive_market_maker.num_agents
            + settings.agents.momentum.num_agents
        )
        assert len(agents) == expected_count

    def test_first_agent_is_exchange(self):
        settings = _default_settings()
        runtime = _build_and_compile(settings)
        assert isinstance(runtime["agents"][0], ExchangeAgent)

    def test_agent_types_present(self):
        settings = _default_settings()
        runtime = _build_and_compile(settings)
        agent_types = {type(a) for a in runtime["agents"]}
        assert ExchangeAgent in agent_types
        assert NoiseAgent in agent_types
        assert ValueAgent in agent_types
        assert AdaptiveMarketMakerAgent in agent_types
        assert MomentumAgent in agent_types


# ---------------------------------------------------------------------------
# Configuration structure
# ---------------------------------------------------------------------------


class TestConfigStructure:
    """Verify compiled runtime dict has expected keys and values."""

    def test_runtime_has_required_keys(self):
        settings = _default_settings()
        runtime = _build_and_compile(settings)

        required_keys = {
            "seed",
            "start_time",
            "stop_time",
            "agents",
            "agent_latency_model",
            "default_computation_delay",
            "custom_properties",
            "random_state_kernel",
            "stdout_log_level",
        }
        assert required_keys <= set(runtime.keys())

    def test_config_seed_matches_settings(self):
        settings = _default_settings()
        runtime = _build_and_compile(settings)
        assert runtime["seed"] == settings.seed

    def test_runtime_has_oracle_in_custom_properties(self):
        settings = _default_settings()
        runtime = _build_and_compile(settings)
        assert "oracle" in runtime["custom_properties"]
        assert isinstance(runtime["custom_properties"]["oracle"], SparseMeanRevertingOracle)

    def test_stop_time_after_start_time(self):
        settings = _default_settings()
        runtime = _build_and_compile(settings)
        assert runtime["stop_time"] > runtime["start_time"]

    def test_ticker_propagated_to_exchange_agent(self):
        settings = _default_settings(ticker="ABM")
        runtime = _build_and_compile(settings)
        exchange = runtime["agents"][0]
        assert "ABM" in exchange.symbols


# ---------------------------------------------------------------------------
# Latency model
# ---------------------------------------------------------------------------


class TestLatencyModel:
    """Verify latency model is produced correctly."""

    def test_default_latency_model(self):
        settings = _default_settings()
        runtime = _build_and_compile(settings)
        latency = runtime["agent_latency_model"]
        assert isinstance(latency, LatencyModel)
        n = len(runtime["agents"])
        assert latency.min_latency.shape == (n, n)

    def test_latency_nonzero_off_diagonal(self):
        settings = _default_settings()
        runtime = _build_and_compile(settings)
        latency = runtime["agent_latency_model"]
        n = len(runtime["agents"])
        off_diag = latency.min_latency[~np.eye(n, dtype=bool)]
        assert np.any(off_diag > 0)


# ---------------------------------------------------------------------------
# Oracle parameter passthrough
# ---------------------------------------------------------------------------


class TestOracle:
    """Verify oracle creation."""

    def test_default_oracle_is_sparse_mean_reverting(self):
        settings = _default_settings()
        runtime = _build_and_compile(settings)
        oracle = runtime["custom_properties"]["oracle"]
        assert isinstance(oracle, SparseMeanRevertingOracle)

    def test_historical_oracle_returns_external_data_oracle(self, tmp_path):
        import pandas as pd
        from abides_markets.oracles import ExternalDataOracle

        from rohan.config import (
            CsvHistoricalProviderSettings,
            HistoricalOracleSettings,
            InterpolationMode,
            OracleType,
            ProviderType,
        )

        # Setup mock CSV
        csv_path = tmp_path / "mock.csv"
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-30T09:00:00", "2026-01-30T16:00:00", freq="1min"),
            }
        )
        df["price_cents"] = 100000
        df.to_csv(csv_path, index=False)

        settings = _default_settings()
        settings.date = "20260130"
        settings.ticker = "ABM"
        settings.agents.oracle.oracle_type = OracleType.HISTORICAL
        settings.agents.oracle.historical = HistoricalOracleSettings(
            provider_type=ProviderType.CSV,
            interpolation=InterpolationMode.FORWARD_FILL,
            csv=CsvHistoricalProviderSettings(csv_path=str(csv_path)),
        )

        config, oracle_instance = build_simulation_config(settings)
        assert isinstance(oracle_instance, ExternalDataOracle)

    def test_historical_oracle_requires_csv_path(self):
        from rohan.config import HistoricalOracleSettings, OracleType, ProviderType

        settings = _default_settings()
        settings.date = "20260130"
        settings.ticker = "ABM"
        settings.agents.oracle.oracle_type = OracleType.HISTORICAL
        settings.agents.oracle.historical = HistoricalOracleSettings(provider_type=ProviderType.CSV)

        with pytest.raises(ValueError, match="historical.csv.csv_path"):
            build_simulation_config(settings)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Same settings must produce identical configurations."""

    def test_same_seed_same_agent_count(self):
        s1 = _default_settings()
        s2 = _default_settings()
        r1 = _build_and_compile(s1)
        r2 = _build_and_compile(s2)
        assert len(r1["agents"]) == len(r2["agents"])
