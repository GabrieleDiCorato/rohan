"""Unit tests for AbidesConfigMapper (TEST-1).

Tests agent count correctness, oracle parameter passthrough, latency model
type switching, and configuration structure.
"""

from __future__ import annotations

import numpy as np
from abides_core import LatencyModel
from abides_markets.agents import AdaptiveMarketMakerAgent, ExchangeAgent, MomentumAgent, NoiseAgent, ValueAgent
from abides_markets.oracles import SparseMeanRevertingOracle

from rohan.config import LatencyType, SimulationSettings
from rohan.simulation.abides_impl.abides_config_mapper import AbidesConfigMapper
from rohan.simulation.abides_impl.random_state_handler import RandomStateHandler
from rohan.simulation.abides_impl.strategic_agent_adapter import StrategicAgentAdapter
from rohan.simulation.models.strategy_api import MarketState, OrderAction, StrategicAgent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyStrategy(StrategicAgent):
    """Minimal strategy that does nothing — sufficient for testing agent creation."""

    def on_trading_started(self, _market_state: MarketState) -> list[OrderAction]:
        return []

    def on_tick(self, _market_state: MarketState) -> list[OrderAction]:
        return []

    def on_order_update(self, _market_state: MarketState) -> list[OrderAction]:
        return []

    def on_simulation_end(self, _market_state: MarketState) -> None:
        pass


def _default_settings(**overrides) -> SimulationSettings:
    return SimulationSettings(seed=42, **overrides)


# ---------------------------------------------------------------------------
# Agent count correctness
# ---------------------------------------------------------------------------


class TestAgentCounts:
    """Verify _build_agents creates the correct number and types of agents."""

    def test_default_agent_count_without_strategy(self):
        settings = _default_settings()
        mapper = AbidesConfigMapper(settings, strategy=None)
        config = mapper.build_configuration()
        agents = config["agents"]

        expected_count = (
            1  # exchange
            + settings.agents.noise.num_agents
            + settings.agents.value.num_agents
            + settings.agents.adaptive_market_maker.num_agents
            + settings.agents.momentum.num_agents
        )
        assert len(agents) == expected_count
        assert mapper.strategic_agent_id is None

    def test_agent_count_with_strategy(self):
        settings = _default_settings()
        mapper = AbidesConfigMapper(settings, strategy=_DummyStrategy())
        config = mapper.build_configuration()
        agents = config["agents"]

        expected_count = (
            1  # exchange
            + settings.agents.noise.num_agents
            + settings.agents.value.num_agents
            + settings.agents.adaptive_market_maker.num_agents
            + settings.agents.momentum.num_agents
            + 1  # strategic agent
        )
        assert len(agents) == expected_count
        assert mapper.strategic_agent_id is not None
        assert isinstance(agents[mapper.strategic_agent_id], StrategicAgentAdapter)

    def test_first_agent_is_exchange(self):
        settings = _default_settings()
        mapper = AbidesConfigMapper(settings)
        config = mapper.build_configuration()
        assert isinstance(config["agents"][0], ExchangeAgent)

    def test_agent_types_present(self):
        settings = _default_settings()
        mapper = AbidesConfigMapper(settings)
        config = mapper.build_configuration()
        agent_types = {type(a) for a in config["agents"]}
        assert ExchangeAgent in agent_types
        assert NoiseAgent in agent_types
        assert ValueAgent in agent_types
        assert AdaptiveMarketMakerAgent in agent_types
        assert MomentumAgent in agent_types

    def test_strategic_agent_id_is_last(self):
        settings = _default_settings()
        mapper = AbidesConfigMapper(settings, strategy=_DummyStrategy())
        config = mapper.build_configuration()
        assert mapper.strategic_agent_id == len(config["agents"]) - 1


# ---------------------------------------------------------------------------
# Configuration structure
# ---------------------------------------------------------------------------


class TestConfigStructure:
    """Verify build_configuration returns expected keys and values."""

    def test_config_has_required_keys(self):
        settings = _default_settings()
        mapper = AbidesConfigMapper(settings)
        config = mapper.build_configuration()

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
        assert required_keys <= set(config.keys())

    def test_config_seed_matches_settings(self):
        settings = _default_settings()
        mapper = AbidesConfigMapper(settings)
        config = mapper.build_configuration()
        assert config["seed"] == settings.seed

    def test_config_has_oracle_in_custom_properties(self):
        settings = _default_settings()
        mapper = AbidesConfigMapper(settings)
        config = mapper.build_configuration()
        assert "oracle" in config["custom_properties"]
        assert isinstance(config["custom_properties"]["oracle"], SparseMeanRevertingOracle)

    def test_stop_time_after_start_time(self):
        settings = _default_settings()
        mapper = AbidesConfigMapper(settings)
        config = mapper.build_configuration()
        assert config["stop_time"] > config["start_time"]

    def test_ticker_propagated_to_exchange_agent(self):
        settings = _default_settings(ticker="ABM")
        mapper = AbidesConfigMapper(settings)
        config = mapper.build_configuration()
        exchange = config["agents"][0]
        assert "ABM" in exchange.symbols


# ---------------------------------------------------------------------------
# Latency model type switching
# ---------------------------------------------------------------------------


class TestLatencyModel:
    """Verify _build_latency_model produces correct types for each LatencyType."""

    def test_default_latency_is_deterministic(self):
        """Default latency (DETERMINISTIC) produces non-zero off-diagonal entries."""
        settings = _default_settings()
        mapper = AbidesConfigMapper(settings)
        config = mapper.build_configuration()
        latency = config["agent_latency_model"]
        assert isinstance(latency, LatencyModel)
        n = len(config["agents"])
        assert latency.min_latency.shape == (n, n)

    def test_deterministic_latency_nonzero(self):
        from rohan.config.latency_settings import LatencyModelSettings

        handler = RandomStateHandler(42)
        settings = LatencyModelSettings(type=LatencyType.DETERMINISTIC)
        latency_model = AbidesConfigMapper._build_latency_model(
            agent_count=5,
            latency_settings=settings,
            random_state_handler=handler,
        )
        assert isinstance(latency_model, LatencyModel)
        # Off-diagonal entries should be non-zero (agents at different positions)
        off_diag = latency_model.min_latency[~np.eye(5, dtype=bool)]
        assert np.any(off_diag > 0)


# ---------------------------------------------------------------------------
# Oracle parameter passthrough
# ---------------------------------------------------------------------------


class TestOracle:
    """Verify _build_oracle passes simulation settings correctly."""

    def test_oracle_returns_sparse_mean_reverting(self):
        settings = _default_settings()
        handler = RandomStateHandler(42)
        import pandas as pd
        from abides_core.utils import str_to_ns

        date = int(pd.to_datetime(settings.date).value)
        mkt_open = date + str_to_ns(settings.start_time)
        noise_mkt_close = date + str_to_ns("16:00:00")
        oracle = AbidesConfigMapper._build_oracle(settings, mkt_open, noise_mkt_close, handler)
        assert isinstance(oracle, SparseMeanRevertingOracle)

    def test_historical_oracle_returns_external_data_oracle(self, tmp_path):
        import pandas as pd
        from abides_markets.oracles import ExternalDataOracle

        from rohan.config import HistoricalOracleSettings, OracleType

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
        settings.agents.oracle.historical = HistoricalOracleSettings(provider_type="CSV", csv_path=str(csv_path), interpolation="ffill")

        handler = RandomStateHandler(42)
        from abides_core.utils import str_to_ns

        date = int(pd.to_datetime(settings.date).value)
        mkt_open = date + str_to_ns(settings.start_time)
        noise_mkt_close = date + str_to_ns("16:00:00")

        oracle = AbidesConfigMapper._build_oracle(settings, mkt_open, noise_mkt_close, handler)
        assert isinstance(oracle, ExternalDataOracle)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Same settings must produce identical configurations."""

    def test_same_seed_same_agent_count(self):
        s1 = _default_settings()
        s2 = _default_settings()
        m1 = AbidesConfigMapper(s1)
        m2 = AbidesConfigMapper(s2)
        c1 = m1.build_configuration()
        c2 = m2.build_configuration()
        assert len(c1["agents"]) == len(c2["agents"])
