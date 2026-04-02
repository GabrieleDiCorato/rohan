"""Tests for StrategicAgentConfig — the registered hasufel agent config.

Covers:
- Agent registration in the hasufel registry
- Agent creation via create_agents()
- Integration with hasufel's compile pipeline
- Config serialisation round-trip through AgentGroupConfig.params
"""

from __future__ import annotations

import numpy as np
import pytest
from abides_markets.config_system.agent_configs import AgentCreationContext
from abides_markets.config_system.registry import AgentRegistry

from rohan.simulation.abides_impl.strategic_agent_adapter import StrategicAgentAdapter
from rohan.simulation.abides_impl.strategic_agent_config import StrategicAgentConfig
from rohan.simulation.models.strategy_spec import StrategySpec

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SIMPLE_STRATEGY_CODE = """
from rohan.simulation.models.strategy_api import (
    AgentConfig, MarketState, Order, OrderAction, StrategicAgent,
)

class SimpleTestStrategy(StrategicAgent):
    def initialize(self, config: AgentConfig) -> None:
        self.config = config
    def on_tick(self, state: MarketState) -> list[OrderAction]:
        return []
    def on_market_data(self, state: MarketState) -> list[OrderAction]:
        return []
    def on_order_update(self, update: Order) -> list[OrderAction]:
        return []
    def on_simulation_end(self, final_state: MarketState) -> None:
        pass
"""


def _make_spec() -> StrategySpec:
    return StrategySpec(source_code=SIMPLE_STRATEGY_CODE, class_name="SimpleTestStrategy")


def _make_context(**overrides) -> AgentCreationContext:
    defaults = {
        "ticker": "ABM",
        "mkt_open": 34_200_000_000_000,
        "mkt_close": 57_600_000_000_000,
        "log_orders": True,
        "oracle_r_bar": 100_000,
        "date_ns": 0,
        "oracle_kappa": 1e-11,
        "oracle_sigma_s": 0.0,
    }
    defaults.update(overrides)
    return AgentCreationContext(**defaults)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_rohan_strategy_is_registered(self):
        registry = AgentRegistry()
        entry = registry.get("rohan_strategy")
        assert entry is not None
        assert entry.name == "rohan_strategy"
        assert entry.category == "strategy"
        assert entry.config_model is StrategicAgentConfig

    def test_registration_metadata(self):
        registry = AgentRegistry()
        entry = registry.get("rohan_strategy")
        assert entry.typical_count_range == (1, 1)
        assert "LLM-generated" in entry.description


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------


class TestCreateAgents:
    def test_creates_single_adapter(self):
        config = StrategicAgentConfig(strategy_spec=_make_spec())
        rng = np.random.RandomState(42)
        ctx = _make_context()

        agents = config.create_agents(count=1, id_start=5, master_rng=rng, context=ctx)

        assert len(agents) == 1
        agent = agents[0]
        assert isinstance(agent, StrategicAgentAdapter)
        assert agent.id == 5
        assert agent.symbol == "ABM"
        assert agent.category == "strategy"

    def test_agent_has_strategy_from_spec(self):
        config = StrategicAgentConfig(strategy_spec=_make_spec())
        rng = np.random.RandomState(42)
        ctx = _make_context()

        agents = config.create_agents(count=1, id_start=0, master_rng=rng, context=ctx)
        agent = agents[0]

        # The strategy should be an instance of the compiled class
        assert hasattr(agent.strategy, "initialize")
        assert hasattr(agent.strategy, "on_tick")
        assert type(agent.strategy).__name__ == "SimpleTestStrategy"

    def test_creates_multiple_agents(self):
        config = StrategicAgentConfig(strategy_spec=_make_spec())
        rng = np.random.RandomState(42)
        ctx = _make_context()

        agents = config.create_agents(count=3, id_start=10, master_rng=rng, context=ctx)

        assert len(agents) == 3
        for i, agent in enumerate(agents):
            assert agent.id == 10 + i
            assert agent.category == "strategy"

    def test_starting_cash_propagates(self):
        config = StrategicAgentConfig(
            strategy_spec=_make_spec(),
            starting_cash=5_000_000,
        )
        rng = np.random.RandomState(42)
        ctx = _make_context()

        agents = config.create_agents(count=1, id_start=0, master_rng=rng, context=ctx)
        assert agents[0].starting_cash == 5_000_000

    def test_wake_up_freq_propagates(self):
        config = StrategicAgentConfig(
            strategy_spec=_make_spec(),
            wake_up_freq="5S",
        )
        rng = np.random.RandomState(42)
        ctx = _make_context()

        agents = config.create_agents(count=1, id_start=0, master_rng=rng, context=ctx)
        from abides_core.utils import str_to_ns

        assert agents[0].wake_up_freq_ns == str_to_ns("5S")

    def test_log_orders_from_context(self):
        config = StrategicAgentConfig(strategy_spec=_make_spec())
        rng = np.random.RandomState(42)
        ctx = _make_context(log_orders=False)

        agents = config.create_agents(count=1, id_start=0, master_rng=rng, context=ctx)
        assert agents[0].log_orders is False

    def test_log_orders_from_config_overrides_context(self):
        config = StrategicAgentConfig(strategy_spec=_make_spec(), log_orders=True)
        rng = np.random.RandomState(42)
        ctx = _make_context(log_orders=False)

        agents = config.create_agents(count=1, id_start=0, master_rng=rng, context=ctx)
        assert agents[0].log_orders is True

    def test_deterministic_rng(self):
        """Same master RNG seed → same agent RNG state."""
        config = StrategicAgentConfig(strategy_spec=_make_spec())
        ctx = _make_context()

        rng1 = np.random.RandomState(42)
        agents1 = config.create_agents(count=1, id_start=0, master_rng=rng1, context=ctx)

        rng2 = np.random.RandomState(42)
        agents2 = config.create_agents(count=1, id_start=0, master_rng=rng2, context=ctx)

        # Both agents should produce the same random sequence
        s1 = agents1[0].random_state.randint(0, 10000)
        s2 = agents2[0].random_state.randint(0, 10000)
        assert s1 == s2


# ---------------------------------------------------------------------------
# Config serialisation
# ---------------------------------------------------------------------------


class TestConfigSerialisation:
    def test_config_model_dump_round_trip(self):
        """StrategicAgentConfig → dict → StrategicAgentConfig"""
        config = StrategicAgentConfig(strategy_spec=_make_spec(), starting_cash=5_000_000)
        d = config.model_dump()
        restored = StrategicAgentConfig.model_validate(d)
        assert restored.strategy_spec == config.strategy_spec
        assert restored.starting_cash == 5_000_000

    def test_config_json_round_trip(self):
        """StrategicAgentConfig → JSON → StrategicAgentConfig"""
        config = StrategicAgentConfig(strategy_spec=_make_spec())
        json_str = config.model_dump_json()
        restored = StrategicAgentConfig.model_validate_json(json_str)
        assert restored.strategy_spec == config.strategy_spec


# ---------------------------------------------------------------------------
# Integration with hasufel compile pipeline
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCompilePipelineIntegration:
    def test_strategy_agent_in_compiled_runtime(self):
        """A SimulationConfig with rohan_strategy compiles and includes the agent."""
        from rohan.config import SimulationSettings
        from rohan.simulation.abides_impl.config_builder import create_simulation_builder

        settings = SimulationSettings(
            seed=42,
            start_time="09:30:00",
            end_time="09:35:00",
            _env_file=None,  # type: ignore[call-arg]
        )
        spec = _make_spec()
        builder = create_simulation_builder(settings, strategy_spec=spec)
        runtime = builder.build_and_compile()

        # Find the strategic agent (filter by type, not category, since other agents may share the category)
        strategy_agents = [a for a in runtime["agents"] if isinstance(a, StrategicAgentAdapter)]
        assert len(strategy_agents) == 1
        agent = strategy_agents[0]
        assert type(agent.strategy).__name__ == "SimpleTestStrategy"

    def test_latency_model_includes_strategy_agent(self):
        """Latency model dimensions should account for the strategy agent."""
        from rohan.config import SimulationSettings
        from rohan.simulation.abides_impl.config_builder import create_simulation_builder

        settings = SimulationSettings(
            seed=42,
            start_time="09:30:00",
            end_time="09:35:00",
            _env_file=None,  # type: ignore[call-arg]
        )
        spec = _make_spec()
        builder = create_simulation_builder(settings, strategy_spec=spec)
        runtime = builder.build_and_compile()

        n_agents = len(runtime["agents"])
        latency = runtime["agent_latency_model"]
        assert latency.min_latency.shape == (n_agents, n_agents)
