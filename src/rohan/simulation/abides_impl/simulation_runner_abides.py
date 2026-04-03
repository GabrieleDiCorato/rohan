"""Implementation of the SimulationRunner interface using abides-hasufel.

Uses the declarative SimulationBuilder + ``run_simulation()`` pipeline from
hasufel.  When a StrategySpec is provided, the ``rohan_strategy`` registered
agent type is included in the config so hasufel's compile() handles
instantiation, seed derivation, and latency model construction.

A pre-built StrategicAgent instance can also be injected via hasufel's
``runtime_agents`` parameter (useful for tests that inspect strategy state
after simulation).
"""

from __future__ import annotations

import logging
from typing import override

import numpy as np
from abides_markets.agents import TradingAgent
from abides_markets.config_system import validate_config
from abides_markets.config_system.compiler import derive_seed
from abides_markets.simulation import ResultProfile, run_simulation

from rohan.config import OracleType, SimulationSettings
from rohan.simulation.abides_impl.config_builder import create_simulation_builder
from rohan.simulation.abides_impl.hasufel_output import HasufelOutput
from rohan.simulation.abides_impl.strategic_agent_adapter import StrategicAgentAdapter
from rohan.simulation.models import SimulationOutput
from rohan.simulation.models.strategy_api import StrategicAgent
from rohan.simulation.models.strategy_spec import StrategySpec
from rohan.simulation.simulation_runner import SimulationRunner

logger = logging.getLogger(__name__)


class SimulationRunnerAbides(SimulationRunner):
    """Hasufel-based implementation of the SimulationRunner interface.

    Two injection modes:

    1. **StrategySpec** (preferred): serializable spec goes through
       hasufel's config → compile pipeline via the registered
       ``rohan_strategy`` agent type.
    2. **StrategicAgent instance** (legacy): pre-built instance is
       injected via ``run_simulation(runtime_agents=[...])``.  Useful
       for tests that need to inspect strategy state.
    """

    def __init__(
        self,
        settings: SimulationSettings,
        strategy_spec: StrategySpec | None = None,
        strategy: StrategicAgent | None = None,
    ):
        super().__init__(settings)
        if strategy_spec is not None and strategy is not None:
            raise ValueError("Cannot specify both strategy_spec and strategy")
        self.strategy_spec = strategy_spec
        self._strategy_instance = strategy

    @override
    def run(self) -> SimulationOutput:
        builder = create_simulation_builder(self.settings, strategy_spec=self.strategy_spec)
        config = builder.build()

        # Validate config against hasufel's cross-agent constraints.
        # The builder computes derived fields (e.g. kappa from
        # mean_reversion_half_life), so we downgrade errors to warnings
        # since the builder output is structurally valid by construction.
        validation = validate_config(config.model_dump(mode="json"))
        if not validation.valid:
            msgs = [getattr(e, "message", str(e)) for e in validation.errors]
            logger.warning("Hasufel config validation: %s", "; ".join(msgs))
        for w in validation.warnings:
            logger.warning("Hasufel config warning: %s", getattr(w, "message", str(w)))

        # Extract oracle instance for historical mode (ExternalDataOracleConfig
        # is a marker type — the real oracle must be injected at runtime).
        oracle_instance = None
        if self.settings.agents.oracle.oracle_type == OracleType.HISTORICAL:
            oracle_instance = builder.get_oracle_instance()

        # Legacy path: build a runtime agent for pre-built strategy instance
        runtime_agents: list[TradingAgent] | None = None
        if self._strategy_instance is not None:
            seed = self.settings.seed
            agent_rng = np.random.RandomState(seed=derive_seed(seed, "rohan_strategy"))
            adapter = StrategicAgentAdapter(
                id=-1,  # run_simulation() will assign a real ID
                strategy=self._strategy_instance,
                symbol=self.settings.ticker,
                starting_cash=self.settings.starting_cash,
                wake_up_freq="1S",
                order_book_depth=10,
                log_orders=self.settings.log_orders,
                random_state=agent_rng,
            )
            adapter.category = "strategy"
            runtime_agents = [adapter]

        hasufel_result = run_simulation(
            config,
            profile=ResultProfile.FULL,
            oracle_instance=oracle_instance,
            runtime_agents=runtime_agents,
        )

        # Discover strategic agent ID from the result
        strategic_agent_id: int | None = None
        strategy_agents = hasufel_result.get_agents_by_category("strategy")
        if strategy_agents:
            strategic_agent_id = strategy_agents[0].agent_id

        return HasufelOutput(
            hasufel_result,
            ticker=self.settings.ticker,
            strategic_agent_id=strategic_agent_id,
            compiled_config=config.model_dump(mode="json"),
        )

    @override
    def cleanup(self) -> None:
        """Cleanup resources after simulation.

        For ABIDES simulations, no specific cleanup is needed.
        """
        pass
