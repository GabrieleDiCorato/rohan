"""Implementation of the SimulationRunner interface using abides-hasufel.

Uses the declarative SimulationBuilder + compile() pipeline from hasufel.
When a StrategySpec is provided, the ``rohan_strategy`` registered agent
type is included in the config so hasufel's compile() handles
instantiation, seed derivation, and latency model construction.

A pre-built StrategicAgent instance can also be injected directly (useful
for tests that need to inspect strategy state after simulation).
"""

from __future__ import annotations

from typing import override
from uuid import uuid4

import numpy as np
from abides_core.abides import run as abides_run
from abides_markets.config_system.compiler import derive_seed
from abides_markets.utils import config_add_agents

from rohan.config import SimulationSettings
from rohan.simulation.abides_impl.abides_output import AbidesOutput
from rohan.simulation.abides_impl.config_builder import create_simulation_builder
from rohan.simulation.abides_impl.strategic_agent_adapter import StrategicAgentAdapter
from rohan.simulation.models import SimulationOutput
from rohan.simulation.models.strategy_api import StrategicAgent
from rohan.simulation.models.strategy_spec import StrategySpec
from rohan.simulation.simulation_runner import SimulationRunner


class SimulationRunnerAbides(SimulationRunner):
    """Hasufel-based implementation of the SimulationRunner interface.

    Two injection modes:

    1. **StrategySpec** (preferred): serializable spec goes through
       hasufel's config → compile pipeline via the registered
       ``rohan_strategy`` agent type.
    2. **StrategicAgent instance** (legacy): pre-built instance is
       injected post-compile via ``config_add_agents()``.  Useful
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
        runtime = builder.build_and_compile()

        strategic_agent_id: int | None = None

        # Legacy path: inject pre-built strategy instance post-compile
        if self._strategy_instance is not None:
            strategic_agent_id = len(runtime["agents"])
            seed = runtime["seed"]
            agent_rng = np.random.RandomState(seed=derive_seed(seed, "rohan_strategy"))
            latency_rng = np.random.RandomState(seed=derive_seed(seed, "runtime_agents"))
            adapter = StrategicAgentAdapter(
                id=strategic_agent_id,
                strategy=self._strategy_instance,
                symbol=self.settings.ticker,
                starting_cash=self.settings.starting_cash,
                wake_up_freq="1S",
                order_book_depth=10,
                log_orders=self.settings.log_orders,
                random_state=agent_rng,
            )
            adapter.category = "strategy"
            config_add_agents(runtime, [adapter], latency_rng)

        end_state = abides_run(
            runtime,
            log_dir=uuid4().hex,
            kernel_random_state=runtime["random_state_kernel"],
        )

        # Find strategic agent ID by category (StrategySpec path)
        if self.strategy_spec is not None:
            for agent in end_state["agents"]:
                if getattr(agent, "category", "") == "strategy":
                    strategic_agent_id = agent.id
                    break

        return AbidesOutput(
            end_state,
            strategic_agent_id=strategic_agent_id,
            ticker=self.settings.ticker,
        )

    @override
    def cleanup(self) -> None:
        """Cleanup resources after simulation.

        For ABIDES simulations, no specific cleanup is needed.
        """
        pass
