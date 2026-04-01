"""Registered hasufel agent config for Rohan's LLM-generated strategies.

Importing this module triggers ``@register_agent("rohan_strategy", ...)``,
making the agent type available to hasufel's ``compile()`` pipeline.  For
``run_batch()`` workers (spawned processes), pass a ``worker_initializer``
that imports this module so the registration fires in each worker::

    from abides_markets.simulation import run_batch

    def _register_rohan_agents():
        import rohan.simulation.abides_impl.strategic_agent_config

    results = run_batch(configs, worker_initializer=_register_rohan_agents)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from abides_markets.config_system.agent_configs import AgentCreationContext, BaseAgentConfig
from abides_markets.config_system.registry import register_agent

from rohan.simulation.abides_impl.strategic_agent_adapter import StrategicAgentAdapter
from rohan.simulation.models.strategy_spec import StrategySpec


@register_agent(
    "rohan_strategy",
    category="strategy",
    description="LLM-generated trading strategy (Rohan framework-internal)",
    typical_count_range=(1, 1),
)
class StrategicAgentConfig(BaseAgentConfig):
    """Config model for Rohan's strategic agent.

    The ``strategy_spec`` field carries the source code and class name as
    plain strings.  Compilation (``exec()``) is deferred to
    :meth:`create_agents`, which runs inside hasufel's ``compile()``
    pipeline — keeping the config serializable and picklable.
    """

    strategy_spec: StrategySpec
    wake_up_freq: str = "1S"
    order_book_depth: int = 10

    def create_agents(
        self,
        count: int,
        id_start: int,
        master_rng: np.random.RandomState,
        context: AgentCreationContext,
    ) -> list[Any]:
        """Compile the strategy spec and create StrategicAgentAdapters.

        Each agent gets:
        - Its own strategy instance (fresh ``exec()`` + instantiation)
        - A deterministic RNG derived from the master RNG
        - ``category = "strategy"`` for identification via
          ``SimulationResult.get_agents_by_category("strategy")``
        """
        strategy_class = self.strategy_spec.compile()
        log_orders = self.log_orders if self.log_orders is not None else context.log_orders

        agents: list[Any] = []
        for j in range(count):
            strategy = strategy_class(**self.strategy_spec.params)

            agent_rng = np.random.RandomState(seed=master_rng.randint(low=0, high=2**32, dtype="uint64"))

            adapter = StrategicAgentAdapter(
                id=id_start + j,
                strategy=strategy,
                symbol=context.ticker,
                starting_cash=self.starting_cash,
                wake_up_freq=self.wake_up_freq,
                order_book_depth=self.order_book_depth,
                log_orders=log_orders,
                random_state=agent_rng,
            )
            adapter.category = "strategy"
            agents.append(adapter)

        return agents
