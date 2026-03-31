"""Implementation of the SimulationRunner interface using abides-hasufel.

Uses the declarative SimulationBuilder + compile() pipeline from hasufel,
with manual injection of StrategicAgentAdapter after compilation.
"""

from __future__ import annotations

import hashlib
from typing import override
from uuid import uuid4

import numpy as np
from abides_core.abides import run as abides_run
from abides_markets.config_system import compile as compile_config
from abides_markets.utils import config_add_agents

from rohan.config import SimulationSettings
from rohan.simulation.abides_impl.abides_output import AbidesOutput
from rohan.simulation.abides_impl.config_builder import build_simulation_config
from rohan.simulation.abides_impl.strategic_agent_adapter import StrategicAgentAdapter
from rohan.simulation.models import SimulationOutput
from rohan.simulation.models.strategy_api import StrategicAgent
from rohan.simulation.simulation_runner import SimulationRunner


def _derive_seed(master_seed: int, component: str) -> int:
    """Derive a deterministic seed from master seed + component name (matches hasufel)."""
    h = hashlib.sha256(f"{master_seed}:{component}:0".encode()).digest()
    return int.from_bytes(h[:4], "big")


class SimulationRunnerAbides(SimulationRunner):
    """Hasufel-based implementation of the SimulationRunner interface."""

    def __init__(self, settings: SimulationSettings, strategy: StrategicAgent | None = None):
        super().__init__(settings)
        self.strategy = strategy
        self.strategic_agent_id: int | None = None

    @override
    def run(self) -> SimulationOutput:
        config, oracle_instance = build_simulation_config(self.settings)
        runtime = compile_config(config, oracle_instance=oracle_instance)

        # Inject StrategicAgentAdapter after compilation
        if self.strategy is not None:
            self.strategic_agent_id = len(runtime["agents"])
            seed = runtime["seed"]
            agent_rng = np.random.RandomState(seed=_derive_seed(seed, "rohan_strategy"))
            latency_rng = np.random.RandomState(seed=_derive_seed(seed, "latency"))
            adapter = StrategicAgentAdapter(
                id=self.strategic_agent_id,
                strategy=self.strategy,
                symbol=self.settings.ticker,
                starting_cash=self.settings.starting_cash,
                wake_up_freq="1S",
                order_book_depth=10,
                log_orders=self.settings.log_orders,
                random_state=agent_rng,
            )
            config_add_agents(runtime, [adapter], latency_rng)

        end_state = abides_run(
            runtime,
            log_dir=uuid4().hex,
            kernel_random_state=runtime["random_state_kernel"],
        )

        return AbidesOutput(
            end_state,
            strategic_agent_id=self.strategic_agent_id,
            ticker=self.settings.ticker,
        )

    @override
    def cleanup(self) -> None:
        """Cleanup resources after simulation.

        For ABIDES simulations, no specific cleanup is needed.
        """
        pass
