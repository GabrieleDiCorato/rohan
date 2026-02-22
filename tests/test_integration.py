"""Minimal integration test — full vertical slice without LLM (§2.7.8d).

Generates a trivial no-op strategy, passes it through StrategyValidator →
execute_strategy_safely → metric computation, and asserts the full chain
produces valid, non-None output.
"""

from __future__ import annotations

from rohan.config import SimulationSettings
from rohan.framework.analysis_service import AnalysisService
from rohan.simulation.strategy_validator import StrategyValidator, execute_strategy_safely

# A no-op strategy: does nothing on market data, returns empty actions.
NOOP_STRATEGY = """\
from rohan.simulation.models.strategy_api import (
    AgentConfig, MarketState, OrderAction, OrderType, Side,
)


class NoOpStrategy:
    def initialize(self, config: AgentConfig) -> None:
        self.symbol = config.symbol

    def on_tick(self, state: MarketState) -> list[OrderAction]:
        return []

    def on_market_data(self, state: MarketState) -> list[OrderAction]:
        return []

    def on_order_update(self, update) -> list[OrderAction]:
        return []

    def on_simulation_end(self, final_state: MarketState) -> None:
        pass
"""


def _short_settings() -> SimulationSettings:
    """5-minute simulation with minimal agents for speed.

    Uses a fixed seed so the test is fully reproducible across runs and
    doesn't depend on wall-clock timing.  Agent counts are chosen to be
    large enough to guarantee L1 order-book snapshots are produced (at
    least one bid and one ask should appear within 5 minutes).
    """
    s = SimulationSettings(seed=42)
    s.start_time = "09:30:00"
    s.end_time = "09:35:00"
    s.agents.noise.num_agents = 10
    s.agents.value.num_agents = 5
    s.agents.momentum.num_agents = 0
    s.agents.adaptive_market_maker.num_agents = 0
    return s


class TestIntegrationVerticalSlice:
    """Integration test: validator → executor → metrics (no LLM)."""

    def test_noop_strategy_full_pipeline(self):
        """A trivial no-op strategy should pass validation, execute, and
        produce non-None metrics without raising any exceptions."""
        # 1. Validate
        validator = StrategyValidator()
        validation = validator.validate(NOOP_STRATEGY)
        assert validation.is_valid, f"Validation failed: {validation.errors}"

        # 2. Execute with a short simulation
        settings = _short_settings()
        sim_result = execute_strategy_safely(NOOP_STRATEGY, settings, timeout_seconds=120)

        assert sim_result.error is None, f"Execution failed: {sim_result.error}"
        assert sim_result.result is not None

        output = sim_result.result

        # 3. Verify output structure
        l1 = output.get_order_book_l1()
        assert not l1.empty, "L1 order book should not be empty"
        assert "bid_price" in l1.columns
        assert "ask_price" in l1.columns

        logs = output.get_logs_df()
        assert not logs.empty, "Agent logs should not be empty"

        # 4. Compute metrics
        analyzer = AnalysisService()
        sim_metrics = analyzer.compute_metrics(output)
        assert sim_metrics is not None
        # At least one metric should be computable even for a minimal sim
        all_fields = [
            sim_metrics.volatility,
            sim_metrics.mean_spread,
            sim_metrics.effective_spread,
            sim_metrics.avg_bid_liquidity,
            sim_metrics.avg_ask_liquidity,
            sim_metrics.traded_volume,
        ]
        assert any(v is not None for v in all_fields), f"Expected at least one non-None metric, got: {sim_metrics}"

        # 5. Compute agent metrics (strategic agent)
        agent_id = output.strategic_agent_id
        assert agent_id is not None, "Strategic agent ID should be set"

        agent_metrics = analyzer.compute_agent_metrics(output, agent_id, initial_cash=settings.starting_cash)
        assert agent_metrics is not None
        assert agent_metrics.initial_cash == settings.starting_cash
        # No-op strategy should have 0 trades
        assert agent_metrics.trade_count == 0

    def test_noop_strategy_duration_tracked(self):
        """SimulationResult.duration_seconds should be > 0."""
        settings = _short_settings()
        sim_result = execute_strategy_safely(NOOP_STRATEGY, settings, timeout_seconds=120)

        assert sim_result.error is None
        assert sim_result.duration_seconds > 0
