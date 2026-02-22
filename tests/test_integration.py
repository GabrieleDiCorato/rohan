"""Minimal integration test — full vertical slice without LLM (§2.7.8d).

Generates a trivial no-op strategy, passes it through StrategyValidator →
execute_strategy_safely → metric computation, and asserts the full chain
produces valid, non-None output.
"""

from __future__ import annotations

import pytest

from rohan.config import SimulationSettings
from rohan.framework.analysis_service import AnalysisService
from rohan.framework.prompts import format_interpreter_prompt
from rohan.simulation import ComparisonResult, MarketImpact, MarketMetrics, RunSummary
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


# ---------------------------------------------------------------------------
# Item 18 — 30-minute end-to-end test (slow, excluded from default run)
# ---------------------------------------------------------------------------


def _30min_settings() -> SimulationSettings:
    """30-minute simulation with enough agents to trigger all microstructure metrics.

    Uses 2 adaptive market makers to ensure a persistent two-sided book,
    200 noise agents to create spread shocks (needed for resilience), and
    20 value agents to generate directional order flow (needed for VPIN).
    A fixed seed makes the test reproducible.
    """
    s = SimulationSettings(seed=42)
    s.start_time = "09:30:00"
    s.end_time = "10:00:00"
    s.agents.noise.num_agents = 200
    s.agents.value.num_agents = 20
    s.agents.momentum.num_agents = 0
    s.agents.adaptive_market_maker.num_agents = 2
    return s


@pytest.mark.slow
class TestLongSimulation:
    """End-to-end test for a 30-minute simulation.

    Verifies that all microstructure metrics are non-None when there is
    sufficient market activity, and that format_interpreter_prompt() produces
    a valid prompt containing the expected metric labels.

    Run explicitly with: pytest -m slow
    Excluded from the default test run via pyproject.toml addopts.
    """

    def test_30min_all_metrics_non_none(self):
        """All market-level microstructure metrics should be computable in a
        30-minute simulation with 200 noise agents, 20 value agents, and
        2 adaptive market makers."""
        # 1. Execute
        settings = _30min_settings()
        sim_result = execute_strategy_safely(NOOP_STRATEGY, settings, timeout_seconds=300)
        assert sim_result.error is None, f"Simulation failed: {sim_result.error}"
        output = sim_result.result
        assert output is not None

        # 2. Compute market metrics
        analyzer = AnalysisService()
        sim_metrics = analyzer.compute_metrics(output)

        assert sim_metrics.volatility is not None, "Volatility should be computable in 30 min"
        assert sim_metrics.vpin is not None, "VPIN should be computable with enough fills"
        assert sim_metrics.lob_imbalance_mean is not None, "LOB imbalance should be computable from two-sided rows"
        assert sim_metrics.resilience_mean_ns is not None, "Market resilience should be detectable with 200 noise agents"
        assert sim_metrics.market_ott_ratio is not None, "Market OTT ratio should be computable with any fills"

        # 3. Build RunSummary and verify format_interpreter_prompt does not raise
        agent_id = output.strategic_agent_id
        assert agent_id is not None
        agent_metrics = analyzer.compute_agent_metrics(output, agent_id, initial_cash=settings.starting_cash)
        market_metrics = MarketMetrics(
            volatility=sim_metrics.volatility,
            mean_spread=sim_metrics.mean_spread,
            effective_spread=sim_metrics.effective_spread,
            avg_bid_liquidity=sim_metrics.avg_bid_liquidity,
            avg_ask_liquidity=sim_metrics.avg_ask_liquidity,
            traded_volume=sim_metrics.traded_volume,
            lob_imbalance_mean=sim_metrics.lob_imbalance_mean,
            lob_imbalance_std=sim_metrics.lob_imbalance_std,
            vpin=sim_metrics.vpin,
            resilience_mean_ns=sim_metrics.resilience_mean_ns,
            market_ott_ratio=sim_metrics.market_ott_ratio,
        )
        comparison = ComparisonResult(
            strategy_metrics=agent_metrics,
            strategy_market_metrics=market_metrics,
            baseline_metrics=MarketMetrics(),
            market_impact=MarketImpact(),
        )
        summary = RunSummary(comparison=comparison)

        prompt = format_interpreter_prompt(summary, goal="30-min integration test")
        assert "VPIN" in prompt, "Formatted prompt should reference VPIN"
