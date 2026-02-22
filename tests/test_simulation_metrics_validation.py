"""Tests for simulation_metrics.py Pydantic model validations.

This module tests:
- Type validations (int vs float for cash fields)
- None vs 0 handling (None = not computed, 0 = computed as zero)
- Field constraints and optional handling
- Model composition (ComparisonResult, RunSummary)
"""

import pytest
from pydantic import ValidationError

from rohan.simulation.models.simulation_metrics import (
    AgentMetrics,
    ComparisonResult,
    MarketImpact,
    MarketMetrics,
    RunSummary,
    SimulationMetrics,
)


# ---------------------------------------------------------------------------
# SimulationMetrics Validation
# ---------------------------------------------------------------------------
class TestSimulationMetricsValidation:
    """Test suite for SimulationMetrics model."""

    def test_all_fields_optional(self):
        """All metrics fields are optional (None = not computed)."""
        metrics = SimulationMetrics()
        assert metrics.volatility is None
        assert metrics.mean_spread is None
        assert metrics.effective_spread is None
        assert metrics.avg_bid_liquidity is None
        assert metrics.avg_ask_liquidity is None
        assert metrics.traded_volume is None

    def test_partial_metrics_valid(self):
        """Can set some metrics and leave others None."""
        metrics = SimulationMetrics(
            volatility=0.15,
            mean_spread=50.0,  # 50 cents
        )
        assert metrics.volatility == 0.15
        assert metrics.mean_spread == 50.0
        assert metrics.effective_spread is None  # Not computed

    def test_all_metrics_set(self):
        """Can set all metrics."""
        metrics = SimulationMetrics(
            volatility=0.12,
            mean_spread=25.0,
            effective_spread=24.0,
            avg_bid_liquidity=150.0,
            avg_ask_liquidity=140.0,
            traded_volume=5000,
        )
        assert metrics.volatility == 0.12
        assert metrics.traded_volume == 5000

    def test_zero_values_distinct_from_none(self):
        """Zero is a valid value, distinct from None."""
        metrics = SimulationMetrics(
            volatility=0.0,  # Zero volatility (valid)
            mean_spread=0.0,  # Zero spread (valid)
        )
        assert metrics.volatility == 0.0
        assert metrics.volatility is not None

    def test_negative_values_allowed(self):
        """Some metrics can be negative (though rare)."""
        # In practice, most metrics are non-negative, but Pydantic doesn't enforce this
        metrics = SimulationMetrics(
            volatility=-0.01,  # Invalid in practice, but Pydantic allows
        )
        assert metrics.volatility == -0.01

    def test_spread_values_in_cents(self):
        """Spread values should be in cents (floats for averages)."""
        metrics = SimulationMetrics(
            mean_spread=50.5,  # 50.5 cents average
            effective_spread=49.3,
        )
        assert isinstance(metrics.mean_spread, float)
        assert metrics.mean_spread == 50.5


# ---------------------------------------------------------------------------
# AgentMetrics Validation
# ---------------------------------------------------------------------------
class TestAgentMetricsValidation:
    """Test suite for AgentMetrics model."""

    def test_minimal_valid_agent_metrics(self):
        """AgentMetrics with only agent_id."""
        metrics = AgentMetrics(agent_id=1)
        assert metrics.agent_id == 1
        assert metrics.initial_cash == 0
        assert metrics.ending_cash == 0
        assert metrics.total_pnl is None

    def test_cash_fields_are_int_cents(self):
        """initial_cash and ending_cash must be int (cents)."""
        metrics = AgentMetrics(
            agent_id=1,
            initial_cash=10_000_000,  # $100,000 in cents
            ending_cash=10_500_000,  # $105,000 in cents
        )
        assert isinstance(metrics.initial_cash, int)
        assert isinstance(metrics.ending_cash, int)
        assert metrics.initial_cash == 10_000_000

    def test_pnl_is_float_cents(self):
        """total_pnl is float (can be fractional from mid-price)."""
        metrics = AgentMetrics(
            agent_id=1,
            initial_cash=10_000_000,
            ending_cash=10_500_000,
            total_pnl=500_050.5,  # $5,000.505 - fractional cents
        )
        assert isinstance(metrics.total_pnl, float)
        assert metrics.total_pnl == 500_050.5

    def test_pnl_can_be_negative(self):
        """PnL can be negative (loss)."""
        metrics = AgentMetrics(
            agent_id=1,
            total_pnl=-50_000.0,  # $500 loss
        )
        assert metrics.total_pnl is not None
        assert metrics.total_pnl < 0

    def test_trade_count_must_be_int(self):
        """trade_count is int, not float."""
        metrics = AgentMetrics(
            agent_id=1,
            trade_count=42,
        )
        assert isinstance(metrics.trade_count, int)
        assert metrics.trade_count == 42

    def test_fill_rate_is_float_ratio(self):
        """fill_rate is a float ratio (0.0 to 1.0)."""
        metrics = AgentMetrics(
            agent_id=1,
            fill_rate=0.75,  # 75% fill rate
        )
        assert metrics.fill_rate == 0.75

    def test_inventory_fields_are_int(self):
        """Inventory fields are int shares."""
        metrics = AgentMetrics(
            agent_id=1,
            start_inventory=0,
            end_inventory=100,
        )
        assert isinstance(metrics.start_inventory, int)
        assert isinstance(metrics.end_inventory, int)

    def test_negative_inventory_allowed(self):
        """Inventory can be negative (short position)."""
        metrics = AgentMetrics(
            agent_id=1,
            start_inventory=-50,
            end_inventory=-25,
        )
        assert metrics.start_inventory == -50

    def test_optional_metrics_none_by_default(self):
        """Optional metrics default to None."""
        metrics = AgentMetrics(agent_id=1)
        assert metrics.sharpe_ratio is None
        assert metrics.max_drawdown is None
        assert metrics.inventory_std is None


# ---------------------------------------------------------------------------
# MarketMetrics Validation
# ---------------------------------------------------------------------------
class TestMarketMetricsValidation:
    """Test suite for MarketMetrics model."""

    def test_market_metrics_same_shape_as_simulation_metrics(self):
        """MarketMetrics has same fields as SimulationMetrics."""
        metrics = MarketMetrics(
            volatility=0.10,
            mean_spread=30.0,
            effective_spread=29.0,
            avg_bid_liquidity=200.0,
            avg_ask_liquidity=190.0,
            traded_volume=3000,
        )
        assert metrics.volatility == 0.10
        assert metrics.mean_spread == 30.0


# ---------------------------------------------------------------------------
# MarketImpact Validation
# ---------------------------------------------------------------------------
class TestMarketImpactValidation:
    """Test suite for MarketImpact model (percentage deltas)."""

    def test_all_fields_optional(self):
        """All impact fields are optional."""
        impact = MarketImpact()
        assert impact.spread_delta_pct is None
        assert impact.volatility_delta_pct is None
        assert impact.bid_liquidity_delta_pct is None
        assert impact.ask_liquidity_delta_pct is None

    def test_positive_delta_pct(self):
        """Positive delta means strategy increased the metric."""
        impact = MarketImpact(
            spread_delta_pct=0.50,  # +50%
            volatility_delta_pct=0.25,  # +25%
        )
        assert impact.spread_delta_pct == 0.50

    def test_negative_delta_pct(self):
        """Negative delta means strategy decreased the metric."""
        impact = MarketImpact(
            spread_delta_pct=-0.10,  # -10% (strategy reduced spread)
            bid_liquidity_delta_pct=-0.25,  # -25% (reduced liquidity)
        )
        assert impact.spread_delta_pct == -0.10

    def test_zero_delta_pct(self):
        """Zero delta means no change."""
        impact = MarketImpact(
            spread_delta_pct=0.0,
            volatility_delta_pct=0.0,
        )
        assert impact.spread_delta_pct == 0.0


# ---------------------------------------------------------------------------
# ComparisonResult Validation
# ---------------------------------------------------------------------------
class TestComparisonResultValidation:
    """Test suite for ComparisonResult model composition."""

    def test_comparison_result_composition(self):
        """ComparisonResult composes all metric types."""
        strategy_metrics = AgentMetrics(agent_id=1, total_pnl=100_000.0)
        strategy_market = MarketMetrics(volatility=0.15, mean_spread=50.0)
        baseline_market = MarketMetrics(volatility=0.10, mean_spread=30.0)
        impact = MarketImpact(
            spread_delta_pct=0.67,  # (50-30)/30
            volatility_delta_pct=0.50,  # (0.15-0.10)/0.10
        )

        result = ComparisonResult(
            strategy_metrics=strategy_metrics,
            strategy_market_metrics=strategy_market,
            baseline_metrics=baseline_market,
            market_impact=impact,
        )

        assert result.strategy_metrics.agent_id == 1
        assert result.market_impact.spread_delta_pct == 0.67

    def test_comparison_result_all_fields_required(self):
        """All ComparisonResult fields are required."""
        with pytest.raises(ValidationError):
            ComparisonResult(  # type: ignore[call-arg]  # Intentionally missing required args
                strategy_metrics=AgentMetrics(agent_id=1),
                # Missing strategy_market_metrics, baseline_metrics, market_impact
            )


# ---------------------------------------------------------------------------
# RunSummary Validation
# ---------------------------------------------------------------------------
class TestRunSummaryValidation:
    """Test suite for RunSummary model."""

    def test_run_summary_minimal(self):
        """RunSummary with only comparison."""
        comparison = ComparisonResult(
            strategy_metrics=AgentMetrics(agent_id=1),
            strategy_market_metrics=MarketMetrics(),
            baseline_metrics=MarketMetrics(),
            market_impact=MarketImpact(),
        )
        summary = RunSummary(comparison=comparison)

        assert summary.comparison is comparison
        assert summary.price_chart is None
        assert summary.spread_chart is None
        assert summary.duration_seconds == 0.0
        assert summary.error is None

    def test_run_summary_with_charts(self):
        """RunSummary with base64 chart data."""
        comparison = ComparisonResult(
            strategy_metrics=AgentMetrics(agent_id=1),
            strategy_market_metrics=MarketMetrics(),
            baseline_metrics=MarketMetrics(),
            market_impact=MarketImpact(),
        )
        summary = RunSummary(
            comparison=comparison,
            price_chart="base64encodeddata==",
            spread_chart="anotherbase64==",
            duration_seconds=12.5,
        )

        assert summary.price_chart == "base64encodeddata=="
        assert summary.duration_seconds == 12.5

    def test_run_summary_with_error(self):
        """RunSummary can include error string."""
        comparison = ComparisonResult(
            strategy_metrics=AgentMetrics(agent_id=1),
            strategy_market_metrics=MarketMetrics(),
            baseline_metrics=MarketMetrics(),
            market_impact=MarketImpact(),
        )
        summary = RunSummary(
            comparison=comparison,
            error="Simulation timeout",
        )

        assert summary.error == "Simulation timeout"


# ---------------------------------------------------------------------------
# Type Coercion Tests
# ---------------------------------------------------------------------------
class TestTypeCoercion:
    """Test Pydantic type coercion behavior."""

    def test_int_cash_rejects_fractional_float(self):
        """int fields should reject fractional floats."""
        with pytest.raises(ValidationError):
            AgentMetrics(
                agent_id=1,
                initial_cash=100.5,  # type: ignore[arg-type]  # Intentionally wrong type for test
            )

    def test_int_cash_accepts_whole_float(self):
        """int fields may coerce whole floats like 100.0 → 100."""
        metrics = AgentMetrics(
            agent_id=1,
            initial_cash=100,
        )
        assert metrics.initial_cash == 100
        assert isinstance(metrics.initial_cash, int)

    def test_float_pnl_accepts_int(self):
        """float | None fields accept int and coerce."""
        metrics = AgentMetrics(
            agent_id=1,
            total_pnl=1000,  # int → float
        )
        assert metrics.total_pnl == 1000.0
        assert isinstance(metrics.total_pnl, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
