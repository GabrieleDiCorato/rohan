from unittest.mock import MagicMock, patch

import pytest

from rohan.simulation.models import AgentMetrics, ComparisonResult, SimulationMetrics


def test_run_with_baseline_impact_calculation():
    """Test that run_with_baseline correctly calculates percentage change."""

    # Setup mock returns
    agent_metrics = AgentMetrics(agent_id=1)

    # Strategy Run: High Volatility
    metrics1_sim = SimulationMetrics(
        volatility=0.15,
        mean_spread=2.0,  # 2 cents
        avg_bid_liquidity=100.0,
        avg_ask_liquidity=100.0,
        traded_volume=1000,
    )

    # Baseline Run: Lower Volatility
    metrics2_sim = SimulationMetrics(
        volatility=0.10,
        mean_spread=1.0,  # 1 cent
        avg_bid_liquidity=200.0,
        avg_ask_liquidity=200.0,
        traded_volume=500,
    )

    # Mock settings
    mock_settings = MagicMock()
    mock_settings.model_copy.return_value = mock_settings

    # Mock simulation results
    res1 = MagicMock()
    res1.error = None
    res1.result = MagicMock()

    res2 = MagicMock()
    res2.error = None
    res2.result = MagicMock()

    with (
        patch("rohan.simulation.utils.execute_strategy_safely", return_value=res1),
        patch("rohan.simulation.utils.SimulationService") as mock_service_cls,
        patch("rohan.simulation.utils.AnalysisService") as mock_analysis,
    ):
        mock_service = mock_service_cls.return_value
        mock_service.run_simulation.return_value = res2

        mock_analysis.compute_agent_metrics.return_value = agent_metrics
        mock_analysis.compute_metrics.side_effect = [metrics1_sim, metrics2_sim]

        # Import here to ensure patches are applied
        from rohan.simulation.utils import run_with_baseline

        result = run_with_baseline("code", mock_settings)

    # Verify Structure
    assert isinstance(result, ComparisonResult)

    # 1. Verify Strategy Market Metrics (Raw Values) are present
    assert hasattr(result, "strategy_market_metrics")
    assert result.strategy_market_metrics.volatility == 0.15
    assert result.strategy_market_metrics.mean_spread == 2.0

    # 2. Verify Baseline Metrics (Raw Values) are present
    assert result.baseline_metrics.volatility == 0.10
    assert result.baseline_metrics.mean_spread == 1.0

    # 3. Verify Percentage Changes (MarketImpact typed model)
    # Volatility: (0.15 - 0.10) / 0.10 = 0.50 (+50%)
    assert result.market_impact.volatility_delta_pct == pytest.approx(0.50)

    # Spread: (0.02 - 0.01) / 0.01 = 1.00 (+100%)
    assert result.market_impact.spread_delta_pct == pytest.approx(1.00)

    # Liquidity Bid: (100 - 200) / 200 = -0.50 (-50%)
    assert result.market_impact.bid_liquidity_delta_pct == pytest.approx(-0.50)

    # Liquidity Ask: (100 - 200) / 200 = -0.50 (-50%)
    assert result.market_impact.ask_liquidity_delta_pct == pytest.approx(-0.50)
