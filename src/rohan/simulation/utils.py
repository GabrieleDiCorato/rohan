from typing import Any

from rohan.config import SimulationSettings
from rohan.framework.analysis_service import AnalysisService
from rohan.simulation import (
    ComparisonResult,
    MarketMetrics,
)
from rohan.simulation.simulation_service import SimulationService
from rohan.simulation.strategy_validator import (
    execute_strategy_safely,
)


def run_with_baseline(
    strategy_code: str,
    settings: SimulationSettings,
    baseline_override: dict[str, Any] | None = None,
    timeout_seconds: int = 300,
) -> ComparisonResult:
    """
    Runs two simulations:
    1. With the provided strategy code injected via execute_strategy_safely.
    2. A baseline run (strategy disabled/default).

    Returns a ComparisonResult containing metrics for both runs and the delta.
    """

    # 1. Run Strategy
    # execute_strategy_safely handles validation, compilation, and execution with timeout
    res1 = execute_strategy_safely(strategy_code, settings, timeout_seconds)
    if res1.error:
        raise RuntimeError(f"Strategy run failed: {res1.error}")

    if not res1.result:
        raise RuntimeError("Strategy run returned no result")

    # Compute agent metrics (Assumes strategy agent is agent_id=1)
    # TODO: Dynamically find the strategic agent ID if not 1
    metrics1 = AnalysisService.compute_agent_metrics(res1.result, 1)

    # Compute market metrics for strategy run (for impact comparison)
    metrics1_sim = AnalysisService.compute_metrics(res1.result)

    # 2. Run Baseline
    # Copy settings and modify for baseline
    baseline_settings = settings.model_copy(deep=True)
    if baseline_override:
        # Simple override of top-level fields or nested dict update
        for k, v in baseline_override.items():
            if hasattr(baseline_settings, k):
                setattr(baseline_settings, k, v)

    service = SimulationService()

    # Baseline run with no strategy (uses default configuration)
    res2 = service.run_simulation(baseline_settings, strategy=None)
    if res2.error:
        raise RuntimeError(f"Baseline run failed: {res2.error}")

    if not res2.result:
        raise RuntimeError("Baseline run returned no result")

    # Compute market metrics for baseline
    metrics2_sim = AnalysisService.compute_metrics(res2.result)

    metrics2 = MarketMetrics(
        volatility=metrics2_sim.volatility,
        mean_spread=metrics2_sim.custom_metrics.get("mean_spread", 0.0),
        avg_bid_liquidity=metrics2_sim.custom_metrics.get("avg_bid_liquidity", 0.0),
        avg_ask_liquidity=metrics2_sim.custom_metrics.get("avg_ask_liquidity", 0.0),
        traded_volume=metrics2_sim.traded_volume,
    )

    # Compare
    # Market Impact: Strategy Volatility vs Baseline Volatility
    impact = {
        "volatility_delta": metrics1_sim.volatility - metrics2.volatility,
        "spread_delta": metrics1_sim.custom_metrics.get("mean_spread", 0.0) - metrics2.mean_spread,
        "liquidity_bid_delta": metrics1_sim.custom_metrics.get("avg_bid_liquidity", 0.0) - metrics2.avg_bid_liquidity,
        "liquidity_ask_delta": metrics1_sim.custom_metrics.get("avg_ask_liquidity", 0.0) - metrics2.avg_ask_liquidity,
    }

    return ComparisonResult(strategy_metrics=metrics1, baseline_metrics=metrics2, market_impact=impact)
