from typing import Any

from rohan.config import SimulationSettings
from rohan.framework.analysis_service import AnalysisService
from rohan.simulation import (
    ComparisonResult,
    MarketImpact,
)
from rohan.simulation.abides_impl.hasufel_output import HasufelOutput
from rohan.simulation.simulation_service import SimulationService
from rohan.simulation.strategy_validator import (
    execute_strategy_safely,
)


def _pct_change(a: float | None, b: float | None) -> float | None:
    """Percentage change (a − b) / b.  Returns None when undefined."""
    if a is None or b is None:
        return None
    if b == 0:
        return 0.0 if a == 0 else None  # undefined when baseline is zero
    return (a - b) / b


def run_with_baseline(
    strategy_code: str,
    settings: SimulationSettings,
    baseline_override: dict[str, Any] | None = None,
    timeout_seconds: int | None = None,
) -> ComparisonResult:
    """Run strategy + baseline simulations and return a typed ComparisonResult."""

    # 1. Run Strategy
    effective_timeout = timeout_seconds if timeout_seconds is not None else settings.timeout_seconds
    res1 = execute_strategy_safely(strategy_code, settings, effective_timeout)
    if res1.error:
        raise RuntimeError(f"Strategy run failed: {res1.error}")
    if not res1.result:
        raise RuntimeError("Strategy run returned no result")
    assert isinstance(res1.result, HasufelOutput)

    if res1.result.strategic_agent_id is None:
        raise RuntimeError("No strategic agent in simulation output")
    metrics1 = AnalysisService.compute_agent_metrics(
        res1.result,
        res1.result.strategic_agent_id,
    )
    metrics1_sim = AnalysisService.compute_metrics(res1.result)

    # 2. Run Baseline
    baseline_settings = settings.model_copy(deep=True)
    if baseline_override:
        for k, v in baseline_override.items():
            if hasattr(baseline_settings, k):
                setattr(baseline_settings, k, v)

    service = SimulationService()
    res2 = service.run_simulation(baseline_settings)
    if res2.error:
        raise RuntimeError(f"Baseline run failed: {res2.error}")
    if not res2.result:
        raise RuntimeError("Baseline run returned no result")
    assert isinstance(res2.result, HasufelOutput)

    metrics2_sim = AnalysisService.compute_metrics(res2.result)

    # Market metrics (SimulationMetrics is now also MarketMetrics)
    strat_market = metrics1_sim
    base_market = metrics2_sim

    # Compute typed market impact
    impact = MarketImpact(
        spread_delta_pct=_pct_change(strat_market.mean_spread, base_market.mean_spread),
        volatility_delta_pct=_pct_change(strat_market.volatility, base_market.volatility),
        bid_liquidity_delta_pct=_pct_change(strat_market.avg_bid_liquidity, base_market.avg_bid_liquidity),
        ask_liquidity_delta_pct=_pct_change(strat_market.avg_ask_liquidity, base_market.avg_ask_liquidity),
        lob_imbalance_delta_pct=_pct_change(strat_market.lob_imbalance_mean, base_market.lob_imbalance_mean),
        vpin_delta_pct=_pct_change(strat_market.vpin, base_market.vpin),
        resilience_delta_pct=_pct_change(strat_market.resilience_mean_ns, base_market.resilience_mean_ns),
        ott_ratio_delta_pct=_pct_change(strat_market.market_ott_ratio, base_market.market_ott_ratio),
        two_sided_delta_pct=_pct_change(strat_market.pct_time_two_sided, base_market.pct_time_two_sided),
    )

    return ComparisonResult(
        strategy_metrics=metrics1,
        strategy_market_metrics=strat_market,
        baseline_metrics=base_market,
        market_impact=impact,
    )
