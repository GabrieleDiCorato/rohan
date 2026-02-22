from typing import Any

from rohan.config import SimulationSettings
from rohan.framework.analysis_service import AnalysisService
from rohan.simulation import (
    ComparisonResult,
    MarketImpact,
    MarketMetrics,
)
from rohan.simulation.models import SimulationMetrics
from rohan.simulation.simulation_service import SimulationService
from rohan.simulation.strategy_validator import (
    execute_strategy_safely,
)


def _to_market_metrics(sim: "SimulationMetrics") -> MarketMetrics:
    """Convert a SimulationMetrics to MarketMetrics (same shape, 1-to-1)."""
    return MarketMetrics(
        volatility=sim.volatility,
        mean_spread=sim.mean_spread,
        effective_spread=sim.effective_spread,
        avg_bid_liquidity=sim.avg_bid_liquidity,
        avg_ask_liquidity=sim.avg_ask_liquidity,
        traded_volume=sim.traded_volume,
        lob_imbalance_mean=sim.lob_imbalance_mean,
        lob_imbalance_std=sim.lob_imbalance_std,
        vpin=sim.vpin,
        resilience_mean_ns=sim.resilience_mean_ns,
        market_ott_ratio=sim.market_ott_ratio,
    )


def _pct_change(a: float | None, b: float | None) -> float | None:
    """Percentage change (a − b) / b.  Returns None when undefined."""
    if a is None or b is None:
        return None
    if b == 0:
        return 0.0 if a == 0 else float("inf")
    return (a - b) / b


def run_with_baseline(
    strategy_code: str,
    settings: SimulationSettings,
    baseline_override: dict[str, Any] | None = None,
    timeout_seconds: int = 300,
) -> ComparisonResult:
    """Run strategy + baseline simulations and return a typed ComparisonResult."""

    # 1. Run Strategy
    res1 = execute_strategy_safely(strategy_code, settings, timeout_seconds)
    if res1.error:
        raise RuntimeError(f"Strategy run failed: {res1.error}")
    if not res1.result:
        raise RuntimeError("Strategy run returned no result")

    metrics1 = AnalysisService.compute_agent_metrics(res1.result, 1)
    metrics1_sim = AnalysisService.compute_metrics(res1.result)

    # 2. Run Baseline
    baseline_settings = settings.model_copy(deep=True)
    if baseline_override:
        for k, v in baseline_override.items():
            if hasattr(baseline_settings, k):
                setattr(baseline_settings, k, v)

    service = SimulationService()
    res2 = service.run_simulation(baseline_settings, strategy=None)
    if res2.error:
        raise RuntimeError(f"Baseline run failed: {res2.error}")
    if not res2.result:
        raise RuntimeError("Baseline run returned no result")

    metrics2_sim = AnalysisService.compute_metrics(res2.result)

    # Convert to MarketMetrics
    strat_market = _to_market_metrics(metrics1_sim)
    base_market = _to_market_metrics(metrics2_sim)

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
    )

    return ComparisonResult(
        strategy_metrics=metrics1,
        strategy_market_metrics=strat_market,
        baseline_metrics=base_market,
        market_impact=impact,
    )
