from typing import Any

from pydantic import BaseModel, Field


class SimulationMetrics(BaseModel):
    """
    Summary metrics for a simulation run.
    """

    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    volatility: float
    traded_volume: int
    custom_metrics: dict[str, Any] = Field(default_factory=dict)
