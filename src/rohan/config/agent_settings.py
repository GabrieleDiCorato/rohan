"""Configures all the non-learning agents that will run in the simulation."""

from enum import Enum

from pydantic import BaseModel, Field


class AgentType(str, Enum):
    """Specifies the type of agent in the simulation."""

    EXCHANGE = "ExchangeAgent"
    NOISE = "NoiseAgent"
    VALUE = "ValueAgent"
    ADAPTIVE_MARKET_MAKER = "AdaptivePOVMarketMakerAgent"
    MOMENTUM = "MomentumAgent"


class BaseAgentSettings(BaseModel):
    """Base settings for all agent types."""

    num_agents: int = Field(default=1, description="Number of agents in the simulation")
    type: AgentType = Field(..., description="Type of agent")


class ExchangeAgentSettings(BaseAgentSettings):
    """Settings for the Exchange Agent."""

    type: AgentType = AgentType.EXCHANGE
    book_logging: bool = Field(default=True, description="Whether to log the order book")
    book_log_depth: int = Field(default=10, description="Depth of the order book to log")
    stream_history_length: int = Field(default=500, description="Length of the stream history to maintain")
    exchange_log_orders: bool | None = Field(default=None, description="Whether the exchange should log all orders")

    pipeline_delay_ns: int = Field(default=0, description="Pipeline delay in nanoseconds")
    computation_delay_ns: int = Field(default=0, description="Computation delay in nanoseconds")


class NoiseAgentSettings(BaseAgentSettings):
    """Settings for Noise Agents."""

    type: AgentType = AgentType.NOISE


class ValueAgentSettings(BaseAgentSettings):
    """Settings for Value Agents."""

    type: AgentType = AgentType.VALUE
    r_bar: int = Field(default=100_000, description="True mean fundamental value")
    kappa: float = Field(default=1.67e-15, description="Value agents appraisal of mean-reversion")
    lambda_a: float = Field(default=5.7e-12, description="ValueAgent arrival rate")


class AdaptiveMarketMakerSettings(BaseAgentSettings):
    """Settings for Adaptive Market Maker Agents."""

    type: AgentType = AgentType.ADAPTIVE_MARKET_MAKER
    window_size: str = Field(default="adaptive", description="Window size for market maker")
    pov: float = Field(default=0.025, description="Percentage of volume")
    num_ticks: int = Field(default=10, description="Number of ticks")
    wake_up_freq: str = Field(default="60s", description="Wake up frequency")
    poisson_arrival: bool = Field(default=True, description="Whether to use Poisson arrival")
    min_order_size: int = Field(default=1, description="Minimum order size")
    skew_beta: float = Field(default=0, description="Skew beta parameter")
    price_skew: int = Field(default=4, description="Price skew parameter")
    level_spacing: int = Field(default=5, description="Level spacing")
    spread_alpha: float = Field(default=0.75, description="Spread alpha parameter")
    backstop_quantity: int = Field(default=0, description="Backstop quantity")
    cancel_limit_delay_ns: int = Field(default=50, description="Cancel limit delay in nanoseconds")


class MomentumAgentSettings(BaseAgentSettings):
    """Settings for Momentum Agents."""

    type: AgentType = AgentType.MOMENTUM
    min_size: int = Field(default=100, description="Minimum order size")
    max_size: int = Field(default=500, description="Maximum order size")
    wake_up_freq: str = Field(default="37s", description="Wake up frequency")
    poisson_arrival: bool = Field(default=True, description="Whether to use Poisson arrival")


class OracleSettings(BaseModel):
    """Configuration for the oracle."""

    kappa: float = Field(default=1.67e-16, description="Mean-reversion of fundamental time series")
    sigma_s: float = Field(default=0, description="Sigma s parameter")
    fund_vol: float = Field(default=5e-5, description="Volatility of fundamental time series (std)")
    megashock_lambda_a: float = Field(default=2.77778e-18, description="Megashock lambda a")
    megashock_mean: int = Field(default=1000, description="Megashock mean")
    megashock_var: int = Field(default=50_000, description="Megashock variance")


class AgentSettings(BaseModel):
    """Configuration for agents in the simulation."""

    exchange_agent: ExchangeAgentSettings = Field(default_factory=ExchangeAgentSettings, description="Settings for the exchange agent")
    noise_agent: NoiseAgentSettings = Field(default_factory=NoiseAgentSettings, description="Settings for the noise agents")
    value_agent: ValueAgentSettings = Field(default_factory=ValueAgentSettings, description="Settings for the value agents")
    adaptive_market_maker: AdaptiveMarketMakerSettings = Field(default_factory=AdaptiveMarketMakerSettings, description="Settings for the adaptive market maker agents")
    momentum_agent: MomentumAgentSettings = Field(default_factory=MomentumAgentSettings, description="Settings for the momentum agents")
    oracle: OracleSettings = Field(default_factory=OracleSettings, description="Settings for the oracle")
