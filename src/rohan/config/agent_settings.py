"""Configures all the non-learning agents that will run in the simulation."""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


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
    exchange_log_orders: bool = Field(default=False, description="Whether the exchange should log all orders")

    pipeline_delay_ns: int = Field(default=0, description="Pipeline delay in nanoseconds")
    computation_delay_ns: int = Field(default=0, description="Computation delay in nanoseconds")


class NoiseAgentSettings(BaseAgentSettings):
    """Settings for Noise Agents."""

    num_agents: int = 1000
    type: AgentType = AgentType.NOISE


class ValueAgentSettings(BaseAgentSettings):
    """Settings for Value Agents."""

    num_agents: int = 102
    type: AgentType = AgentType.VALUE
    r_bar: int = Field(default=100_000, description="True mean fundamental value in cents (e.g., 100_000 = $1000)")
    kappa: float = Field(default=1.67e-15, description="Value agents appraisal of mean-reversion")
    lambda_a: float = Field(default=5.7e-12, description="ValueAgent arrival rate")

    @field_validator("r_bar")
    @classmethod
    def validate_r_bar(cls, v: int) -> int:
        """Validate r_bar to prevent int32 overflow in ABIDES ValueAgent.placeOrder().

        The ValueAgent uses r_bar to calculate order size adjustments via random_state.randint(),
        which requires values within int32 range (< 2^31 - 1 = 2,147,483,647).
        A reasonable upper limit is 10,000,000 cents ($100,000 stock price).
        """
        if v <= 0:
            raise ValueError(f"r_bar must be positive, got {v}")
        if v > 10_000_000:
            raise ValueError(f"r_bar too large ({v:,}). Must be ≤ 10,000,000 cents ($100,000) to prevent int32 overflow in ABIDES. " f"For reference: r_bar=100,000 represents a $1,000 stock.")
        return v

    @field_validator("lambda_a")
    @classmethod
    def validate_lambda_a(cls, v: float) -> float:
        """Validate lambda_a arrival rate to ensure reasonable values."""
        if v <= 0:
            raise ValueError(f"lambda_a must be positive, got {v}")
        if v > 1e-3:
            raise ValueError(f"lambda_a too large ({v}). Recommended range: 1e-6 to 1e-4. " f"Very large values can cause numerical issues.")
        return v

    @field_validator("kappa")
    @classmethod
    def validate_kappa(cls, v: float) -> float:
        """Validate kappa mean-reversion parameter."""
        if v < 0:
            raise ValueError(f"kappa must be non-negative, got {v}")
        if v > 1e-10:
            raise ValueError(f"kappa too large ({v}). Recommended range: 1e-16 to 1e-12 for realistic mean-reversion.")
        return v


class AdaptiveMarketMakerSettings(BaseAgentSettings):
    """Settings for Adaptive Market Maker Agents."""

    num_agents: int = Field(default=2, description="Number of agents in the simulation")
    type: AgentType = AgentType.ADAPTIVE_MARKET_MAKER
    window_size: int | str = Field(default="adaptive", description=" Size in ticks (cents). If equal to string 'adaptive', ladder starts at best bid and ask")
    pov: float = Field(default=0.025, description="Percentage of volume")
    num_ticks: int = Field(default=10, description="Number of ticks")
    wake_up_freq: str = Field(default="60s", description="Wake up frequency")
    poisson_arrival: bool = Field(default=True, description="Whether to use Poisson arrival")
    min_order_size: int = Field(default=1, description="Minimum order size")
    skew_beta: int = Field(default=0, description="Skew beta parameter")
    price_skew: int = Field(default=4, description="Price skew parameter")
    level_spacing: int = Field(default=5, description="Level spacing")
    spread_alpha: float = Field(default=0.75, description="Spread alpha parameter")
    backstop_quantity: int = Field(default=0, description="Backstop quantity")
    cancel_limit_delay_ns: int = Field(default=50, description="Cancel limit delay in nanoseconds")


class MomentumAgentSettings(BaseAgentSettings):
    """Settings for Momentum Agents."""

    num_agents: int = 12
    type: AgentType = AgentType.MOMENTUM
    min_size: int = Field(default=1, description="Minimum order size")
    max_size: int = Field(default=10, description="Maximum order size")
    wake_up_freq: str = Field(default="37s", description="Wake up frequency")
    poisson_arrival: bool = Field(default=True, description="Whether to use Poisson arrival")

    @field_validator("max_size")
    @classmethod
    def validate_max_size(cls, v: int) -> int:
        """Validate maximum order size."""
        if v <= 0:
            raise ValueError(f"max_size must be positive, got {v}")
        if v > 1000:
            raise ValueError(f"max_size too large ({v}). Recommended maximum: 1000 shares to prevent market impact issues.")
        return v

    @field_validator("min_size")
    @classmethod
    def validate_min_size(cls, v: int) -> int:
        """Validate minimum order size."""
        if v <= 0:
            raise ValueError(f"min_size must be positive, got {v}")
        return v

    def model_post_init(self, __context) -> None:
        """Validate that min_size <= max_size after both fields are set."""
        super().model_post_init(__context)
        if self.min_size > self.max_size:
            raise ValueError(f"min_size ({self.min_size}) must be ≤ max_size ({self.max_size})")


class OracleSettings(BaseModel):
    """Configuration for the oracle."""

    kappa: float = Field(default=1.67e-16, description="Mean-reversion of fundamental time series")
    sigma_s: float = Field(default=0, description="Sigma s parameter")
    fund_vol: float = Field(default=5e-5, description="Volatility of fundamental time series (std)")
    megashock_lambda_a: float = Field(default=2.77778e-18, description="Megashock lambda a (arrival rate)")
    megashock_mean: int = Field(default=1000, description="Megashock mean in cents")
    megashock_var: int = Field(default=50_000, description="Megashock variance")

    @field_validator("megashock_mean")
    @classmethod
    def validate_megashock_mean(cls, v: int) -> int:
        """Validate megashock_mean to prevent extreme values."""
        if abs(v) > 100_000:
            raise ValueError(f"megashock_mean too large ({v:,}). Must be within ±100,000 cents (±$1,000) for realistic shocks.")
        return v

    @field_validator("megashock_var")
    @classmethod
    def validate_megashock_var(cls, v: int) -> int:
        """Validate megashock_var to prevent extreme values."""
        if v < 0:
            raise ValueError(f"megashock_var must be non-negative, got {v}")
        if v > 10_000_000:
            raise ValueError(f"megashock_var too large ({v:,}). Must be ≤ 10,000,000 for realistic variance.")
        return v

    @field_validator("fund_vol")
    @classmethod
    def validate_fund_vol(cls, v: float) -> float:
        """Validate fundamental volatility."""
        if v < 0:
            raise ValueError(f"fund_vol must be non-negative, got {v}")
        if v > 1e-2:
            raise ValueError(f"fund_vol too large ({v}). Recommended range: 1e-5 to 1e-3 for realistic volatility.")
        return v


class AgentSettings(BaseModel):
    """Configuration for agents in the simulation."""

    exchange: ExchangeAgentSettings = Field(default_factory=ExchangeAgentSettings, description="Settings for the exchange agent")
    noise: NoiseAgentSettings = Field(default_factory=NoiseAgentSettings, description="Settings for the noise agents")
    value: ValueAgentSettings = Field(default_factory=ValueAgentSettings, description="Settings for the value agents")
    adaptive_market_maker: AdaptiveMarketMakerSettings = Field(default_factory=AdaptiveMarketMakerSettings, description="Settings for the adaptive market maker agents")
    momentum: MomentumAgentSettings = Field(default_factory=MomentumAgentSettings, description="Settings for the momentum agents")
    oracle: OracleSettings = Field(default_factory=OracleSettings, description="Settings for the oracle")
