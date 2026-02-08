"""Strategic Agent API — the contract between LLM-generated strategies and the simulator.

This is the **ONLY** interface that generated strategy code interacts with.
All other simulation internals (ABIDES kernel, order book, etc.) are hidden.

Units & conventions (matching ABIDES):
    * **Prices:** ``int``, in cents (e.g. ``18550`` = $185.50).
    * **Quantities:** ``int``, in shares.
    * **Cash:** ``int``, in cents.
    * **Timestamps:** ``int``, nanoseconds since epoch.
"""

from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field, model_validator


class Side(str, Enum):
    BID = "BID"
    ASK = "ASK"


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


class OrderStatus(str, Enum):
    NEW = "NEW"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class Order(BaseModel):
    """A live or historical order visible to the strategy."""

    order_id: int
    symbol: str
    side: Side
    quantity: int = Field(ge=1, description="Number of shares")
    price: int = Field(description="Limit price in cents (0 for market orders)")
    order_type: OrderType
    status: OrderStatus
    filled_quantity: int = Field(default=0, ge=0, description="Shares filled so far")


class MarketState(BaseModel):
    """Snapshot of the market visible to the strategy at a given instant.

    Passed to ``StrategicAgent.on_market_data`` on every L1 update.
    """

    timestamp_ns: int = Field(description="Nanoseconds since epoch")
    best_bid: int | None = Field(default=None, description="Best bid price in cents")
    best_ask: int | None = Field(default=None, description="Best ask price in cents")
    last_trade: int | None = Field(default=None, description="Last trade price in cents")
    inventory: int = Field(description="Signed position in shares")
    cash: int = Field(description="Available cash in cents")
    open_orders: list[Order]


class OrderAction(BaseModel):
    """An order instruction returned by the strategy.

    For ``LIMIT`` orders, ``price`` is mandatory.
    For ``MARKET`` orders, ``price`` must be ``None``.
    """

    side: Side
    quantity: int = Field(ge=1, description="Number of shares")
    price: int | None = Field(default=None, description="Limit price in cents (required for LIMIT)")
    order_type: OrderType
    cancel_order_id: int | None = Field(default=None, description="If set, cancel this order instead of placing a new one")

    @model_validator(mode="after")
    def _validate_price_for_order_type(self) -> "OrderAction":
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("price is required for LIMIT orders")
        if self.order_type == OrderType.MARKET and self.price is not None:
            raise ValueError("price must be None for MARKET orders")
        return self


class AgentConfig(BaseModel):
    """Configuration passed to ``StrategicAgent.initialize`` at simulation start."""

    starting_cash: int = Field(description="Initial cash in cents")
    symbol: str = Field(description="Ticker symbol to trade")
    latency_ns: int = Field(description="Simulated network latency in nanoseconds")


class StrategicAgent(Protocol):
    """Protocol for LLM-generated trading strategies."""

    def initialize(self, config: AgentConfig) -> None:
        """Called once at the start of the simulation."""
        ...

    def on_market_data(self, state: MarketState) -> list[OrderAction]:
        """Called when new market data (L1/L2) is received."""
        ...

    def on_order_update(self, update: Order) -> list[OrderAction]:
        """Called when an order status changes."""
        ...
