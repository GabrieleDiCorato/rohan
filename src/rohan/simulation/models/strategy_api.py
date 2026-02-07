from enum import Enum
from typing import Protocol

from pydantic import BaseModel


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
    order_id: int
    symbol: str
    side: Side
    quantity: int
    price: int  # Price in cents
    order_type: OrderType
    status: OrderStatus
    filled_quantity: int = 0


class MarketState(BaseModel):
    timestamp_ns: int
    best_bid: int | None
    best_ask: int | None
    last_trade: int | None
    inventory: int  # Signed position
    cash: int  # Cash in cents
    open_orders: list[Order]


class OrderAction(BaseModel):
    side: Side
    quantity: int
    price: int | None  # Required for LIMIT
    order_type: OrderType
    cancel_order_id: int | None = None  # If set, this is a cancel request


class AgentConfig(BaseModel):
    starting_cash: int
    symbol: str
    latency_ns: int


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
