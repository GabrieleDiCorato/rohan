from typing import Literal, Protocol

from pydantic import BaseModel, Field


class MarketState(BaseModel):
    """
    Represents the state of the market at a point in time.
    """

    timestamp: int = Field(..., description="Nanoseconds from midnight")
    best_bid: float = Field(..., description="Best bid price")
    best_ask: float = Field(..., description="Best ask price")
    last_trade_price: float = Field(..., description="Last trade price")
    inventory: int = Field(..., description="Current position")
    cash: float = Field(..., description="Available cash")


class OrderAction(BaseModel):
    """
    Represents an action to be taken by the agent.
    """

    side: Literal["BUY", "SELL"]
    type: Literal["LIMIT", "MARKET"]
    price: float = Field(..., description="Price for LIMIT orders. Ignore for MARKET.")
    quantity: int = Field(..., description="Quantity to trade")


class StrategicAgent(Protocol):
    """
    Protocol for LLM-generated strategies.
    Must be stateless between calls (state stored in instance attributes).
    """

    def on_market_update(self, market_state: MarketState) -> list[OrderAction]: ...
