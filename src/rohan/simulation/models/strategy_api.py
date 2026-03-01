"""Strategic Agent API — the contract between LLM-generated strategies and the simulator.

This is the **ONLY** interface that generated strategy code interacts with.
All other simulation internals (ABIDES kernel, order book, etc.) are hidden.

Units & conventions (matching ABIDES):
    * **Prices:** ``int``, in cents (e.g. ``18550`` = $185.50).
    * **Quantities:** ``int``, in shares.
    * **Cash:** ``int``, in cents.
    * **Timestamps:** ``int``, nanoseconds since epoch.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field, computed_field, model_validator


class Side(str, Enum):
    BID = "BID"
    ASK = "ASK"


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


class OrderStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    NEW = "NEW"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    MODIFIED = "MODIFIED"
    PARTIAL_CANCELLED = "PARTIAL_CANCELLED"
    REPLACED = "REPLACED"


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
    fill_price: int | None = Field(default=None, description="Price at which the last fill occurred (cents)")


class MarketState(BaseModel):
    """Snapshot of the market visible to the strategy at a given instant.

    Passed to ``StrategicAgent.on_market_data`` on every market data update
    and to ``StrategicAgent.on_tick`` on every periodic wakeup.

    ``best_bid`` / ``best_ask`` are the L1 top-of-book.  ``bid_depth`` /
    ``ask_depth`` expose multi-level order book data (best first, as
    ``(price_cents, quantity)`` tuples).  Strategies can use either.
    """

    timestamp_ns: int = Field(description="Nanoseconds since epoch")
    best_bid: int | None = Field(default=None, description="Best bid price in cents")
    best_ask: int | None = Field(default=None, description="Best ask price in cents")
    bid_depth: list[tuple[int, int]] = Field(
        default_factory=list,
        description="Order book bid levels: [(price_cents, qty), ...], best first",
    )
    ask_depth: list[tuple[int, int]] = Field(
        default_factory=list,
        description="Order book ask levels: [(price_cents, qty), ...], best first",
    )
    last_trade: int | None = Field(default=None, description="Last trade price in cents")
    inventory: int = Field(description="Signed position in shares")
    cash: int = Field(description="Available cash in cents")
    open_orders: list[Order]

    # ── Step 1: Situational Awareness ────────────────────────────────────

    portfolio_value: int = Field(
        default=0,
        description="Mark-to-market portfolio value (cash + positions) in cents",
    )
    unrealized_pnl: int = Field(
        default=0,
        description="Unrealized PnL = portfolio_value - starting_cash, in cents",
    )
    time_remaining_ns: int | None = Field(
        default=None,
        description="Nanoseconds until market close. None if market hours unknown.",
    )
    is_market_closed: bool = Field(
        default=False,
        description="True when the market has closed. No further orders can be placed.",
    )

    # ── Step 2: Liquidity ────────────────────────────────────────────────

    bid_liquidity: int = Field(
        default=0,
        description="Total bid-side volume within 0.5%% of best bid (in shares)",
    )
    ask_liquidity: int = Field(
        default=0,
        description="Total ask-side volume within 0.5%% of best ask (in shares)",
    )

    # ── Computed convenience fields ──────────────────────────────────────

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mid_price(self) -> int | None:
        """Midpoint price in cents, or None if either side is missing."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) // 2
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def spread(self) -> int | None:
        """Bid-ask spread in cents, or None if either side is missing."""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None


# ---------------------------------------------------------------------------
# Cancellation sentinel: cancel_order_id = CANCEL_ALL cancels every open order.
# ---------------------------------------------------------------------------
CANCEL_ALL: int = -1


class OrderActionType(str, Enum):
    """Discriminator for order action intent."""

    PLACE = "PLACE"
    CANCEL = "CANCEL"
    CANCEL_ALL = "CANCEL_ALL"
    MODIFY = "MODIFY"
    PARTIAL_CANCEL = "PARTIAL_CANCEL"
    REPLACE = "REPLACE"


class OrderAction(BaseModel):
    """An order instruction returned by the strategy.

    For ``LIMIT`` orders, ``price`` is mandatory.
    For ``MARKET`` orders, ``price`` must be ``None``.

    To cancel a specific order::

        OrderAction.cancel(order_id=123)

    To cancel **all** open orders::

        OrderAction.cancel_all()

    To modify an existing order::

        OrderAction.modify(order_id=123, new_price=10050)

    To partially cancel (reduce quantity)::

        OrderAction.partial_cancel(order_id=123, reduce_by=50)

    To atomically replace an order::

        OrderAction.replace(order_id=123, side=Side.BID, quantity=100, price=10050)
    """

    action_type: OrderActionType = Field(
        default=OrderActionType.PLACE,
        description="What this action does. Default is PLACE (new order).",
    )
    side: Side = Field(default=Side.BID, description="Order side (ignored for cancellations)")
    quantity: int = Field(default=1, ge=1, description="Number of shares (ignored for cancellations)")
    price: int | None = Field(default=None, description="Limit price in cents (required for LIMIT)")
    order_type: OrderType = Field(default=OrderType.LIMIT, description="Order type (ignored for cancellations)")
    cancel_order_id: int | None = Field(
        default=None,
        description="Target order ID for CANCEL/MODIFY/PARTIAL_CANCEL/REPLACE. Use -1 to cancel all.",
    )

    # ── Step 2: Order qualifiers ─────────────────────────────────────────

    is_hidden: bool = Field(
        default=False,
        description="Iceberg order: quantity is hidden from the book (LIMIT only)",
    )
    is_post_only: bool = Field(
        default=False,
        description="Reject if order would immediately execute (LIMIT only)",
    )

    # ── Step 3: Modification fields ──────────────────────────────────────

    new_price: int | None = Field(
        default=None,
        description="New limit price for MODIFY/REPLACE actions (cents)",
    )
    new_quantity: int | None = Field(
        default=None,
        description="New quantity for MODIFY/REPLACE, or reduction amount for PARTIAL_CANCEL",
    )

    @model_validator(mode="after")
    def _validate_price_for_order_type(self) -> OrderAction:
        # Skip price validation for non-PLACE actions
        if self.action_type != OrderActionType.PLACE:
            return self
        # Legacy: skip when cancel_order_id is set (backward compat)
        if self.cancel_order_id is not None:
            return self
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("price is required for LIMIT orders")
        if self.order_type == OrderType.MARKET and self.price is not None:
            raise ValueError("price must be None for MARKET orders")
        return self

    @model_validator(mode="after")
    def _validate_hidden_post_only(self) -> OrderAction:
        """Warn when hidden/post-only flags are used with MARKET orders."""
        if self.action_type != OrderActionType.PLACE:
            return self
        if self.cancel_order_id is not None:
            return self
        if self.order_type == OrderType.MARKET and (self.is_hidden or self.is_post_only):
            raise ValueError("is_hidden and is_post_only are only valid for LIMIT orders")
        return self

    @model_validator(mode="after")
    def _infer_action_type(self) -> OrderAction:
        """Auto-infer action_type from cancel_order_id for backward compatibility."""
        if self.action_type == OrderActionType.PLACE and self.cancel_order_id is not None:
            if self.cancel_order_id == CANCEL_ALL:
                self.action_type = OrderActionType.CANCEL_ALL
            else:
                self.action_type = OrderActionType.CANCEL
        return self

    # ── Convenience factories ────────────────────────────────────────────

    @classmethod
    def cancel(cls, order_id: int) -> OrderAction:
        """Create a cancellation action for the given *order_id*."""
        return cls(action_type=OrderActionType.CANCEL, cancel_order_id=order_id)

    @classmethod
    def cancel_all(cls) -> OrderAction:
        """Create a cancellation action that cancels **every** open order."""
        return cls(action_type=OrderActionType.CANCEL_ALL, cancel_order_id=CANCEL_ALL)

    @classmethod
    def modify(
        cls,
        order_id: int,
        *,
        new_price: int | None = None,
        new_quantity: int | None = None,
    ) -> OrderAction:
        """Modify an existing order's price and/or quantity."""
        if new_price is None and new_quantity is None:
            raise ValueError("At least one of new_price or new_quantity must be specified")
        return cls(
            action_type=OrderActionType.MODIFY,
            cancel_order_id=order_id,
            new_price=new_price,
            new_quantity=new_quantity,
        )

    @classmethod
    def partial_cancel(cls, order_id: int, reduce_by: int) -> OrderAction:
        """Reduce an existing order's quantity by *reduce_by* shares."""
        return cls(
            action_type=OrderActionType.PARTIAL_CANCEL,
            cancel_order_id=order_id,
            new_quantity=reduce_by,
        )

    @classmethod
    def replace(
        cls,
        order_id: int,
        *,
        side: Side,
        quantity: int,
        price: int,
    ) -> OrderAction:
        """Atomically cancel an order and place a new one."""
        return cls(
            action_type=OrderActionType.REPLACE,
            cancel_order_id=order_id,
            side=side,
            quantity=quantity,
            price=price,
            order_type=OrderType.LIMIT,
        )


class AgentConfig(BaseModel):
    """Configuration passed to ``StrategicAgent.initialize`` at simulation start."""

    starting_cash: int = Field(description="Initial cash in cents")
    symbol: str = Field(description="Ticker symbol to trade")
    latency_ns: int = Field(description="Simulated network latency in nanoseconds")
    mkt_open_ns: int | None = Field(
        default=None,
        description="Market open timestamp in nanoseconds (None until known)",
    )
    mkt_close_ns: int | None = Field(
        default=None,
        description="Market close timestamp in nanoseconds (None until known)",
    )


class StrategicAgent(Protocol):
    """Protocol for LLM-generated trading strategies."""

    def initialize(self, config: AgentConfig) -> None:
        """Called once at the start of the simulation."""
        ...

    def on_tick(self, state: MarketState) -> list[OrderAction]:
        """Called on every periodic wakeup (time-driven).

        Use this for rebalancing or other time-scheduled logic,
        independent of market data arrivals.
        """
        ...

    def on_market_data(self, state: MarketState) -> list[OrderAction]:
        """Called when new market data (L1/L2) is received."""
        ...

    def on_order_update(self, update: Order) -> list[OrderAction]:
        """Called when an order is filled, partially filled, or cancelled."""
        ...

    def on_simulation_end(self, final_state: MarketState) -> None:
        """Called once at the end of the simulation for cleanup and logging.

        The market is closed at this point — no orders can be placed.
        Use this to compute final metrics or log strategy state.
        """
        ...
