"""Adapter that bridges the StrategicAgent protocol to ABIDES TradingAgent.

This module provides `StrategicAgentAdapter`, which wraps a user-provided
`StrategicAgent` implementation and translates ABIDES events into the
simplified protocol callbacks.

All prices and cash values are in **integer cents**, matching ABIDES internals.
No unit conversion is performed by this adapter.
"""

import contextlib
import logging

from abides_core.utils import str_to_ns
from abides_markets.agents import TradingAgent
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import LimitOrder
from abides_markets.orders import Side as AbidesSide

from rohan.simulation.models.strategy_api import (
    CANCEL_ALL,
    AgentConfig,
    MarketState,
    Order,
    OrderAction,
    OrderActionType,
    OrderStatus,
    OrderType,
    Side,
    StrategicAgent,
)

logger = logging.getLogger(__name__)


class StrategicAgentAdapter(TradingAgent):
    """ABIDES TradingAgent that wraps a StrategicAgent protocol implementation.

    This adapter:
    1. Requests market data (with configurable depth) on every wakeup
    2. Translates ABIDES messages to StrategicAgent callbacks
    3. Executes OrderActions returned by the strategy
    4. Fires ``on_order_update`` on fills / cancellations
    5. Tracks cumulative ``filled_quantity`` for partial fills

    Lifecycle mapping:
        kernel_starting()                        → strategy.initialize()
        wakeup()                                 → strategy.on_tick()
        receive_message(QuerySpreadResponseMsg)  → strategy.on_market_data()
        order_executed() / order_cancelled()      → strategy.on_order_update()
        kernel_stopping()                        → strategy.on_simulation_end()

    Performance:
        An incremental ``_open_orders_cache`` avoids rebuilding the full
        open-order list on every tick.  The cache is updated inside the
        ABIDES lifecycle hooks (``order_accepted``, ``order_executed``,
        ``order_cancelled``).
    """

    def __init__(
        self,
        id: int,
        strategy: StrategicAgent,
        symbol: str,
        starting_cash: int,
        wake_up_freq: str = "1S",
        order_book_depth: int = 10,
        log_orders: bool = True,
        random_state=None,
    ):
        super().__init__(
            id=id,
            name=f"StrategicAgent_{id}",
            type="StrategicAgent",
            starting_cash=starting_cash,
            log_orders=log_orders,
            random_state=random_state,
        )
        self.strategy: StrategicAgent = strategy
        self.symbol: str = symbol
        self.wake_up_freq_ns: int = str_to_ns(wake_up_freq)
        self.order_book_depth: int = order_book_depth
        self._initialized: bool = False

        # Incremental cache: order_id → protocol Order
        self._open_orders_cache: dict[int, Order] = {}

        # Cumulative fill tracking: order_id → total shares filled so far
        self._filled_quantities: dict[int, int] = {}

    def get_wake_frequency(self) -> int:
        """Return the wake-up frequency in nanoseconds (required by TradingAgent)."""
        return int(self.wake_up_freq_ns)

    # ── Kernel lifecycle ──────────────────────────────────────────────────

    def kernel_starting(self, start_time: int) -> None:
        """Called when the kernel starts. Initialize strategy and request initial data."""
        super().kernel_starting(start_time)

        config = AgentConfig(
            starting_cash=self.starting_cash,
            symbol=self.symbol,
            latency_ns=0,
            mkt_open_ns=self.mkt_open,  # None until MarketHoursMsg arrives
            mkt_close_ns=self.mkt_close,  # None until MarketHoursMsg arrives
        )
        self.strategy.initialize(config)
        self._initialized = True

        self.set_wakeup(start_time + self.wake_up_freq_ns)

    def kernel_stopping(self) -> None:
        """Called when the kernel stops. Notify strategy and log final state."""
        # Build final snapshot and notify the strategy
        if self._initialized:
            final_state = self._build_market_state(self.current_time)
            self.strategy.on_simulation_end(final_state)

            # Log structured final metrics for post-simulation analysis
            self.logEvent(
                "STRATEGY_FINAL_STATE",
                {
                    "final_cash": self.holdings.get("CASH", 0),
                    "final_inventory": self.holdings.get(self.symbol, 0),
                    "total_fills_tracked": len(self._filled_quantities),
                },
            )

        # Base class logs FINAL_HOLDINGS, FINAL_CASH_POSITION, ENDING_CASH
        super().kernel_stopping()

    # ── Wakeup / message handling ─────────────────────────────────────────

    def wakeup(self, current_time: int) -> None:
        """Called periodically. Fire on_tick, then request a market data update."""
        super().wakeup(current_time)

        # Time-driven callback: strategy can act on periodic schedule
        if self._initialized:
            state = self._build_market_state(current_time)
            actions = self.strategy.on_tick(state)
            self._execute_actions(actions)

        self.get_current_spread(self.symbol, depth=self.order_book_depth)
        self.set_wakeup(current_time + self.wake_up_freq_ns)

    def receive_message(self, current_time: int, sender_id: int, message) -> None:
        """Process incoming messages and route to strategy callbacks."""
        super().receive_message(current_time, sender_id, message)

        if not self._initialized:
            return

        if isinstance(message, QuerySpreadResponseMsg):
            self._handle_market_data(current_time)

    # ── ABIDES order lifecycle hooks ──────────────────────────────────────

    def order_accepted(self, order) -> None:
        """Called by the kernel when the exchange accepts a new order."""
        super().order_accepted(order)
        self._cache_add(order)
        # Notify strategy that order was accepted by the exchange
        if self._initialized:
            proto = self._abides_to_proto(order, status=OrderStatus.ACCEPTED)
            actions = self.strategy.on_order_update(proto)
            self._execute_actions(actions)

    def order_executed(self, order) -> None:
        """Called when an order is (partially) filled.

        Translates into ``strategy.on_order_update()`` with FILLED / PARTIAL
        status so the strategy can track inventory changes.

        Tracks cumulative ``filled_quantity`` across partial fills.
        """
        # The incoming `order.quantity` is the number of shares filled *this*
        # execution.  ABIDES base class then decrements the remaining quantity
        # on the stored order.  We compute the cumulative fill before super().
        this_fill_qty = order.quantity
        fill_price = getattr(order, "fill_price", getattr(order, "limit_price", 0))

        # Update cumulative fill count
        prev_filled = self._filled_quantities.get(order.order_id, 0)
        cumulative_filled = prev_filled + this_fill_qty
        self._filled_quantities[order.order_id] = cumulative_filled

        super().order_executed(order)

        # After super(), order is removed from self.orders if fully filled,
        # or its quantity is decremented.  Check if it was a full fill.
        is_full_fill = order.order_id not in self.orders

        # Build protocol Order for strategy callback
        status = OrderStatus.FILLED if is_full_fill else OrderStatus.PARTIAL
        proto_order = self._abides_to_proto(
            order,
            status=status,
            fill_price=fill_price,
            filled_quantity=cumulative_filled,
        )

        if is_full_fill:
            self._open_orders_cache.pop(order.order_id, None)
            self._filled_quantities.pop(order.order_id, None)
        else:
            # Update cached entry for partial fill
            self._open_orders_cache[order.order_id] = self._abides_to_proto(
                order,
                status=OrderStatus.NEW,
                filled_quantity=cumulative_filled,
            )

        actions = self.strategy.on_order_update(proto_order)
        self._execute_actions(actions)

    def order_cancelled(self, order) -> None:
        """Called when an order is cancelled by the exchange."""
        super().order_cancelled(order)

        filled = self._filled_quantities.pop(order.order_id, 0)
        self._open_orders_cache.pop(order.order_id, None)

        proto_order = self._abides_to_proto(order, status=OrderStatus.CANCELLED, filled_quantity=filled)
        actions = self.strategy.on_order_update(proto_order)
        self._execute_actions(actions)

    def order_modified(self, order) -> None:
        """Called when an order is successfully modified by the exchange."""
        super().order_modified(order)
        # Update cache with new order state
        self._open_orders_cache[order.order_id] = self._abides_to_proto(
            order,
            status=OrderStatus.NEW,
            filled_quantity=self._filled_quantities.get(order.order_id, 0),
        )
        proto = self._abides_to_proto(
            order,
            status=OrderStatus.MODIFIED,
            filled_quantity=self._filled_quantities.get(order.order_id, 0),
        )
        actions = self.strategy.on_order_update(proto)
        self._execute_actions(actions)

    def order_partial_cancelled(self, order) -> None:
        """Called when an order's quantity is partially reduced."""
        super().order_partial_cancelled(order)
        self._open_orders_cache[order.order_id] = self._abides_to_proto(
            order,
            status=OrderStatus.NEW,
            filled_quantity=self._filled_quantities.get(order.order_id, 0),
        )
        proto = self._abides_to_proto(
            order,
            status=OrderStatus.PARTIAL_CANCELLED,
            filled_quantity=self._filled_quantities.get(order.order_id, 0),
        )
        actions = self.strategy.on_order_update(proto)
        self._execute_actions(actions)

    def order_replaced(self, old_order, new_order) -> None:
        """Called when an order is atomically replaced."""
        super().order_replaced(old_order, new_order)
        # Remove old from cache, add new
        self._open_orders_cache.pop(old_order.order_id, None)
        self._filled_quantities.pop(old_order.order_id, None)
        self._cache_add(new_order)
        # Notify strategy about old order being replaced
        proto = self._abides_to_proto(old_order, status=OrderStatus.REPLACED)
        actions = self.strategy.on_order_update(proto)
        self._execute_actions(actions)

    def market_closed(self) -> None:
        """Called when the exchange signals market closure."""
        super().market_closed()
        if self._initialized:
            state = self._build_market_state(self.current_time)
            state = state.model_copy(update={"is_market_closed": True})
            actions = self.strategy.on_tick(state)
            self._execute_actions(actions)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _handle_market_data(self, current_time: int) -> None:
        """Build MarketState and call strategy.on_market_data()."""
        state = self._build_market_state(current_time)
        actions = self.strategy.on_market_data(state)
        self._execute_actions(actions)

    def _build_market_state(self, current_time: int) -> MarketState:
        """Build a MarketState from current TradingAgent state.

        Uses the incremental ``_open_orders_cache`` instead of rebuilding
        the entire list each tick.  Includes full order book depth from
        ``known_bids`` / ``known_asks``.
        """
        # get_known_bid_ask raises KeyError if no data has arrived yet
        bid: int | None = None
        ask: int | None = None
        if self.symbol in self.known_bids:
            bid, _, ask, _ = self.get_known_bid_ask(self.symbol)  # pyright: ignore[reportAssignmentType]

        # Full depth from ABIDES caches (populated by get_current_spread)
        bid_depth: list[tuple[int, int]] = []
        raw_bids = self.known_bids.get(self.symbol)
        if raw_bids:
            bid_depth = list(raw_bids)

        ask_depth: list[tuple[int, int]] = []
        raw_asks = self.known_asks.get(self.symbol)
        if raw_asks:
            ask_depth = list(raw_asks)

        # Populate last_trade from ABIDES TradingAgent tracking dict
        last_trade_price: int | None = None
        if hasattr(self, "last_trade") and isinstance(self.last_trade, dict):
            lt = self.last_trade.get(self.symbol)
            if lt is not None:
                last_trade_price = int(lt) if not isinstance(lt, int) else lt

        # Portfolio valuation (Step 1)
        portfolio_value = self.mark_to_market(self.holdings)
        unrealized_pnl = portfolio_value - self.starting_cash

        # Time remaining (Step 1)
        time_remaining: int | None = None
        if self.mkt_close is not None:
            time_remaining = max(0, self.mkt_close - current_time)

        # Near-touch liquidity (Step 2)
        bid_liq, ask_liq = 0, 0
        if self.symbol in self.known_bids:
            with contextlib.suppress(KeyError, ZeroDivisionError):
                bid_liq, ask_liq = self.get_known_liquidity(self.symbol, within=0.005)

        return MarketState(
            timestamp_ns=current_time,
            best_bid=bid,
            best_ask=ask,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            last_trade=last_trade_price,
            inventory=self.holdings.get(self.symbol, 0),
            cash=self.holdings.get("CASH", 0),
            open_orders=list(self._open_orders_cache.values()),
            portfolio_value=portfolio_value,
            unrealized_pnl=unrealized_pnl,
            time_remaining_ns=time_remaining,
            bid_liquidity=bid_liq,
            ask_liquidity=ask_liq,
        )

    def _execute_actions(self, actions: list[OrderAction]) -> None:
        """Execute a list of OrderActions by dispatching on action_type."""
        for action in actions:
            match action.action_type:
                case OrderActionType.CANCEL:
                    self._handle_cancel(action.cancel_order_id)  # type: ignore[arg-type]
                case OrderActionType.CANCEL_ALL:
                    self._handle_cancel(CANCEL_ALL)
                case OrderActionType.MODIFY:
                    self._handle_modify(action)
                case OrderActionType.PARTIAL_CANCEL:
                    self._handle_partial_cancel(action)
                case OrderActionType.REPLACE:
                    self._handle_replace(action)
                case OrderActionType.PLACE:
                    self._handle_place(action)

    def _handle_cancel(self, cancel_id: int) -> None:
        """Cancel a single order or all open orders (when *cancel_id* == CANCEL_ALL)."""
        if cancel_id == CANCEL_ALL:
            # Snapshot keys — dict may change during iteration as cancel callbacks fire
            for order in list(self.orders.values()):
                if isinstance(order, LimitOrder):
                    self.cancel_order(order)
        else:
            order_to_cancel = self.orders.get(cancel_id)
            if order_to_cancel is not None and isinstance(order_to_cancel, LimitOrder):
                self.cancel_order(order_to_cancel)

    def _handle_place(self, action: OrderAction) -> None:
        """Place a new LIMIT or MARKET order."""
        side = AbidesSide.BID if action.side == Side.BID else AbidesSide.ASK
        if action.order_type == OrderType.MARKET:
            self.place_market_order(self.symbol, action.quantity, side)
        elif action.price is not None:
            self.place_limit_order(
                self.symbol,
                action.quantity,
                side,
                action.price,
                is_hidden=action.is_hidden,
                is_post_only=action.is_post_only,
            )

    def _handle_modify(self, action: OrderAction) -> None:
        """Modify an existing order's price and/or quantity."""
        old = self.orders.get(action.cancel_order_id)  # type: ignore[arg-type]
        if old is None or not isinstance(old, LimitOrder):
            return
        new_order = self.create_limit_order(
            self.symbol,
            quantity=action.new_quantity or old.quantity,
            side=old.side,
            limit_price=action.new_price or old.limit_price,
        )
        self.modify_order(old, new_order)

    def _handle_partial_cancel(self, action: OrderAction) -> None:
        """Reduce an existing order's quantity."""
        order = self.orders.get(action.cancel_order_id)  # type: ignore[arg-type]
        if order is None or not isinstance(order, LimitOrder):
            return
        if action.new_quantity is None or action.new_quantity <= 0:
            return
        self.partial_cancel_order(order, action.new_quantity)

    def _handle_replace(self, action: OrderAction) -> None:
        """Atomically cancel an order and place a new one."""
        old = self.orders.get(action.cancel_order_id)  # type: ignore[arg-type]
        if old is None or not isinstance(old, LimitOrder):
            return
        side = AbidesSide.BID if action.side == Side.BID else AbidesSide.ASK
        new_order = self.create_limit_order(
            self.symbol,
            quantity=action.quantity,
            side=side,
            limit_price=action.price,  # type: ignore[arg-type]
        )
        self.replace_order(old, new_order)

    # ── Conversion helpers ────────────────────────────────────────────────

    def _cache_add(self, abides_order) -> None:
        """Add an ABIDES order to the incremental cache."""
        self._open_orders_cache[abides_order.order_id] = self._abides_to_proto(abides_order, status=OrderStatus.NEW)

    @staticmethod
    def _abides_to_proto(
        abides_order,
        *,
        status: OrderStatus = OrderStatus.NEW,
        fill_price: int | None = None,
        filled_quantity: int = 0,
    ) -> Order:
        """Convert an ABIDES order object to a protocol ``Order``."""
        return Order(
            order_id=abides_order.order_id,
            symbol=abides_order.symbol,
            side=Side.BID if abides_order.side == AbidesSide.BID else Side.ASK,
            quantity=max(abides_order.quantity, 1),
            price=getattr(abides_order, "limit_price", 0),
            order_type=OrderType.LIMIT,
            status=status,
            filled_quantity=filled_quantity,
            fill_price=fill_price,
        )
