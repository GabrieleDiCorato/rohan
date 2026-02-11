"""Adapter that bridges the StrategicAgent protocol to ABIDES TradingAgent.

This module provides `StrategicAgentAdapter`, which wraps a user-provided
`StrategicAgent` implementation and translates ABIDES events into the
simplified protocol callbacks.

All prices and cash values are in **integer cents**, matching ABIDES internals.
No unit conversion is performed by this adapter.
"""

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
    OrderStatus,
    OrderType,
    Side,
    StrategicAgent,
)

logger = logging.getLogger(__name__)


class StrategicAgentAdapter(TradingAgent):
    """ABIDES TradingAgent that wraps a StrategicAgent protocol implementation.

    This adapter:
    1. Subscribes to L1 market data during kernel_starting
    2. Translates ABIDES messages to StrategicAgent callbacks
    3. Executes OrderActions returned by the strategy
    4. Fires ``on_order_update`` on fills / cancellations

    Lifecycle mapping:
        kernel_starting()                        → strategy.initialize()
        receive_message(QuerySpreadResponseMsg)  → strategy.on_market_data()
        order_executed() / order_cancelled()      → strategy.on_order_update()

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
        self._initialized: bool = False

        # Incremental cache: order_id → protocol Order
        self._open_orders_cache: dict[int, Order] = {}

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
        )
        self.strategy.initialize(config)
        self._initialized = True

        self.set_wakeup(start_time + self.wake_up_freq_ns)

    # ── Wakeup / message handling ─────────────────────────────────────────

    def wakeup(self, current_time: int) -> None:
        """Called periodically. Request market data update."""
        super().wakeup(current_time)
        self.get_current_spread(self.symbol)
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

    def order_executed(self, order) -> None:
        """Called when an order is (partially) filled.

        Translates into ``strategy.on_order_update()`` with FILLED / PARTIAL
        status so the strategy can track inventory changes.
        """
        # Determine fill info *before* super() removes the order from self.orders
        is_full_fill = order.quantity == 0  # ABIDES decrements quantity on partial fills
        fill_price = getattr(order, "fill_price", getattr(order, "limit_price", 0))

        super().order_executed(order)

        # Build protocol Order for strategy callback
        status = OrderStatus.FILLED if is_full_fill else OrderStatus.PARTIAL
        proto_order = self._abides_to_proto(order, status=status, fill_price=fill_price)

        if is_full_fill:
            self._open_orders_cache.pop(order.order_id, None)
        else:
            # Update cached entry for partial fill
            self._open_orders_cache[order.order_id] = self._abides_to_proto(order, status=OrderStatus.NEW)

        actions = self.strategy.on_order_update(proto_order)
        self._execute_actions(actions)

    def order_cancelled(self, order) -> None:
        """Called when an order is cancelled by the exchange."""
        super().order_cancelled(order)

        self._open_orders_cache.pop(order.order_id, None)

        proto_order = self._abides_to_proto(order, status=OrderStatus.CANCELLED)
        actions = self.strategy.on_order_update(proto_order)
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
        the entire list each tick.
        """
        bid, _, ask, _ = self.get_known_bid_ask(self.symbol)

        # Populate last_trade from ABIDES TradingAgent tracking dict
        last_trade_price: int | None = None
        if hasattr(self, "last_trade") and isinstance(self.last_trade, dict):
            lt = self.last_trade.get(self.symbol)
            if lt is not None:
                last_trade_price = int(lt) if not isinstance(lt, int) else lt

        return MarketState(
            timestamp_ns=current_time,
            best_bid=bid,
            best_ask=ask,
            last_trade=last_trade_price,
            inventory=self.holdings.get(self.symbol, 0),
            cash=self.holdings.get("CASH", 0),
            open_orders=list(self._open_orders_cache.values()),
        )

    def _execute_actions(self, actions: list[OrderAction]) -> None:
        """Execute a list of OrderActions by placing / cancelling orders."""
        for action in actions:
            if action.cancel_order_id is not None:
                self._handle_cancel(action.cancel_order_id)
            elif action.order_type == OrderType.MARKET:
                side = AbidesSide.BID if action.side == Side.BID else AbidesSide.ASK
                self.place_market_order(self.symbol, action.quantity, side)
            else:
                if action.price is None:
                    continue
                side = AbidesSide.BID if action.side == Side.BID else AbidesSide.ASK
                self.place_limit_order(self.symbol, action.quantity, side, action.price)

    def _handle_cancel(self, cancel_id: int) -> None:
        """Cancel a single order or all open orders (when *cancel_id* == CANCEL_ALL)."""
        if cancel_id == CANCEL_ALL:
            # Snapshot keys — dict may change during iteration as cancel callbacks fire
            for order in list(self.orders.values()):
                if isinstance(order, LimitOrder):
                    self.cancel_order(order)
        else:
            order_to_cancel = self.orders.get(cancel_id)
            if order_to_cancel is not None:
                self.cancel_order(order_to_cancel)  # type: ignore[arg-type]

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
            filled_quantity=0,
            fill_price=fill_price,
        )
