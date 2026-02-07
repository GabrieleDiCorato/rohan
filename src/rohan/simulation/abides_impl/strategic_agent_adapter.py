"""Adapter that bridges the StrategicAgent protocol to ABIDES TradingAgent.

This module provides `StrategicAgentAdapter`, which wraps a user-provided
`StrategicAgent` implementation and translates ABIDES events into the
simplified protocol callbacks.
"""

from abides_core.utils import str_to_ns
from abides_markets.agents import TradingAgent
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import Side as AbidesSide

from rohan.simulation.models.strategy_api import (
    AgentConfig,
    MarketState,
    Order,
    OrderAction,
    OrderStatus,
    OrderType,
    Side,
    StrategicAgent,
)


class StrategicAgentAdapter(TradingAgent):
    """ABIDES TradingAgent that wraps a StrategicAgent protocol implementation.

    This adapter:
    1. Subscribes to L1 market data during kernel_starting
    2. Translates ABIDES messages to StrategicAgent callbacks
    3. Executes OrderActions returned by the strategy

    Lifecycle mapping:
        kernel_starting() → strategy.initialize()
        receive_message(QuerySpreadResponseMsg) → strategy.on_market_data()
        (Order fills handled via next market data update with updated holdings)
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
        self.strategy = strategy
        self.symbol = symbol
        self.wake_up_freq_ns = str_to_ns(wake_up_freq)
        self._initialized = False
        self._previous_holdings: dict[str, int] = {}

    def get_wake_frequency(self) -> int:
        """Return the wake-up frequency in nanoseconds (required by TradingAgent)."""
        return int(self.wake_up_freq_ns)

    def kernel_starting(self, start_time: int) -> None:
        """Called when the kernel starts. Initialize strategy and request initial data."""
        super().kernel_starting(start_time)

        # Initialize the strategy with config
        config = AgentConfig(
            starting_cash=self.starting_cash,
            symbol=self.symbol,
            latency_ns=0,  # Will be refined based on actual latency model
        )
        self.strategy.initialize(config)
        self._initialized = True

        # Track initial holdings for change detection
        self._previous_holdings = dict(self.holdings)

        # Schedule first wakeup (which will request market data)
        self.set_wakeup(start_time + self.wake_up_freq_ns)

    def wakeup(self, current_time: int) -> None:
        """Called periodically. Request market data update."""
        super().wakeup(current_time)

        # Request current spread to trigger on_market_data
        self.get_current_spread(self.symbol)

        # Schedule next wakeup
        self.set_wakeup(current_time + self.wake_up_freq_ns)

    def receive_message(self, current_time: int, sender_id: int, message) -> None:
        """Process incoming messages and route to strategy callbacks."""
        super().receive_message(current_time, sender_id, message)

        if not self._initialized:
            return

        # Market data update (response to get_current_spread) → on_market_data
        if isinstance(message, QuerySpreadResponseMsg):
            self._handle_market_data(current_time)

    def _handle_market_data(self, current_time: int) -> None:
        """Build MarketState and call strategy.on_market_data()."""
        # Detect position changes since last update
        # current_symbol_holding = self.holdings.get(self.symbol, 0)
        # previous_symbol_holding = self._previous_holdings.get(self.symbol, 0)

        # If holdings changed, it means an order was filled - we could notify here
        # For now, the strategy sees the updated state in MarketState

        state = self._build_market_state(current_time)
        actions = self.strategy.on_market_data(state)
        self._execute_actions(actions)

        # Update previous holdings
        self._previous_holdings = dict(self.holdings)

    def _build_market_state(self, current_time: int) -> MarketState:
        """Build a MarketState from current TradingAgent state."""
        # get_known_bid_ask returns (bid_price, bid_size, ask_price, ask_size)
        bid, _, ask, _ = self.get_known_bid_ask(self.symbol)

        # Build open orders list
        open_orders = [
            Order(
                order_id=order.order_id,
                symbol=order.symbol,
                side=Side.BID if order.side == AbidesSide.BID else Side.ASK,
                quantity=order.quantity,
                price=order.limit_price,
                order_type=OrderType.LIMIT,
                status=OrderStatus.NEW,  # Active orders are "NEW" from strategy perspective
                filled_quantity=0,
            )
            for order in self.orders.values()
        ]

        return MarketState(
            timestamp_ns=current_time,
            best_bid=bid,
            best_ask=ask,
            last_trade=None,  # Would require additional tracking
            inventory=self.holdings.get(self.symbol, 0),
            cash=self.holdings.get("CASH", 0),
            open_orders=open_orders,
        )

    def _execute_actions(self, actions: list[OrderAction]) -> None:
        """Execute a list of OrderActions by placing/cancelling orders."""
        for action in actions:
            if action.cancel_order_id is not None:
                # Cancel existing order
                order_to_cancel = self.orders.get(action.cancel_order_id)
                if order_to_cancel:
                    self.cancel_order(order_to_cancel)
            elif action.order_type == OrderType.MARKET:
                # Place market order
                side = AbidesSide.BID if action.side == Side.BID else AbidesSide.ASK
                self.place_market_order(self.symbol, action.quantity, side)
            else:
                # Place limit order
                if action.price is None:
                    continue  # Skip invalid limit orders without price
                side = AbidesSide.BID if action.side == Side.BID else AbidesSide.ASK
                self.place_limit_order(self.symbol, action.quantity, side, action.price)
