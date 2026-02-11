"""Quick debug script to trace order placement."""

from rohan.config import SimulationSettings
from rohan.simulation.models.strategy_api import OrderAction, OrderType, Side
from rohan.simulation.simulation_service import SimulationService


class DebugStrategy:
    def initialize(self, config):
        print(f"[INIT] {config.symbol}")
        self.tick = 0

    def on_market_data(self, state):
        self.tick += 1
        print(f"[TICK {self.tick}] open={len(state.open_orders)}, bbo={state.best_bid}/{state.best_ask}")
        if self.tick == 1 and state.best_ask:
            price = state.best_ask - 1
            print(f"  -> Placing BID @ {price}")
            return [OrderAction(side=Side.BID, quantity=1, price=price, order_type=OrderType.LIMIT)]
        if self.tick == 3:
            print(f"  -> Order IDs in state: {[o.order_id for o in state.open_orders]}")
        return []

    def on_order_update(self, update):
        print(f"[ORDER_UPDATE] id={update.order_id} status={update.status}")
        return []


if __name__ == "__main__":
    s = DebugStrategy()
    settings = SimulationSettings(
        seed=42,
        start_time="09:30:00",
        end_time="09:35:00",  # 5 min minimum
        _env_file=None,  # type: ignore[call-arg]  # Pydantic-settings parameter
    )
    result = SimulationService().run_simulation(settings, strategy=s)
    print(f"\n[DONE] error={result.error}")
