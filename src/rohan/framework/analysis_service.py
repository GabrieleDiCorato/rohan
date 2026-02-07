import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rohan.simulation import AgentMetrics, SimulationMetrics, SimulationOutput

# Use non-interactive backend for server/CI environments
matplotlib.use("Agg")


class AnalysisService:
    """Service to analyze simulation results."""

    @staticmethod
    def compute_metrics(result: SimulationOutput) -> SimulationMetrics:
        """Computes summary metrics from the simulation output."""
        l1 = result.get_order_book_l1()

        # Basic PnL Calculation (assuming we track a specific agent, e.g. the strategic one)
        # For Phase 1 (Observer), we might look at the aggregate market stats or a specific agent type.
        # Let's calculate market-wide stats for now.

        if l1.empty:
            return SimulationMetrics(
                total_pnl=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                volatility=0.0,
                traded_volume=0,
                custom_metrics={},
            )

        # Volatility (Std Dev of Mid Price returns)
        l1["mid_price"] = (l1["bid_price"] + l1["ask_price"]) / 2
        l1["returns"] = l1["mid_price"].pct_change().fillna(0)
        volatility = l1["returns"].std() * np.sqrt(252 * 390 * 60)  # Annualized approx (assuming 1 sec samples) - rough

        # Spread
        mean_spread = (l1["ask_price"] - l1["bid_price"]).mean()

        # Market Impact / Liquidity (Avg Volume at Best)
        avg_bid_liq = l1["bid_qty"].mean()
        avg_ask_liq = l1["ask_qty"].mean()

        return SimulationMetrics(
            total_pnl=0.0,  # Placeholder until we target a specific agent
            sharpe_ratio=0.0,  # Placeholder
            max_drawdown=0.0,  # Placeholder
            win_rate=0.0,  # Placeholder
            volatility=float(volatility) if not pd.isna(volatility) else 0.0,
            traded_volume=0,  # Placeholder
            custom_metrics={
                "mean_spread": float(mean_spread),
                "avg_bid_liquidity": float(avg_bid_liq),
                "avg_ask_liquidity": float(avg_ask_liq),
            },
        )

    @staticmethod
    def compute_agent_metrics(result: SimulationOutput, agent_id: int) -> AgentMetrics:
        """Computes metrics for a specific agent."""
        if not hasattr(result, "end_state"):
            return AgentMetrics(agent_id=agent_id)

        agents = {a.id: a for a in result.end_state["agents"]}
        if agent_id not in agents:
            raise ValueError(f"Agent {agent_id} not found in simulation output")

        agent = agents[agent_id]

        # Extract holdings
        cash = agent.holdings.get("CASH", 0)
        inventory = 0
        for k, v in agent.holdings.items():
            if k != "CASH":
                inventory = v
                break

        # Get last mid price for Mark-to-Market
        l1 = result.get_order_book_l1()
        if not l1.empty:
            last_row = l1.iloc[-1]
            last_mid = (last_row["bid_price"] + last_row["ask_price"]) / 2
        else:
            last_mid = 0

        total_pnl = cash + (inventory * last_mid)

        # Parse logs for trade stats
        # Agent logs structure: list of (time, type, payload) usually?
        # ABIDES EventType is an enum, usually logged as string name or enum object.
        # We need to robustly check event type.
        log = getattr(agent, "log", [])

        executed_orders = [e for e in log if hasattr(e, "event_type") and str(e.event_type) == "ORDER_EXECUTED" or (isinstance(e, tuple) and e[1] == "ORDER_EXECUTED")]
        submitted_orders = [e for e in log if hasattr(e, "event_type") and str(e.event_type) == "ORDER_SUBMITTED" or (isinstance(e, tuple) and e[1] == "ORDER_SUBMITTED")]

        # Note: ABIDES Kernel log format varies. Assuming tuple (time, type, ...) or object with event_type.
        # My StrategicAgentAdapter inherits TradingAgent which uses self.log_event(EventType, Msg).
        # We'll need to verify this against actual ABIDES log format.

        trade_count = len(executed_orders)
        fill_rate = trade_count / len(submitted_orders) if submitted_orders else 0.0

        return AgentMetrics(
            agent_id=agent_id,
            realized_pnl=float(cash),
            unrealized_pnl=float(inventory * last_mid),
            total_pnl=float(total_pnl),
            sharpe_ratio=0.0,  # Placeholder
            max_drawdown=0.0,  # Placeholder
            trade_count=trade_count,
            fill_rate=float(fill_rate),
            end_inventory=inventory,
        )

    @staticmethod
    def plot_price_series(result: SimulationOutput) -> plt.Figure:
        """
        Generate a plot of price series (bid, ask, mid) over time.
        Returns a matplotlib Figure object.
        """
        l1 = result.get_order_book_l1()

        if l1.empty:
            # Return empty figure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return fig

        # Calculate mid price
        l1["mid_price"] = (l1["bid_price"] + l1["ask_price"]) / 2

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot prices
        ax.plot(l1.index, l1["bid_price"], label="Bid", alpha=0.7, linewidth=1)
        ax.plot(l1.index, l1["ask_price"], label="Ask", alpha=0.7, linewidth=1)
        ax.plot(l1.index, l1["mid_price"], label="Mid", linewidth=2)

        ax.set_xlabel("Time Index")
        ax.set_ylabel("Price")
        ax.set_title("Price Series")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_volume(result: SimulationOutput) -> plt.Figure:
        """
        Generate a plot of volume at best bid/ask over time.
        Returns a matplotlib Figure object.
        """
        l1 = result.get_order_book_l1()

        if l1.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return fig

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot volumes
        ax.plot(l1.index, l1["bid_qty"], label="Bid Volume", alpha=0.7)
        ax.plot(l1.index, l1["ask_qty"], label="Ask Volume", alpha=0.7)

        ax.set_xlabel("Time Index")
        ax.set_ylabel("Quantity")
        ax.set_title("Volume at Best Bid/Ask")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_spread(result: SimulationOutput) -> plt.Figure:
        """
        Generate a plot of bid-ask spread over time.
        Returns a matplotlib Figure object.
        """
        l1 = result.get_order_book_l1()

        if l1.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return fig

        # Calculate spread
        l1["spread"] = l1["ask_price"] - l1["bid_price"]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(l1.index, l1["spread"], linewidth=1.5)
        ax.fill_between(l1.index, l1["spread"], alpha=0.3)

        ax.set_xlabel("Time Index")
        ax.set_ylabel("Spread")
        ax.set_title("Bid-Ask Spread Over Time")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def generate_plots() -> list[str]:
        """Generates plots and returns a list of paths (virtual or temp).
        In the final system these bytes are returned to be saved in DB.
        For now we just return bytes? Or create matplotlib figures.
        """
        # TODO: Implement Plotting to returning bytes for DB storage
        return []
