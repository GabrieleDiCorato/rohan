from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from rohan.simulation import AgentMetrics, SimulationMetrics, SimulationOutput

if TYPE_CHECKING:
    from rohan.simulation import ComparisonResult, RunSummary

# Use non-interactive backend for server/CI environments
matplotlib.use("Agg")


class AnalysisService:
    """Service to analyze simulation results.

    All monetary values are in **integer cents**, matching ABIDES conventions.
    """

    @staticmethod
    def compute_metrics(result: SimulationOutput) -> SimulationMetrics:
        """Compute market-wide metrics from L1 order-book snapshots."""
        l1 = result.get_order_book_l1()

        if l1.empty:
            return SimulationMetrics()

        # --- Mid price & returns ---
        l1 = l1.copy()
        l1["mid_price"] = (l1["bid_price"] + l1["ask_price"]) / 2
        l1["returns"] = l1["mid_price"].pct_change().fillna(0)

        # --- Volatility (annualised) ---
        # Derive annualisation factor from actual observation frequency,
        # not a hard-coded assumption of 1-second bars.
        time_col = l1["time"]
        if len(time_col) > 1:
            median_dt_ns = float(time_col.diff().dropna().median())
            if median_dt_ns > 0:
                trading_day_ns = 6.5 * 3600 * 1e9  # 6.5 h
                obs_per_day = trading_day_ns / median_dt_ns
                obs_per_year = obs_per_day * 252
                vol = float(l1["returns"].std()) * np.sqrt(obs_per_year)
            else:
                vol = None
        else:
            vol = None

        volatility = vol if vol is not None and not pd.isna(vol) else None

        # --- Quoted spread ---
        mean_spread = float((l1["ask_price"] - l1["bid_price"]).mean())

        # --- Liquidity ---
        avg_bid_liq = float(l1["bid_qty"].mean())
        avg_ask_liq = float(l1["ask_qty"].mean())

        return SimulationMetrics(
            volatility=volatility,
            mean_spread=mean_spread if not pd.isna(mean_spread) else None,
            avg_bid_liquidity=avg_bid_liq if not pd.isna(avg_bid_liq) else None,
            avg_ask_liquidity=avg_ask_liq if not pd.isna(avg_ask_liq) else None,
        )

    @staticmethod
    def compute_agent_metrics(
        result: SimulationOutput,
        agent_id: int,
        initial_cash: int = 0,
    ) -> AgentMetrics:
        """Compute performance metrics for a specific agent.

        Args:
            result: Simulation output containing end_state.
            agent_id: Numeric ID of the agent to analyse.
            initial_cash: The agent's starting cash (integer cents).  Required
                for correct PnL computation.
        """
        if not hasattr(result, "end_state"):
            return AgentMetrics(agent_id=agent_id, initial_cash=initial_cash)

        agents = {a.id: a for a in result.end_state["agents"]}
        if agent_id not in agents:
            raise ValueError(f"Agent {agent_id} not found in simulation output")

        agent = agents[agent_id]

        # --- Holdings ---
        ending_cash = agent.holdings.get("CASH", 0)
        inventory = 0
        for k, v in agent.holdings.items():
            if k != "CASH":
                inventory = v
                break

        # --- Mark-to-market ---
        l1 = result.get_order_book_l1()
        if not l1.empty:
            last_row = l1.iloc[-1]
            last_mid = float((last_row["bid_price"] + last_row["ask_price"]) / 2)
        else:
            last_mid = 0.0

        ending_value = ending_cash + (inventory * last_mid)
        total_pnl = ending_value - initial_cash

        # --- Execution stats ---
        log = getattr(agent, "log", [])

        executed_orders = [e for e in log if (hasattr(e, "event_type") and str(e.event_type) == "ORDER_EXECUTED") or (isinstance(e, tuple) and len(e) > 1 and e[1] == "ORDER_EXECUTED")]
        submitted_orders = [e for e in log if (hasattr(e, "event_type") and str(e.event_type) == "ORDER_SUBMITTED") or (isinstance(e, tuple) and len(e) > 1 and e[1] == "ORDER_SUBMITTED")]

        trade_count = len(executed_orders)
        n_submitted = len(submitted_orders)
        fill_rate = trade_count / n_submitted if n_submitted else None
        otr = n_submitted / trade_count if trade_count else None

        return AgentMetrics(
            agent_id=agent_id,
            initial_cash=initial_cash,
            ending_cash=ending_cash,
            total_pnl=total_pnl,
            trade_count=trade_count,
            fill_rate=fill_rate,
            order_to_trade_ratio=otr,
            end_inventory=inventory,
        )

    @staticmethod
    def plot_price_series(result: SimulationOutput, title: str = "Price Series") -> Figure:
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
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_volume(result: SimulationOutput, title: str = "Volume at Best Bid/Ask") -> Figure:
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
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_spread(result: SimulationOutput, title: str = "Bid-Ask Spread Over Time") -> Figure:
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
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def figure_to_base64(fig: Figure) -> str:
        """Convert a matplotlib figure to a base64-encoded PNG string.

        Args:
            fig: Matplotlib Figure object.

        Returns:
            Base64-encoded PNG string.
        """
        import base64

        raw = AnalysisService.figure_to_bytes(fig)
        return base64.b64encode(raw).decode("utf-8")

    @staticmethod
    def figure_to_bytes(fig: Figure) -> bytes:
        """Convert a matplotlib figure to raw PNG bytes.

        Args:
            fig: Matplotlib Figure object.

        Returns:
            PNG bytes suitable for storage as an artifact.
        """
        from io import BytesIO

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        data = buf.read()
        buf.close()
        plt.close(fig)
        return data

    @staticmethod
    def generate_summary(
        comparison: "ComparisonResult",
        strategy_output: SimulationOutput | None = None,
        duration_seconds: float = 0.0,
        error: str | None = None,
    ) -> "RunSummary":
        """Generate a structured summary for LLM interpretation.

        Args:
            comparison: ComparisonResult from run_with_baseline.
            strategy_output: Optional SimulationOutput to generate charts.
            duration_seconds: Execution time.
            error: Error message if any.

        Returns:
            RunSummary with metrics and optional charts.
        """
        from rohan.simulation import RunSummary

        price_chart = None
        spread_chart = None

        if strategy_output is not None:
            try:
                price_fig = AnalysisService.plot_price_series(strategy_output)
                price_chart = AnalysisService.figure_to_base64(price_fig)
            except Exception:
                pass

            try:
                spread_fig = AnalysisService.plot_spread(strategy_output)
                spread_chart = AnalysisService.figure_to_base64(spread_fig)
            except Exception:
                pass

        return RunSummary(
            comparison=comparison,
            price_chart=price_chart,
            spread_chart=spread_chart,
            duration_seconds=duration_seconds,
            error=error,
        )
