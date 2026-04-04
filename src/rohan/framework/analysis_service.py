from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from abides_core.utils import ns_date
from matplotlib.figure import Figure

from rohan.framework.analysis_models import (
    CounterpartySummary,
    FillRecord,
    InventoryPoint,
    L2Snapshot,
    MidPricePoint,
    OrderLifecycleRecord,
    PnLPoint,
    RichAnalysisBundle,
)
from rohan.simulation import AgentMetrics, SimulationMetrics, SimulationOutput
from rohan.simulation.abides_impl.hasufel_output import HasufelOutput

if TYPE_CHECKING:
    from rohan.simulation import ComparisonResult, RunSummary

# Use non-interactive backend for server/CI environments
matplotlib.use("Agg")

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service to analyze simulation results.

    All monetary values are in **integer cents**, matching ABIDES conventions.

    Design note on NaN in L1 data
    -----------------------------
    A NaN ``bid_price`` (or ``ask_price``) means **no resting orders exist on
    that side of the book** at that instant — it is a genuine illiquidity
    event, not a data artifact.  We therefore **never** forward-fill prices.
    Metrics that require a well-defined mid-price (volatility, returns,
    effective spread) are computed only from *two-sided* snapshots where
    both bid and ask are present.  The ``pct_time_two_sided`` metric
    measures the fraction of snapshots that are two-sided (i.e. market
    availability).  Average bid/ask liquidity is computed from all rows
    where each respective side is present.
    """

    # ==================================================================
    # Shared extraction helpers
    # ==================================================================

    @staticmethod
    def _get_two_sided_l1(result: SimulationOutput) -> pd.DataFrame:
        """Extract two-sided L1 rows (both bid and ask present).

        Returns an empty DataFrame when no two-sided observations exist.
        """
        l1 = result.get_order_book_l1()
        if l1.empty:
            return pd.DataFrame()
        return l1.dropna(subset=["bid_price", "ask_price"]).copy()

    # ==================================================================
    # Market-wide metrics
    # ==================================================================

    @staticmethod
    def compute_metrics(result: HasufelOutput) -> SimulationMetrics:
        """Compute market-wide metrics from hasufel's rich metrics.

        Returns a :class:`SimulationMetrics` with all computable fields
        populated.  Fields remain ``None`` when there is insufficient data.
        """
        from abides_markets.simulation import (
            compute_avg_liquidity,
            compute_effective_spread,
            compute_mean_spread,
            compute_volatility,
            compute_vpin,
        )

        rich = result.rich_metrics
        ticker = result._ticker  # noqa: SLF001
        market = rich.markets.get(ticker)
        if market is None:
            return SimulationMetrics()

        l1 = market.l1_series
        if l1 is None or len(l1.times_ns) == 0:
            return SimulationMetrics()

        micro = market.microstructure

        # Standalone helpers operate on hasufel's L1Snapshots directly
        volatility = compute_volatility(l1)
        mean_spread = compute_mean_spread(l1)
        avg_bid_liq, avg_ask_liq = compute_avg_liquidity(l1)

        # Build fills tuples (price_cents, quantity, fill_time_ns) from trades
        fill_tuples: list[tuple[int, int, int]] = []
        if market.trades is not None:
            for t in market.trades:
                fill_tuples.append((t.price_cents, t.quantity, t.time_ns))

        effective_spread = compute_effective_spread(fill_tuples, l1) if fill_tuples else None
        vpin = compute_vpin(fill_tuples, l1) if fill_tuples else None

        # Traded volume from liquidity metrics
        traded_volume: int | None = None
        if market.liquidity is not None:
            traded_volume = market.liquidity.total_exchanged_volume

        # pct_time_two_sided: hasufel reports 0-100, Rohan expects 0-1
        pct_two_sided: float | None = None
        if micro is not None:
            pct_two_sided = micro.pct_time_two_sided / 100.0

        return SimulationMetrics(
            volatility=volatility,
            mean_spread=mean_spread,
            effective_spread=effective_spread,
            avg_bid_liquidity=avg_bid_liq,
            avg_ask_liquidity=avg_ask_liq,
            traded_volume=traded_volume,
            lob_imbalance_mean=micro.lob_imbalance_mean if micro else None,
            lob_imbalance_std=micro.lob_imbalance_std if micro else None,
            vpin=vpin,
            resilience_mean_ns=micro.resilience_mean_ns if micro else None,
            market_ott_ratio=micro.market_ott_ratio if micro else None,
            pct_time_two_sided=pct_two_sided,
        )

    # ==================================================================
    # Per-agent metrics
    # ==================================================================

    @staticmethod
    def compute_agent_metrics(
        result: HasufelOutput,
        agent_id: int,
    ) -> AgentMetrics:
        """Compute performance metrics for a specific agent.

        Args:
            result: Simulation output (HasufelOutput).
            agent_id: Numeric ID of the agent to analyse.

        Metrics are populated from hasufel's ``RichAgentMetrics``.
        """
        rich = result.rich_metrics

        # Find the RichAgentMetrics for this agent_id
        rich_agent = None
        for ra in rich.agents:
            if ra.agent_id == agent_id:
                rich_agent = ra
                break
        if rich_agent is None:
            raise ValueError(f"Agent {agent_id} not found in simulation output")

        # Find matching AgentData from the underlying SimulationResult
        agent_data = None
        for ad in result.hasufel_result.agents:
            if ad.agent_id == agent_id:
                agent_data = ad
                break

        # Capital from AgentData
        ending_cash = agent_data.final_holdings.get("CASH", 0) if agent_data else 0

        # End inventory: sum of non-CASH holdings
        end_inv = sum(v for k, v in rich_agent.end_inventory.items()) if rich_agent.end_inventory else 0

        # fill_rate: hasufel 0-100, Rohan 0-1
        fill_rate: float | None = None
        if rich_agent.fill_rate_pct is not None:
            fill_rate = rich_agent.fill_rate_pct / 100.0

        return AgentMetrics(
            agent_id=agent_id,
            ending_cash=ending_cash,
            total_pnl=float(rich_agent.total_pnl_cents) if rich_agent.total_pnl_cents is not None else None,
            sharpe_ratio=rich_agent.sharpe_ratio,
            max_drawdown=float(rich_agent.max_drawdown_cents) if rich_agent.max_drawdown_cents is not None else None,
            inventory_std=rich_agent.inventory_std,
            trade_count=rich_agent.trade_count,
            fill_rate=fill_rate,
            order_to_trade_ratio=rich_agent.order_to_trade_ratio,
            vwap_cents=rich_agent.vwap_cents,
            end_inventory=end_inv,
        )

    # ==================================================================
    # Private: shared helpers
    # ==================================================================

    @staticmethod
    def _build_mid_lookup(two_sided: pd.DataFrame) -> pd.Series:
        """Build a sorted mid-price Series indexed by ns-since-midnight."""
        mid = (two_sided["bid_price"] + two_sided["ask_price"]) / 2
        return mid.set_axis(two_sided["time"]).sort_index()

    @staticmethod
    def _nearest_mid(mid_lookup: pd.Series, t_midnight: int) -> float | None:
        """Find the nearest mid-price to a given timestamp."""
        if mid_lookup.empty:
            return None
        idx = int(mid_lookup.index.searchsorted(t_midnight))
        candidates = []
        if idx > 0:
            candidates.append(mid_lookup.index[idx - 1])
        if idx < len(mid_lookup):
            candidates.append(mid_lookup.index[idx])
        if not candidates:
            return None
        nearest_t = min(candidates, key=lambda t: abs(t - t_midnight))
        val = mid_lookup.loc[nearest_t]
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        return float(val) if not pd.isna(val) else None

    # ==================================================================
    # Rich analysis methods
    # ==================================================================

    @staticmethod
    def get_fill_analysis(
        result: HasufelOutput,
        agent_id: int,
    ) -> list[FillRecord]:
        """Per-fill execution quality with slippage and counterparty info.

        Returns an empty list if the agent has no fills.
        """
        rich = result.rich_metrics
        if rich.fills is None:
            return []

        # Agent type lookup from hasufel result
        agent_type_map: dict[int, str] = {a.agent_id: a.agent_type for a in result.hasufel_result.agents}

        # Build counterparty map from TradeAttribution
        cp_map: dict[tuple[int, int, int, int], str] = {}
        ticker = result._ticker  # noqa: SLF001
        market = result.hasufel_result.markets.get(ticker)
        if market is not None and market.trades is not None:
            for t in market.trades:
                cp_map[(t.time_ns, t.price_cents, t.quantity, t.passive_agent_id)] = agent_type_map.get(t.aggressive_agent_id, "Unknown")
                cp_map[(t.time_ns, t.price_cents, t.quantity, t.aggressive_agent_id)] = agent_type_map.get(t.passive_agent_id, "Unknown")

        # Build mid lookup for mid_at_fill
        two_sided = AnalysisService._get_two_sided_l1(result)
        mid_lookup = AnalysisService._build_mid_lookup(two_sided) if not two_sided.empty else pd.Series(dtype=float)

        records: list[FillRecord] = []
        for f in rich.fills:
            if f.agent_id != agent_id:
                continue

            t_midnight = f.time_ns - ns_date(f.time_ns)
            mid = AnalysisService._nearest_mid(mid_lookup, t_midnight)
            cp_type = cp_map.get((f.time_ns, f.price_cents, f.quantity, f.agent_id))

            records.append(
                FillRecord(
                    timestamp_ns=t_midnight,
                    side=f.side,
                    price=f.price_cents,
                    qty=f.quantity,
                    mid_at_fill=mid,
                    slippage_bps=float(f.slippage_bps) if f.slippage_bps is not None else None,
                    counterparty_type=cp_type,
                )
            )
        return records

    @staticmethod
    def get_pnl_curve(
        result: HasufelOutput,
        agent_id: int,
    ) -> list[PnLPoint]:
        """Mark-to-market PnL curve from hasufel's dense EquityCurve.

        Uses ``compute_equity_curve(fill_events, l1=...)`` to produce one
        NAV observation per two-sided L1 tick.

        Returns an empty list when there are no fills or no equity curve.
        """
        from abides_markets.simulation import compute_equity_curve

        # Find matching AgentData for this agent
        agent_data = None
        for ad in result.hasufel_result.agents:
            if ad.agent_id == agent_id:
                agent_data = ad
                break

        if agent_data is None or agent_data.equity_curve is None:
            return []

        sparse = agent_data.equity_curve

        # Attempt dense interpolation against L1
        ticker = result._ticker  # noqa: SLF001
        market = result.hasufel_result.markets.get(ticker)
        l1 = market.l1_series if market else None

        fill_events = list(zip(sparse.times_ns, sparse.nav_cents, sparse.peak_nav_cents, strict=True))
        dense = compute_equity_curve(fill_events, l1=l1) if l1 else None
        ec = dense if dense is not None else sparse

        initial_nav = ec.nav_cents[0] if len(ec.nav_cents) > 0 else 0

        points: list[PnLPoint] = []
        for i in range(len(ec.times_ns)):
            t_midnight = ec.times_ns[i] - ns_date(ec.times_ns[i])
            pnl = float(ec.nav_cents[i] - initial_nav)
            points.append(PnLPoint(timestamp_ns=int(t_midnight), mark_to_market_pnl=pnl))

        return points

    @staticmethod
    def get_inventory_trajectory(
        result: HasufelOutput,
        agent_id: int,
    ) -> list[InventoryPoint]:
        """Position trajectory at each fill event.

        Returns a list of ``InventoryPoint`` showing cumulative position
        after each fill.  Includes an initial point at timestamp 0 with
        position 0.
        """
        rich = result.rich_metrics
        if rich.fills is None:
            return []

        agent_fills = [f for f in rich.fills if f.agent_id == agent_id]
        if not agent_fills:
            return []

        agent_fills.sort(key=lambda f: f.time_ns)

        points: list[InventoryPoint] = [InventoryPoint(timestamp_ns=0, position=0)]
        cumulative = 0
        for f in agent_fills:
            signed_qty = f.quantity if f.side == "BUY" else -f.quantity
            cumulative += signed_qty
            t_midnight = f.time_ns - ns_date(f.time_ns)
            points.append(InventoryPoint(timestamp_ns=int(t_midnight), position=cumulative))
        return points

    @staticmethod
    def get_adverse_selection(
        result: HasufelOutput,
        agent_id: int,
        window_ns: int = 500_000_000,
    ) -> float | None:
        """Average mid-price move against fill direction within a look-ahead window.

        Returns the average adverse move in basis points (positive = adverse).
        Returns ``None`` if there are no fills or insufficient L1 data.
        """
        rich = result.rich_metrics
        if rich.fills is None:
            return None

        # Map window_ns to the closest label in hasufel's adverse_selection_bps
        window_labels = {
            100_000_000: "100ms",
            500_000_000: "500ms",
            1_000_000_000: "1s",
            5_000_000_000: "5s",
        }
        label = window_labels.get(window_ns)

        agent_fills = [f for f in rich.fills if f.agent_id == agent_id]
        if not agent_fills:
            return None

        if label is not None:
            # Use pre-computed adverse selection from hasufel
            values = [float(f.adverse_selection_bps[label]) for f in agent_fills if label in f.adverse_selection_bps and f.adverse_selection_bps[label] is not None]  # pyright: ignore[reportArgumentType]
            if not values:
                return None
            # Hasufel convention: negative = adverse. Rohan convention: positive = adverse.
            return float(-np.mean(values))

        # Non-standard window: fall back to local computation using L1
        two_sided = AnalysisService._get_two_sided_l1(result)
        if two_sided.empty:
            return None

        mid_lookup = AnalysisService._build_mid_lookup(two_sided)
        mid_times = mid_lookup.index.values
        mid_values = mid_lookup.values

        adverse_moves: list[float] = []
        for f in agent_fills:
            t_midnight = f.time_ns - ns_date(f.time_ns)
            mid_at_fill = AnalysisService._nearest_mid(mid_lookup, t_midnight)
            if mid_at_fill is None or mid_at_fill <= 0:
                continue

            target_time = t_midnight + window_ns
            idx = int(mid_times.searchsorted(target_time))
            if idx >= len(mid_times):
                idx = len(mid_times) - 1
            if idx < 0:
                continue

            raw_mid = mid_values[idx]
            if pd.isna(raw_mid):
                continue
            future_mid = float(raw_mid)

            side_sign = 1.0 if f.side == "BUY" else -1.0
            move_bps = -side_sign * (future_mid - mid_at_fill) / mid_at_fill * 10_000
            adverse_moves.append(move_bps)

        if not adverse_moves:
            return None
        return float(np.mean(adverse_moves))

    @staticmethod
    def get_order_lifecycle(
        result: HasufelOutput,
        agent_id: int,
    ) -> list[OrderLifecycleRecord]:
        """Order lifecycle: submission → fill/cancel/resting for each order.

        Populated from hasufel's ``RichAgentMetrics.order_lifecycles``.
        """
        rich = result.rich_metrics

        rich_agent = None
        for ra in rich.agents:
            if ra.agent_id == agent_id:
                rich_agent = ra
                break

        if rich_agent is None or not rich_agent.order_lifecycles:
            return []

        records: list[OrderLifecycleRecord] = []
        for lc in rich_agent.order_lifecycles:
            submitted_at = lc.submitted_at_ns - ns_date(lc.submitted_at_ns)

            # Hasufel has 'partially_filled' — map to 'filled'
            status = "filled" if lc.status in ("filled", "partially_filled") else lc.status

            resting_time: int | None = None
            if lc.resting_time_ns is not None and lc.resting_time_ns > 0:
                resting_time = lc.resting_time_ns

            records.append(
                OrderLifecycleRecord(
                    order_id=lc.order_id,
                    submitted_at_ns=submitted_at,
                    status=status,
                    resting_time_ns=resting_time,
                    filled_qty=lc.filled_qty,
                    submitted_qty=lc.submitted_qty,
                )
            )

        return records

    @staticmethod
    def get_counterparty_breakdown(
        result: HasufelOutput,
        agent_id: int,
    ) -> list[CounterpartySummary]:
        """Breakdown of which agent types the strategy traded against."""
        agent_type_map: dict[int, str] = {a.agent_id: a.agent_type for a in result.hasufel_result.agents}

        ticker = result._ticker  # noqa: SLF001
        market = result.hasufel_result.markets.get(ticker)
        if market is None or market.trades is None:
            return []

        type_stats: dict[str, dict] = {}
        for t in market.trades:
            if t.passive_agent_id == agent_id:
                cp_id = t.aggressive_agent_id
            elif t.aggressive_agent_id == agent_id:
                cp_id = t.passive_agent_id
            else:
                continue

            cp_type = agent_type_map.get(cp_id, "Unknown")
            if cp_type not in type_stats:
                type_stats[cp_type] = {"count": 0, "total_qty": 0}
            type_stats[cp_type]["count"] += 1
            type_stats[cp_type]["total_qty"] += t.quantity

        return [
            CounterpartySummary(
                agent_type=at,
                trade_count=stats["count"],
                avg_size=stats["total_qty"] / stats["count"] if stats["count"] > 0 else 0.0,
                total_volume=stats["total_qty"],
            )
            for at, stats in sorted(type_stats.items())
        ]

    @staticmethod
    def query_book_depth(
        result: SimulationOutput,
        timestamp_ns: int,
        n_levels: int = 5,
    ) -> dict[str, list[tuple[float, int]]]:
        """L2 order-book snapshot at the nearest timestamp.

        Note: requires a live ``SimulationOutput`` — cannot be used from
        serialized data.  For container scenarios, pre-compute interesting
        snapshots during the executor phase.

        Returns ``{"bids": [(price, qty), ...], "asks": [(price, qty), ...]}``
        or empty lists if data is unavailable.
        """
        try:
            l2 = result.get_order_book_l2(n_levels)
        except Exception:
            return {"bids": [], "asks": []}

        if l2.empty:
            return {"bids": [], "asks": []}

        # Find the nearest timestamp in the L2 data
        times = l2["time"].unique()
        if len(times) == 0:
            return {"bids": [], "asks": []}

        nearest_t = times[int(np.argmin(np.abs(times.astype(np.int64) - timestamp_ns)))]
        snapshot = l2[l2["time"] == nearest_t]

        bids_df = snapshot.loc[snapshot["side"] == "bid"].sort_values("level")
        asks_df = snapshot.loc[snapshot["side"] == "ask"].sort_values("level")

        bids = [(float(row["price"]), int(row["qty"])) for _, row in bids_df.iterrows()]
        asks = [(float(row["price"]), int(row["qty"])) for _, row in asks_df.iterrows()]

        return {"bids": bids, "asks": asks}

    @staticmethod
    def compute_rich_analysis(
        result: HasufelOutput,
        agent_id: int,
        adverse_window_ns: int = 500_000_000,
    ) -> RichAnalysisBundle:
        """Compute the full rich-analysis bundle for one agent in one scenario.

        Convenience method that calls all analysis methods and returns
        a serialisable ``RichAnalysisBundle``.  The bundle includes raw
        queryable data (mid-price series, sampled L2 snapshots) so that
        investigation tools can operate purely from serialised JSON.
        """
        fills = AnalysisService.get_fill_analysis(result, agent_id)
        pnl_curve = AnalysisService.get_pnl_curve(result, agent_id)
        inventory = AnalysisService.get_inventory_trajectory(result, agent_id)
        adverse = AnalysisService.get_adverse_selection(result, agent_id, adverse_window_ns)
        counterparties = AnalysisService.get_counterparty_breakdown(result, agent_id)
        lifecycle = AnalysisService.get_order_lifecycle(result, agent_id)

        # --- Multi-window adverse selection ---
        adverse_by_window: dict[str, float] = {}
        windows = {
            "100ms": 100_000_000,
            "500ms": 500_000_000,
            "1s": 1_000_000_000,
            "5s": 5_000_000_000,
        }
        for label, wns in windows.items():
            val = AnalysisService.get_adverse_selection(result, agent_id, wns)
            if val is not None:
                adverse_by_window[label] = val

        # --- Mid-price series for investigation tools ---
        two_sided = AnalysisService._get_two_sided_l1(result)
        mid_price_series: list[MidPricePoint] = []
        if not two_sided.empty:
            mid_lookup = AnalysisService._build_mid_lookup(two_sided)
            for t, m in zip(mid_lookup.index, mid_lookup.values, strict=False):
                if not pd.isna(m):
                    mid_price_series.append(MidPricePoint(timestamp_ns=int(t), mid_price=float(m)))

        # --- Sampled L2 snapshots ---
        l2_snapshots: list[L2Snapshot] = []
        snapshot_times: set[int] = set()
        # Snapshot at each fill timestamp
        for f in fills:
            snapshot_times.add(f.timestamp_ns)
        # Snapshot at PnL turning points (local extrema)
        if len(pnl_curve) >= 3:
            for i in range(1, len(pnl_curve) - 1):
                prev_pnl = pnl_curve[i - 1].mark_to_market_pnl
                curr_pnl = pnl_curve[i].mark_to_market_pnl
                next_pnl = pnl_curve[i + 1].mark_to_market_pnl
                if (curr_pnl >= prev_pnl and curr_pnl >= next_pnl) or (curr_pnl <= prev_pnl and curr_pnl <= next_pnl):
                    snapshot_times.add(pnl_curve[i].timestamp_ns)
        # Regular interval snapshots (~5s apart)
        if mid_price_series:
            t_min = mid_price_series[0].timestamp_ns
            t_max = mid_price_series[-1].timestamp_ns
            interval = 5_000_000_000  # 5 seconds
            t = t_min
            while t <= t_max:
                snapshot_times.add(t)
                t += interval

        # Cap total snapshots to keep bundle size manageable
        sorted_times = sorted(snapshot_times)
        if len(sorted_times) > 200:
            # Downsample: keep first, last, and evenly spaced middle
            step = len(sorted_times) / 200
            sorted_times = [sorted_times[int(i * step)] for i in range(200)]

        for ts in sorted_times:
            snap = AnalysisService.query_book_depth(result, ts, n_levels=5)
            if snap["bids"] or snap["asks"]:
                l2_snapshots.append(
                    L2Snapshot(
                        timestamp_ns=ts,
                        bids=[(int(p), int(q)) for p, q in snap["bids"]],
                        asks=[(int(p), int(q)) for p, q in snap["asks"]],
                    )
                )

        return RichAnalysisBundle(
            fills=fills,
            pnl_curve=pnl_curve,
            inventory_trajectory=inventory,
            adverse_selection_bps=adverse,
            adverse_selection_by_window=adverse_by_window,
            counterparty_breakdown=counterparties,
            order_lifecycle=lifecycle,
            mid_price_series=mid_price_series,
            l2_snapshots=l2_snapshots,
        )

    # ==================================================================
    # Chart methods — rich analysis  (Step 8)
    # ==================================================================

    @staticmethod
    def plot_pnl_curve(
        pnl_points: list[PnLPoint],
        title: str = "Mark-to-Market PnL",
    ) -> Figure:
        """Line chart of PnL over time with drawdown shading.

        Accepts ``list[PnLPoint]`` (serialised data), not raw
        SimulationOutput — container-safe.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        if not pnl_points:
            ax.text(0.5, 0.5, "No PnL data available", ha="center", va="center")
            return fig

        times = [p.timestamp_ns for p in pnl_points]
        pnl_vals = [p.mark_to_market_pnl for p in pnl_points]

        ax.plot(times, pnl_vals, linewidth=1.5, label="PnL")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)

        # Shade drawdown regions
        pnl_arr = np.array(pnl_vals)
        running_max = np.maximum.accumulate(pnl_arr)
        drawdown = running_max - pnl_arr
        dd_mask = drawdown > 0
        ax.fill_between(times, pnl_vals, running_max.tolist(), where=dd_mask.tolist(), alpha=0.2, color="red", label="Drawdown")

        ax.set_xlabel("Time (ns since midnight)")
        ax.set_ylabel("PnL (cents)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_inventory(
        inventory_points: list[InventoryPoint],
        title: str = "Inventory Trajectory",
    ) -> Figure:
        """Step chart of inventory position over time.

        Accepts ``list[InventoryPoint]`` — container-safe.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        if not inventory_points:
            ax.text(0.5, 0.5, "No inventory data available", ha="center", va="center")
            return fig

        times = [p.timestamp_ns for p in inventory_points]
        positions = [p.position for p in inventory_points]

        ax.step(times, positions, where="post", linewidth=1.5, label="Position")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)

        # Color long/short regions
        pos_arr = np.array(positions)
        ax.fill_between(times, positions, 0, where=(pos_arr > 0).tolist(), step="post", alpha=0.2, color="green", label="Long")
        ax.fill_between(times, positions, 0, where=(pos_arr < 0).tolist(), step="post", alpha=0.2, color="red", label="Short")

        ax.set_xlabel("Time (ns since midnight)")
        ax.set_ylabel("Position (shares)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_fills_vs_mid(
        fill_records: list[FillRecord],
        title: str = "Fills vs Mid-Price",
    ) -> Figure:
        """Scatter of fills plotted against mid-price with buy/sell markers.

        Accepts ``list[FillRecord]`` — container-safe.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        if not fill_records:
            ax.text(0.5, 0.5, "No fill data available", ha="center", va="center")
            return fig

        # Plot mid-price line where available
        fills_with_mid = [f for f in fill_records if f.mid_at_fill is not None]
        if fills_with_mid:
            mid_times = [f.timestamp_ns for f in fills_with_mid]
            mid_prices: list[float] = [f.mid_at_fill for f in fills_with_mid if f.mid_at_fill is not None]
            ax.plot(mid_times, mid_prices, color="gray", linewidth=0.8, alpha=0.7, label="Mid at fill")

        # Scatter buys and sells
        buys = [f for f in fill_records if f.side == "BUY"]
        sells = [f for f in fill_records if f.side == "SELL"]

        if buys:
            ax.scatter(
                [f.timestamp_ns for f in buys],
                [f.price for f in buys],
                marker="^",
                color="green",
                s=40,
                alpha=0.8,
                label="Buy",
                zorder=5,
            )
        if sells:
            ax.scatter(
                [f.timestamp_ns for f in sells],
                [f.price for f in sells],
                marker="v",
                color="red",
                s=40,
                alpha=0.8,
                label="Sell",
                zorder=5,
            )

        ax.set_xlabel("Time (ns since midnight)")
        ax.set_ylabel("Price (cents)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

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
        comparison: ComparisonResult,
        strategy_output: SimulationOutput | None = None,
        duration_seconds: float = 0.0,
        error: str | None = None,
    ) -> RunSummary:
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
                logger.debug("Price chart generation failed in generate_summary", exc_info=True)

            try:
                spread_fig = AnalysisService.plot_spread(strategy_output)
                spread_chart = AnalysisService.figure_to_base64(spread_fig)
            except Exception:
                logger.debug("Spread chart generation failed in generate_summary", exc_info=True)

        return RunSummary(
            comparison=comparison,
            price_chart=price_chart,
            spread_chart=spread_chart,
            duration_seconds=duration_seconds,
            error=error,
        )
