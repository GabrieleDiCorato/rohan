from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimum-data thresholds — below these we return None rather than computing
# a statistically meaningless number.
# ---------------------------------------------------------------------------
_MIN_RETURNS_FOR_VOL = 30  # need ≥30 return observations for annualised vol
_MIN_RETURNS_FOR_SHARPE = 30
_MIN_FILLS_FOR_VPIN = 20  # too few fills → VPIN is noise
_VPIN_N_BUCKETS = 50  # standard VPIN parameterisation
_RESILIENCE_SHOCK_K = 2.0  # spread > mean + k*std triggers a shock


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
    both bid and ask are present.  Metrics where one-sided states are
    informative (LOB imbalance, resilience) use the full L1 timeline.
    """

    # ==================================================================
    # Market-wide metrics
    # ==================================================================

    @staticmethod
    def compute_metrics(result: SimulationOutput) -> SimulationMetrics:
        """Compute market-wide metrics from L1 order-book snapshots and logs.

        Returns a :class:`SimulationMetrics` with all computable fields
        populated.  Fields remain ``None`` when there is insufficient data
        to compute a meaningful value (e.g. too few observations for
        annualised volatility).
        """
        l1 = result.get_order_book_l1()

        if l1.empty:
            return SimulationMetrics()

        l1 = l1.copy()

        # ----- Two-sided subset (both bid and ask present) -----
        # Only these rows have a meaningful mid-price.  Rows where one
        # side is NaN represent genuinely empty book states (cannot trade).
        two_sided = l1.dropna(subset=["bid_price", "ask_price"]).copy()

        if two_sided.empty:
            return SimulationMetrics()

        two_sided["mid_price"] = (two_sided["bid_price"] + two_sided["ask_price"]) / 2
        two_sided["spread"] = two_sided["ask_price"] - two_sided["bid_price"]

        # --- Returns (from consecutive two-sided snapshots) ---
        # pct_change() on non-contiguous timestamps is correct: each return
        # measures the relative price change between the two nearest moments
        # when the book was tradeable.  We do NOT fillna(0) — a NaN first
        # return is simply dropped.
        two_sided["returns"] = two_sided["mid_price"].pct_change()
        returns = pd.Series(two_sided["returns"].dropna())

        # ----- Volatility (annualised) -----
        volatility = AnalysisService._annualised_volatility(returns, pd.Series(two_sided["time"]))

        # ----- Quoted spread -----
        mean_spread = float(two_sided["spread"].mean())

        # ----- Liquidity (use all rows where each side is present) -----
        # bid_qty is meaningful even when ask is NaN (and vice versa).
        avg_bid_liq = float(l1["bid_qty"].dropna().mean()) if bool(l1["bid_qty"].notna().any()) else None
        avg_ask_liq = float(l1["ask_qty"].dropna().mean()) if bool(l1["ask_qty"].notna().any()) else None

        # ----- Effective spread & traded volume (from fill logs) -----
        effective_spread, traded_volume = AnalysisService._effective_spread_and_volume(result, two_sided)

        # ----- LOB Imbalance -----
        lob_imb_mean, lob_imb_std = AnalysisService._lob_imbalance(l1)

        # ----- VPIN -----
        vpin = AnalysisService._vpin(result, two_sided)

        # ----- Market Resilience -----
        resilience_ns = AnalysisService._market_resilience(two_sided)

        # ----- Market-wide OTT ratio -----
        market_ott = AnalysisService._market_ott_ratio(result)

        return SimulationMetrics(
            volatility=volatility,
            mean_spread=mean_spread if not pd.isna(mean_spread) else None,
            effective_spread=effective_spread,
            avg_bid_liquidity=avg_bid_liq,
            avg_ask_liquidity=avg_ask_liq,
            traded_volume=traded_volume,
            lob_imbalance_mean=lob_imb_mean,
            lob_imbalance_std=lob_imb_std,
            vpin=vpin,
            resilience_mean_ns=resilience_ns,
            market_ott_ratio=market_ott,
        )

    # ==================================================================
    # Per-agent metrics
    # ==================================================================

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
            initial_cash: The agent's starting cash (integer cents).
                Required for correct PnL computation.
        """
        if not hasattr(result, "end_state"):
            return AgentMetrics(agent_id=agent_id, initial_cash=initial_cash)

        agents = {a.id: a for a in result.end_state["agents"]}  # type: ignore[not-subscriptable]
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

        # --- Mark-to-market using last two-sided mid ---
        l1 = result.get_order_book_l1()
        two_sided = l1.dropna(subset=["bid_price", "ask_price"]).copy() if not l1.empty else pd.DataFrame()
        last_mid = 0.0
        if not two_sided.empty:
            last_row = two_sided.iloc[-1]
            last_mid = float((last_row["bid_price"] + last_row["ask_price"]) / 2)

        ending_value = ending_cash + inventory * last_mid
        total_pnl = ending_value - initial_cash

        # --- Execution stats from raw agent log ---
        log: list = getattr(agent, "log", [])

        # Agent log entries are 3-tuples: (timestamp_ns, event_type_str, payload)
        fills: list[tuple] = []
        submissions: list[tuple] = []
        for entry in log:
            if isinstance(entry, tuple) and len(entry) > 1:
                etype = entry[1]
                if etype == "ORDER_EXECUTED":
                    fills.append(entry)
                elif etype == "ORDER_SUBMITTED":
                    submissions.append(entry)

        trade_count = len(fills)
        n_submitted = len(submissions)
        fill_rate = trade_count / n_submitted if n_submitted else None
        otr = n_submitted / trade_count if trade_count else None

        # --- Risk metrics from mark-to-market PnL curve ---
        sharpe_ratio = None
        max_drawdown = None
        inventory_std = None

        if fills and not two_sided.empty:
            sharpe_ratio, max_drawdown, inventory_std = AnalysisService._agent_risk_metrics(fills, two_sided, initial_cash)

        return AgentMetrics(
            agent_id=agent_id,
            initial_cash=initial_cash,
            ending_cash=ending_cash,
            total_pnl=total_pnl,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            inventory_std=inventory_std,
            trade_count=trade_count,
            fill_rate=fill_rate,
            order_to_trade_ratio=otr,
            end_inventory=inventory,
        )

    # ==================================================================
    # Private: core metric computations
    # ==================================================================

    @staticmethod
    def _annualised_volatility(returns: pd.Series, time_col: pd.Series) -> float | None:
        """Annualised realised volatility from a clean returns series.

        Derives the annualisation factor from the actual observation
        frequency (median inter-observation interval) rather than
        assuming fixed-frequency bars.  Returns None if there are
        fewer than ``_MIN_RETURNS_FOR_VOL`` observations.
        """
        clean_returns = returns.dropna()
        if len(clean_returns) < _MIN_RETURNS_FOR_VOL:
            return None

        sigma = float(clean_returns.std())
        if pd.isna(sigma) or sigma == 0.0:
            return None

        # Annualisation: infer observation frequency from timestamps
        diffs = time_col.diff().dropna()
        if diffs.empty:
            return None
        median_dt_ns = float(diffs.median())
        if median_dt_ns <= 0:
            return None

        trading_day_ns = 6.5 * 3600 * 1e9  # 6.5 trading hours
        obs_per_day = trading_day_ns / median_dt_ns
        obs_per_year = obs_per_day * 252

        vol = sigma * np.sqrt(obs_per_year)
        return float(vol) if not pd.isna(vol) else None

    @staticmethod
    def _effective_spread_and_volume(
        result: SimulationOutput,
        two_sided: pd.DataFrame,
    ) -> tuple[float | None, int | None]:
        """Compute effective spread and total traded volume from fill events.

        Effective spread = 2 × |P_fill − P_mid_nearest| averaged over all
        fills.  This captures the *actual* cost of trading including market
        impact, which the quoted spread does not.

        We match each fill to the nearest two-sided L1 mid-price by
        timestamp.  Fills during periods with no two-sided book are excluded
        (no meaningful mid to compare against).
        """
        try:
            logs = result.get_logs_df()
        except Exception:
            return None, None

        if logs.empty:
            return None, None

        # Filter fill events — ABIDES logs ORDER_EXECUTED with fill_price
        fill_mask = logs["EventType"] == "ORDER_EXECUTED"
        fills = logs[fill_mask].copy()

        if fills.empty:
            return None, 0

        # Total traded volume (each fill row's quantity = fill qty for that
        # execution tranche; aggregate across all agents)
        volume: int | None = None
        if "quantity" in fills.columns:
            qty_series = pd.Series(pd.to_numeric(fills["quantity"], errors="coerce")).dropna()
            if not qty_series.empty:
                volume = int(qty_series.sum())

        # Effective spread needs fill_price and a timestamp to match to mid
        if "fill_price" not in fills.columns or two_sided.empty:
            return None, volume

        fills = fills.dropna(subset=["fill_price"])  # type: ignore[call-overload]
        if fills.empty or "EventTime" not in fills.columns:
            return None, volume

        # Build a mid-price lookup from the two-sided L1 (use ns-since-midnight 'time')
        mid_series = two_sided.set_index("time")["mid_price"].sort_index()

        # ABIDES EventTime is ns-since-epoch; convert to ns-since-midnight
        # to match the L1 'time' column.
        from abides_markets.order_book import ns_date

        eff_spreads = []
        for _, fill_row in fills.iterrows():
            fp_raw = pd.to_numeric(fill_row.get("fill_price"), errors="coerce")
            try:
                fp = float(fp_raw)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if np.isnan(fp):
                continue

            event_time = fill_row.get("EventTime")
            if event_time is None or pd.isna(event_time):
                continue

            event_time_int = int(event_time)
            try:
                t_midnight = event_time_int - ns_date(event_time_int)
            except Exception:
                continue

            # Find nearest mid-price by timestamp
            idx = mid_series.index.searchsorted(t_midnight)
            # Pick the closer of the two neighbours
            candidates = []
            if idx > 0:
                candidates.append(mid_series.index[idx - 1])
            if idx < len(mid_series):
                candidates.append(mid_series.index[idx])
            if not candidates:
                continue

            nearest_t = min(candidates, key=lambda t: abs(t - t_midnight))
            mid_at_fill = mid_series.loc[nearest_t]
            # .loc may return a Series if the index has duplicates — take first
            if isinstance(mid_at_fill, pd.Series):
                mid_at_fill = mid_at_fill.iloc[0]
            if pd.isna(mid_at_fill):
                continue

            # Effective spread = 2 × |fill_price − mid|
            eff_spreads.append(2.0 * abs(fp - float(mid_at_fill)))  # type: ignore[arg-type]

        effective_spread = float(np.mean(eff_spreads)) if eff_spreads else None
        return effective_spread, volume

    @staticmethod
    def _lob_imbalance(l1: pd.DataFrame) -> tuple[float | None, float | None]:
        """L1 order-book imbalance: (Q_bid − Q_ask) / (Q_bid + Q_ask).

        Computed on rows where **both** bid_qty and ask_qty are present.
        When one side is NaN the book is empty there — imbalance is
        technically ±1, but including these would conflate illiquidity events
        with directional pressure.  We restrict to two-sided rows to measure
        genuine order-flow imbalance.

        Returns (mean, std) of the imbalance time series.
        """
        mask = l1["bid_qty"].notna() & l1["ask_qty"].notna()
        both = l1[mask]

        if both.empty:
            return None, None

        total_qty = both["bid_qty"] + both["ask_qty"]
        # Avoid division by zero when both sides have qty=0 simultaneously
        valid = total_qty > 0
        if not bool(valid.any()):
            return None, None

        imb = (both.loc[valid, "bid_qty"] - both.loc[valid, "ask_qty"]) / total_qty[valid]
        return float(imb.mean()), float(imb.std())

    @staticmethod
    def _vpin(
        result: SimulationOutput,
        two_sided: pd.DataFrame,
    ) -> float | None:
        """Volume-Synchronised Probability of Informed Trading.

        Implementation follows Easley, López de Prado & O'Hara (2012):
        1. Classify fills as buy- or sell-initiated via the Lee-Ready test
           (fill_price vs nearest mid → direction; at the mid use tick rule).
        2. Bucket fills into equal-volume bars.
        3. VPIN = mean(|V_buy − V_sell| / V_bucket) over all buckets.

        Returns None if there are fewer than ``_MIN_FILLS_FOR_VPIN`` fills.
        """
        try:
            logs = result.get_logs_df()
        except Exception:
            return None

        if logs.empty or two_sided.empty:
            return None

        fills = logs[logs["EventType"] == "ORDER_EXECUTED"].copy()
        if len(fills) < _MIN_FILLS_FOR_VPIN:
            return None

        required = {"fill_price", "quantity", "EventTime"}
        if not required.issubset(fills.columns):
            return None

        fills = fills.dropna(subset=["fill_price", "quantity", "EventTime"])  # type: ignore[call-overload]
        fills["fill_price"] = pd.to_numeric(fills["fill_price"], errors="coerce")
        fills["quantity"] = pd.to_numeric(fills["quantity"], errors="coerce")
        fills = fills.dropna(subset=["fill_price", "quantity"])  # type: ignore[call-overload]

        if fills.empty or fills["quantity"].sum() <= 0:
            return None

        # Build mid-price lookup (time → mid) for Lee-Ready classification
        from abides_markets.order_book import ns_date

        mid_lookup = two_sided.set_index("time")["mid_price"].sort_index()

        # Classify each fill
        signed_qty: list[float] = []  # +qty = buy, -qty = sell
        prev_price: float | None = None

        for _, row in fills.sort_values("EventTime").iterrows():
            fp = float(row["fill_price"])
            qty = float(row["quantity"])

            event_time_int = int(row["EventTime"])
            try:
                t_midnight = event_time_int - ns_date(event_time_int)
            except Exception:
                prev_price = fp
                continue

            # Find nearest mid
            idx = mid_lookup.index.searchsorted(t_midnight)
            candidates = []
            if idx > 0:
                candidates.append(mid_lookup.index[idx - 1])
            if idx < len(mid_lookup):
                candidates.append(mid_lookup.index[idx])
            if not candidates:
                prev_price = fp
                continue

            nearest_t = min(candidates, key=lambda t: abs(t - t_midnight))
            mid = float(mid_lookup.loc[nearest_t])

            # Lee-Ready classification
            if fp > mid:
                direction = 1.0  # buy-initiated
            elif fp < mid:
                direction = -1.0  # sell-initiated
            elif prev_price is not None:
                # At the mid: use tick rule (compare to previous trade price)
                if fp > prev_price:
                    direction = 1.0
                elif fp < prev_price:
                    direction = -1.0
                else:
                    direction = 0.0  # indeterminate — split evenly
            else:
                direction = 0.0

            signed_qty.append(direction * qty)
            prev_price = fp

        if not signed_qty:
            return None

        total_volume = sum(abs(q) for q in signed_qty)
        if total_volume <= 0:
            return None

        # Bucket into equal-volume bars
        n_buckets = min(_VPIN_N_BUCKETS, max(1, len(signed_qty) // 2))
        bucket_size = total_volume / n_buckets

        order_imbalances: list[float] = []
        bucket_buy = 0.0
        bucket_sell = 0.0
        bucket_vol = 0.0

        for sq in signed_qty:
            remaining = abs(sq)
            buy_part = max(sq, 0.0)
            sell_part = max(-sq, 0.0)

            while remaining > 0:
                space = bucket_size - bucket_vol
                fill = min(remaining, space)

                # Proportion of this fill that is buy vs sell
                ratio = fill / abs(sq) if abs(sq) > 0 else 0.0
                bucket_buy += buy_part * ratio
                bucket_sell += sell_part * ratio
                bucket_vol += fill
                remaining -= fill

                if bucket_vol >= bucket_size - 1e-9:
                    # Bucket complete
                    if bucket_size > 0:
                        oi = abs(bucket_buy - bucket_sell) / bucket_size
                        order_imbalances.append(oi)
                    bucket_buy = 0.0
                    bucket_sell = 0.0
                    bucket_vol = 0.0

        if not order_imbalances:
            return None

        vpin_value = float(np.mean(order_imbalances))
        return vpin_value if not pd.isna(vpin_value) else None

    @staticmethod
    def _market_resilience(two_sided: pd.DataFrame) -> float | None:
        """Mean spread recovery time after shock events.

        A *shock* is defined as a spread exceeding
        ``mean + k × std`` of a rolling window.  Recovery is the time (ns)
        until the spread returns to ``mean + 1 × std``.

        Returns None if no shocks are detected or insufficient data.
        """
        if len(two_sided) < 50:
            return None

        spread = two_sided["spread"] if "spread" in two_sided.columns else (two_sided["ask_price"] - two_sided["bid_price"])
        time_col = two_sided["time"]

        # Rolling statistics — window is 10% of data or 100 ticks, whichever smaller
        window = min(100, max(10, len(spread) // 10))
        roll_mean = spread.rolling(window, min_periods=window // 2).mean()
        roll_std = spread.rolling(window, min_periods=window // 2).std()

        # Identify shock indices: spread > mean + k*std
        threshold = roll_mean + _RESILIENCE_SHOCK_K * roll_std
        recovery_threshold = roll_mean + 1.0 * roll_std

        shock_mask = spread > threshold
        shock_indices = two_sided.index[shock_mask]

        if shock_indices.empty:
            return None

        recovery_times: list[float] = []
        last_shock_idx = -window  # debounce: don't count overlapping shocks

        for shock_idx in shock_indices:
            _loc = two_sided.index.get_loc(shock_idx)
            if not isinstance(_loc, int):
                continue  # non-unique index position (duplicate label) — skip
            pos: int = _loc
            if pos - last_shock_idx < window // 2:
                continue  # too close to previous shock, skip
            last_shock_idx = pos

            shock_time = time_col.iloc[pos]
            recovery_level = recovery_threshold.iloc[pos]  # type: ignore[union-attr]

            # Scan forward for recovery
            for j in range(pos + 1, len(spread)):
                if spread.iloc[j] <= recovery_level:
                    recovery_time = time_col.iloc[j] - shock_time
                    recovery_times.append(float(recovery_time))
                    break
            # If no recovery found, this shock is excluded (market didn't recover
            # within the simulation window — including it would bias the metric).

        if not recovery_times:
            return None

        return float(np.mean(recovery_times))

    @staticmethod
    def _market_ott_ratio(result: SimulationOutput) -> float | None:
        """Market-wide order-to-trade ratio: total submissions / total fills.

        Measures market-level quoting activity relative to actual execution.
        MiFID II RTS 9 caps this at 4:1 for certain instruments.
        """
        try:
            logs = result.get_logs_df()
        except Exception:
            return None

        if logs.empty:
            return None

        n_submitted = int((logs["EventType"] == "ORDER_SUBMITTED").sum())
        n_executed = int((logs["EventType"] == "ORDER_EXECUTED").sum())

        if n_executed == 0:
            return None

        return float(n_submitted) / float(n_executed)

    @staticmethod
    def _agent_risk_metrics(
        fills: list[tuple],
        two_sided: pd.DataFrame,
        initial_cash: int,
    ) -> tuple[float | None, float | None, float | None]:
        """Compute Sharpe ratio, max drawdown, and inventory std for an agent.

        Reconstructs a mark-to-market PnL curve from the agent's fill events
        and the two-sided L1 mid-price timeline.

        Args:
            fills: List of raw log tuples ``(timestamp_ns, "ORDER_EXECUTED", payload)``.
            two_sided: L1 DataFrame filtered to rows with both bid and ask.
            initial_cash: Agent's starting cash in cents.

        Returns:
            (sharpe_ratio, max_drawdown_cents, inventory_std)
        """
        from abides_markets.order_book import ns_date

        # --- Extract fill events into a structured list ---
        fill_records: list[dict] = []
        for entry in fills:
            ts_ns = entry[0]  # nanosecond timestamp (epoch)
            payload = entry[2]  # order dict from to_dict()
            if not isinstance(payload, dict):
                continue

            fp = payload.get("fill_price")
            qty = payload.get("quantity")
            side = payload.get("side")
            if fp is None or qty is None or side is None:
                continue

            # Determine sign: buy (+inventory) vs sell (-inventory)
            # In ABIDES, Side.BID = buy, Side.ASK = sell
            side_str = str(side)
            if "BID" in side_str.upper():
                signed_qty = int(qty)
                cash_delta = -int(fp) * int(qty)  # pay cash to buy
            elif "ASK" in side_str.upper():
                signed_qty = -int(qty)
                cash_delta = int(fp) * int(qty)  # receive cash from sell
            else:
                continue

            try:
                t_midnight = int(ts_ns) - ns_date(int(ts_ns))
            except Exception:
                continue

            fill_records.append(
                {
                    "time": t_midnight,
                    "signed_qty": signed_qty,
                    "cash_delta": cash_delta,
                }
            )

        if not fill_records:
            return None, None, None

        fill_records.sort(key=lambda r: r["time"])

        # --- Build inventory & cash trajectory at each fill ---
        cash = initial_cash
        inv = 0
        inventory_series: list[int] = [inv]  # inventory before any fill

        # Also build PnL at each L1 two-sided snapshot for Sharpe estimation
        mid_times = two_sided["time"].values
        mid_prices = ((two_sided["bid_price"] + two_sided["ask_price"]) / 2).values

        # Merge fill and mid timelines for a full PnL curve
        # Strategy: compute PnL at each mid-price observation using the
        # most recent cash & inventory state (updated by fills).
        fill_idx = 0
        pnl_curve: list[float] = []

        for i in range(len(mid_times)):
            # Process any fills that occurred before this mid snapshot
            while fill_idx < len(fill_records) and fill_records[fill_idx]["time"] <= mid_times[i]:
                cash += fill_records[fill_idx]["cash_delta"]
                inv += fill_records[fill_idx]["signed_qty"]
                inventory_series.append(inv)
                fill_idx += 1

            # Mark-to-market: PnL = cash + inventory × mid − initial_cash
            mtm = cash + inv * mid_prices[i]
            pnl_curve.append(float(mtm - initial_cash))

        # Process remaining fills after all mid snapshots
        while fill_idx < len(fill_records):
            cash += fill_records[fill_idx]["cash_delta"]
            inv += fill_records[fill_idx]["signed_qty"]
            inventory_series.append(inv)
            fill_idx += 1

        # --- Sharpe ratio ---
        sharpe = None
        if len(pnl_curve) >= _MIN_RETURNS_FOR_SHARPE:
            pnl_arr = np.array(pnl_curve, dtype=float)
            # Period-over-period PnL change as returns
            pnl_returns = np.diff(pnl_arr)
            if len(pnl_returns) > 1:
                mu = float(np.mean(pnl_returns))
                sigma = float(np.std(pnl_returns, ddof=1))
                if sigma > 0:
                    # Annualise using the same frequency approach as volatility
                    diffs = np.diff(mid_times.astype(float))
                    median_dt = float(np.median(diffs)) if len(diffs) > 0 else 0
                    if median_dt > 0:
                        trading_day_ns = 6.5 * 3600 * 1e9
                        obs_per_year = (trading_day_ns / median_dt) * 252
                        sharpe = (mu / sigma) * np.sqrt(obs_per_year)
                        sharpe = float(sharpe) if not pd.isna(sharpe) else None

        # --- Max drawdown ---
        max_dd = None
        if pnl_curve:
            pnl_arr = np.array(pnl_curve, dtype=float)
            running_max = np.maximum.accumulate(pnl_arr)
            drawdown = running_max - pnl_arr
            max_dd_val = float(np.max(drawdown))
            max_dd = max_dd_val if max_dd_val > 0 else None

        # --- Inventory std ---
        inv_std = None
        if len(inventory_series) > 1:
            inv_std = float(np.std(inventory_series, ddof=1))

        return sharpe, max_dd, inv_std

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
