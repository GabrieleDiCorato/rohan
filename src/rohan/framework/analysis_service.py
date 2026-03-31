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

    @staticmethod
    def _get_agent_fills(
        result: SimulationOutput,
        agent_id: int,
    ) -> tuple[object, list[dict]] | None:
        """Extract an agent object and its parsed fill records.

        Returns ``(agent, parsed_fills)`` or ``None`` when the agent cannot
        be found or has no fills.  The *agent* object is the raw ABIDES
        agent from ``end_state["agents"]``.
        """
        if not hasattr(result, "end_state"):
            return None

        agents = {a.id: a for a in result.end_state["agents"]}  # type: ignore[union-attr]
        if agent_id not in agents:
            return None

        agent = agents[agent_id]
        log: list = getattr(agent, "log", [])
        raw_fills = [e for e in log if isinstance(e, tuple) and len(e) > 1 and e[1] == "ORDER_EXECUTED"]
        parsed = AnalysisService._parse_fills(raw_fills)
        if not parsed:
            return None

        return agent, parsed

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
        two_sided = AnalysisService._get_two_sided_l1(result)

        # Market availability: fraction of snapshots that are tradeable
        pct_two_sided = float(len(two_sided) / len(l1))

        if two_sided.empty:
            return SimulationMetrics(pct_time_two_sided=0.0)

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
            pct_time_two_sided=pct_two_sided,
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
        two_sided = AnalysisService._get_two_sided_l1(result)
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

        # Annualisation: infer observation frequency from timestamps.
        # Filter to positive diffs only — zero-interval duplicates from
        # multiple book state changes within the same nanosecond carry no
        # timing information and would collapse the median to zero.
        diffs = time_col.diff().dropna()
        diffs = pd.Series(diffs[diffs > 0])
        if diffs.empty:
            return None
        median_dt_ns = float(diffs.median())
        if median_dt_ns <= 0:
            return None

        # Minimum time-span guard: with less than 60 seconds of distinct
        # timestamps the annualisation factor becomes astronomical and
        # the resulting volatility number is statistically meaningless.
        min_span_ns = 60 * 1e9  # 60 seconds
        total_span_ns = float(diffs.sum())
        if total_span_ns < min_span_ns:
            logger.debug(
                "Skipping volatility: total time span %.1f s < 60 s minimum",
                total_span_ns / 1e9,
            )
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

        # Build a mid-price lookup from the two-sided L1
        mid_lookup = AnalysisService._build_mid_lookup(two_sided)

        eff_spreads = []
        for _, fill_row in fills.iterrows():
            fp_raw = pd.to_numeric(fill_row.get("fill_price"), errors="coerce")  # type: ignore[arg-type]
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

            mid_at_fill = AnalysisService._nearest_mid(mid_lookup, t_midnight)
            if mid_at_fill is None:
                continue

            # Effective spread = 2 × |fill_price − mid|
            eff_spreads.append(2.0 * abs(fp - mid_at_fill))

        if not eff_spreads:
            return None, volume
        effective_spread = float(np.nanmean(eff_spreads))
        # Guard against all-NaN effective spreads (entirely one-sided book)
        if np.isnan(effective_spread):
            return None, volume
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
        mid_lookup = AnalysisService._build_mid_lookup(two_sided)

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

            mid = AnalysisService._nearest_mid(mid_lookup, t_midnight)
            if mid is None:
                prev_price = fp
                continue

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

        # Guard against zero bucket size — would cause an infinite loop
        # when all fill quantities are zero.
        if bucket_size <= 0:
            return None

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

    # ==================================================================
    # Private: fill parsing (shared by risk metrics & rich analysis)
    # ==================================================================

    @staticmethod
    def _parse_fills(raw_fills: list[tuple]) -> list[dict]:
        """Parse raw agent log fill tuples into structured dicts.

        Each raw entry is ``(timestamp_ns_epoch, "ORDER_EXECUTED", payload_dict)``.
        Returns a sorted list of dicts with keys:
        ``time`` (ns-since-midnight), ``signed_qty``, ``cash_delta``,
        ``price`` (int cents), ``qty`` (unsigned int), ``side`` ("BUY"/"SELL"),
        ``order_id`` (int or None).
        """
        records: list[dict] = []
        for entry in raw_fills:
            ts_ns = entry[0]
            payload = entry[2]
            if not isinstance(payload, dict):
                continue

            fp = payload.get("fill_price")
            qty = payload.get("quantity")
            side = payload.get("side")
            if fp is None or qty is None or side is None:
                continue

            side_str = str(side)
            if "BID" in side_str.upper():
                signed_qty = int(qty)
                cash_delta = -int(fp) * int(qty)
                side_label = "BUY"
            elif "ASK" in side_str.upper():
                signed_qty = -int(qty)
                cash_delta = int(fp) * int(qty)
                side_label = "SELL"
            else:
                continue

            try:
                t_midnight = int(ts_ns) - ns_date(int(ts_ns))
            except Exception:
                continue

            records.append(
                {
                    "time": t_midnight,
                    "signed_qty": signed_qty,
                    "cash_delta": cash_delta,
                    "price": int(fp),
                    "qty": int(qty),
                    "side": side_label,
                    "order_id": payload.get("order_id"),
                }
            )

        records.sort(key=lambda r: r["time"])
        return records

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

    @staticmethod
    def _agent_risk_metrics(
        fills: list[tuple],
        two_sided: pd.DataFrame,
        initial_cash: int,
        *,
        parsed_fills: list[dict] | None = None,
    ) -> tuple[float | None, float | None, float | None]:
        """Compute Sharpe ratio, max drawdown, and inventory std for an agent.

        Reconstructs a mark-to-market PnL curve from the agent's fill events
        and the two-sided L1 mid-price timeline.

        Args:
            fills: List of raw log tuples ``(timestamp_ns, "ORDER_EXECUTED", payload)``.
            two_sided: L1 DataFrame filtered to rows with both bid and ask.
            initial_cash: Agent's starting cash in cents.
            parsed_fills: Optional pre-parsed fills from ``_parse_fills``.
                When provided, *fills* is ignored.

        Returns:
            (sharpe_ratio, max_drawdown_cents, inventory_std)
        """
        fill_records = parsed_fills if parsed_fills is not None else AnalysisService._parse_fills(fills)

        if not fill_records:
            return None, None, None

        # --- Build inventory & cash trajectory at each fill ---
        cash = initial_cash
        inv = 0
        inventory_series: list[int] = [inv]

        mid_times = two_sided["time"].values
        mid_prices = ((two_sided["bid_price"] + two_sided["ask_price"]) / 2).values

        fill_idx = 0
        pnl_curve: list[float] = []

        for i in range(len(mid_times)):
            while fill_idx < len(fill_records) and fill_records[fill_idx]["time"] <= mid_times[i]:
                cash += fill_records[fill_idx]["cash_delta"]
                inv += fill_records[fill_idx]["signed_qty"]
                inventory_series.append(inv)
                fill_idx += 1

            mtm = cash + inv * mid_prices[i]
            pnl_curve.append(float(mtm - initial_cash))

        while fill_idx < len(fill_records):
            cash += fill_records[fill_idx]["cash_delta"]
            inv += fill_records[fill_idx]["signed_qty"]
            inventory_series.append(inv)
            fill_idx += 1

        # --- Sharpe ratio ---
        sharpe = None
        if len(pnl_curve) >= _MIN_RETURNS_FOR_SHARPE:
            pnl_arr = np.array(pnl_curve, dtype=float)
            pnl_returns = np.diff(pnl_arr)
            if len(pnl_returns) > 1:
                mu = float(np.mean(pnl_returns))
                sigma = float(np.std(pnl_returns, ddof=1))
                if sigma > 0:
                    diffs = np.diff(np.asarray(mid_times, dtype=float))
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

    # ==================================================================
    # Rich analysis methods  (Step 8)
    # ==================================================================

    @staticmethod
    def get_fill_analysis(
        result: SimulationOutput,
        agent_id: int,
    ) -> list[FillRecord]:
        """Per-fill execution quality with slippage and counterparty info.

        For each fill, joins to the nearest L1 mid-price and computes
        signed slippage in basis points.  Counterparty type is resolved
        from ``end_state["agents"]`` when available.

        Returns an empty list if the agent has no fills.
        """
        agent_fills = AnalysisService._get_agent_fills(result, agent_id)
        if agent_fills is None:
            return []
        _agent, parsed = agent_fills

        # Build mid lookup
        two_sided = AnalysisService._get_two_sided_l1(result)
        mid_lookup = AnalysisService._build_mid_lookup(two_sided) if not two_sided.empty else pd.Series(dtype=float)

        # Build counterparty map — match fills across all agents
        # Both sides of a trade see ORDER_EXECUTED at (same timestamp, same fill_price, same qty, opposite side)
        counterparty_map = AnalysisService._build_counterparty_map(result, agent_id)

        records: list[FillRecord] = []
        for fill in parsed:
            mid = AnalysisService._nearest_mid(mid_lookup, fill["time"])
            slippage: float | None = None
            if mid is not None and mid > 0:
                side_sign = 1.0 if fill["side"] == "BUY" else -1.0
                slippage = side_sign * (fill["price"] - mid) / mid * 10_000

            cp_type = counterparty_map.get((fill["time"], fill["price"], fill["qty"]))

            records.append(
                FillRecord(
                    timestamp_ns=fill["time"],
                    side=fill["side"],
                    price=fill["price"],
                    qty=fill["qty"],
                    mid_at_fill=mid,
                    slippage_bps=slippage,
                    counterparty_type=cp_type,
                )
            )
        return records

    @staticmethod
    def _build_counterparty_map(
        result: SimulationOutput,
        agent_id: int,
    ) -> dict[tuple[int, int, int], str | None]:
        """Build a mapping from (time, price, qty) → counterparty agent type.

        Scans all agents' ORDER_EXECUTED events for fills that match the
        strategic agent's fills on the opposite side.
        """
        if not hasattr(result, "end_state"):
            return {}

        agents_list = result.end_state["agents"]  # type: ignore[union-attr]
        agent_type_map: dict[int, str] = {a.id: getattr(a, "type", type(a).__name__) for a in agents_list}

        # Get strategic agent fills via helper
        agent_fills = AnalysisService._get_agent_fills(result, agent_id)
        if agent_fills is None:
            return {}
        _agent, strategic_parsed = agent_fills

        # Collect fills from all OTHER agents
        other_fills: list[tuple[int, int, int, str, str]] = []  # (time, price, qty, side, agent_type)
        for agent in agents_list:
            if agent.id == agent_id:
                continue
            a_log: list = getattr(agent, "log", [])
            a_raw = [e for e in a_log if isinstance(e, tuple) and len(e) > 1 and e[1] == "ORDER_EXECUTED"]
            a_parsed = AnalysisService._parse_fills(a_raw)
            a_type = agent_type_map.get(agent.id, "Unknown")
            for f in a_parsed:
                other_fills.append((f["time"], f["price"], f["qty"], f["side"], a_type))

        # Build lookup: for each strategic fill, find a matching counterparty fill
        # (same time, same price, same qty, opposite side)
        opposite = {"BUY": "SELL", "SELL": "BUY"}
        other_index: dict[tuple[int, int, int, str], str] = {}
        for t, p, q, s, at in other_fills:
            other_index[(t, p, q, s)] = at

        cp_map: dict[tuple[int, int, int], str | None] = {}
        for f in strategic_parsed:
            opp_side = opposite.get(f["side"], "")
            key = (f["time"], f["price"], f["qty"], opp_side)
            cp_map[(f["time"], f["price"], f["qty"])] = other_index.get(key)

        return cp_map

    @staticmethod
    def get_pnl_curve(
        result: SimulationOutput,
        agent_id: int,
        initial_cash: int = 0,
    ) -> list[PnLPoint]:
        """Mark-to-market PnL curve at each L1 observation.

        Merges agent fill events with the two-sided L1 mid-price timeline
        to produce a PnL observation at every mid-price tick.

        Returns an empty list when there are no fills or no two-sided book.
        """
        agent_fills = AnalysisService._get_agent_fills(result, agent_id)
        if agent_fills is None:
            return []
        _agent, parsed = agent_fills

        two_sided = AnalysisService._get_two_sided_l1(result)

        if two_sided.empty:
            return []

        mid_times = two_sided["time"].values
        mid_prices = ((two_sided["bid_price"] + two_sided["ask_price"]) / 2).values

        cash = initial_cash
        inv = 0
        fill_idx = 0
        points: list[PnLPoint] = []

        for i in range(len(mid_times)):
            while fill_idx < len(parsed) and parsed[fill_idx]["time"] <= mid_times[i]:
                cash += parsed[fill_idx]["cash_delta"]
                inv += parsed[fill_idx]["signed_qty"]
                fill_idx += 1

            mtm = cash + inv * mid_prices[i]
            points.append(PnLPoint(timestamp_ns=int(mid_times[i]), mark_to_market_pnl=float(mtm - initial_cash)))

        return points

    @staticmethod
    def get_inventory_trajectory(
        result: SimulationOutput,
        agent_id: int,
    ) -> list[InventoryPoint]:
        """Position trajectory at each fill event.

        Returns a list of ``InventoryPoint`` showing cumulative position
        after each fill.  Includes an initial point at timestamp 0 with
        position 0.
        """
        agent_fills = AnalysisService._get_agent_fills(result, agent_id)
        if agent_fills is None:
            return []
        _agent, parsed = agent_fills

        points: list[InventoryPoint] = [InventoryPoint(timestamp_ns=0, position=0)]
        cumulative = 0
        for fill in parsed:
            cumulative += fill["signed_qty"]
            points.append(InventoryPoint(timestamp_ns=fill["time"], position=cumulative))
        return points

    @staticmethod
    def get_adverse_selection(
        result: SimulationOutput,
        agent_id: int,
        window_ns: int = 500_000_000,
    ) -> float | None:
        """Average mid-price move against fill direction within a look-ahead window.

        For each fill, measures how the mid-price moves in the subsequent
        ``window_ns`` nanoseconds.  *Adverse* selection means the market
        moved against the agent after filling (buy → price drops,
        sell → price rises).

        Returns the average adverse move in basis points (positive = adverse).
        Returns ``None`` if there are no fills or insufficient L1 data.
        """
        agent_fills = AnalysisService._get_agent_fills(result, agent_id)
        if agent_fills is None:
            return None
        _agent, parsed = agent_fills

        two_sided = AnalysisService._get_two_sided_l1(result)
        if two_sided.empty:
            return None

        mid_lookup = AnalysisService._build_mid_lookup(two_sided)
        mid_times = mid_lookup.index.values
        mid_values = mid_lookup.values

        adverse_moves: list[float] = []
        for fill in parsed:
            mid_at_fill = AnalysisService._nearest_mid(mid_lookup, fill["time"])
            if mid_at_fill is None or mid_at_fill <= 0:
                continue

            # Find mid-price at fill_time + window_ns
            target_time = fill["time"] + window_ns
            idx = mid_times.searchsorted(target_time)
            # Pick nearest observation at or before target_time
            if idx >= len(mid_times):
                idx = len(mid_times) - 1
            if idx < 0:
                continue

            raw_mid = mid_values[idx]
            if pd.isna(raw_mid):
                continue
            future_mid = float(raw_mid)  # type: ignore[arg-type]

            # Adverse selection: for a BUY, adverse = price drops; for SELL, adverse = price rises
            # Convention: positive return = adverse
            side_sign = 1.0 if fill["side"] == "BUY" else -1.0
            move_bps = -side_sign * (future_mid - mid_at_fill) / mid_at_fill * 10_000
            adverse_moves.append(move_bps)

        if not adverse_moves:
            return None

        return float(np.mean(adverse_moves))

    @staticmethod
    def get_order_lifecycle(
        result: SimulationOutput,
        agent_id: int,
    ) -> list[OrderLifecycleRecord]:
        """Order lifecycle: submission → fill/cancel/resting for each order.

        Cross-references ``ORDER_SUBMITTED``, ``ORDER_EXECUTED``, and
        ``ORDER_CANCELLED`` events by order_id in the agent's raw log.
        """
        if not hasattr(result, "end_state"):
            return []

        agents = {a.id: a for a in result.end_state["agents"]}  # type: ignore[union-attr]
        if agent_id not in agents:
            return []

        agent = agents[agent_id]
        log: list = getattr(agent, "log", [])

        # Collect events keyed by order_id
        submissions: dict[int, dict] = {}
        executions: dict[int, list[dict]] = {}
        cancellations: dict[int, dict] = {}

        for entry in log:
            if not isinstance(entry, tuple) or len(entry) < 3:
                continue
            ts_ns = entry[0]
            etype = entry[1]
            payload = entry[2]
            if not isinstance(payload, dict):
                continue

            oid = payload.get("order_id")
            if oid is None:
                continue
            oid = int(oid)

            try:
                t_midnight = int(ts_ns) - ns_date(int(ts_ns))
            except Exception:
                continue

            if etype == "ORDER_SUBMITTED":
                submissions[oid] = {
                    "submitted_at_ns": t_midnight,
                    "qty": int(payload.get("quantity", 0)),
                }
            elif etype == "ORDER_EXECUTED":
                executions.setdefault(oid, []).append(
                    {
                        "time": t_midnight,
                        "qty": int(payload.get("quantity", 0)),
                    }
                )
            elif etype == "ORDER_CANCELLED":
                cancellations[oid] = {"time": t_midnight}

        # Build lifecycle records
        all_order_ids = set(submissions.keys()) | set(executions.keys()) | set(cancellations.keys())
        records: list[OrderLifecycleRecord] = []

        for oid in sorted(all_order_ids):
            sub = submissions.get(oid)
            submitted_at = sub["submitted_at_ns"] if sub else 0
            submitted_qty = sub["qty"] if sub else 0

            execs = executions.get(oid, [])
            filled_qty = sum(e["qty"] for e in execs)

            if oid in cancellations:
                status = "cancelled"
                resolve_time = cancellations[oid]["time"]
            elif filled_qty > 0:
                status = "filled"
                resolve_time = max(e["time"] for e in execs)
            else:
                status = "resting"
                resolve_time = None

            resting_time = (resolve_time - submitted_at) if resolve_time is not None and sub else None

            records.append(
                OrderLifecycleRecord(
                    order_id=oid,
                    submitted_at_ns=submitted_at,
                    status=status,
                    resting_time_ns=resting_time,
                    filled_qty=filled_qty,
                    submitted_qty=submitted_qty,
                )
            )

        return records

    @staticmethod
    def get_counterparty_breakdown(
        result: SimulationOutput,
        agent_id: int,
    ) -> list[CounterpartySummary]:
        """Breakdown of which agent types the strategy traded against.

        Builds a mapping from ``end_state["agents"]`` and matches fills
        by ``(timestamp, fill_price, qty, opposite side)``.
        """
        cp_map = AnalysisService._build_counterparty_map(result, agent_id)

        if not cp_map:
            return []

        # Also need fill records to get qty
        agent_fills = AnalysisService._get_agent_fills(result, agent_id)
        if agent_fills is None:
            return []
        _agent, parsed = agent_fills

        # Aggregate by counterparty type
        type_stats: dict[str, dict] = {}
        for fill in parsed:
            cp_type = cp_map.get((fill["time"], fill["price"], fill["qty"]), "Unknown")
            if cp_type is None:
                cp_type = "Unknown"
            if cp_type not in type_stats:
                type_stats[cp_type] = {"count": 0, "total_qty": 0}
            type_stats[cp_type]["count"] += 1
            type_stats[cp_type]["total_qty"] += fill["qty"]

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
        result: SimulationOutput,
        agent_id: int,
        initial_cash: int = 0,
        adverse_window_ns: int = 500_000_000,
    ) -> RichAnalysisBundle:
        """Compute the full rich-analysis bundle for one agent in one scenario.

        Convenience method that calls all analysis methods and returns
        a serialisable ``RichAnalysisBundle``.  The bundle includes raw
        queryable data (mid-price series, sampled L2 snapshots) so that
        investigation tools can operate purely from serialised JSON.
        """
        fills = AnalysisService.get_fill_analysis(result, agent_id)
        pnl_curve = AnalysisService.get_pnl_curve(result, agent_id, initial_cash)
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
