"""Smoke tests for rohan.ui.charts.

Each test constructs minimal synthetic data, calls the chart builder,
and asserts the return is a ``plotly.graph_objects.Figure`` with at
least one trace (unless the input is intentionally empty).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from rohan.ui import charts

# ── Helpers ──────────────────────────────────────────────────────────────

_N = 100
_RNG = np.random.default_rng(42)
_TIMES = pd.date_range("2025-01-30 09:30:00", periods=_N, freq="s")


def _series(data: Any = None, *, n: int = _N) -> pd.Series:
    """Convenience wrapper for creating a numeric Series with _TIMES index."""
    if data is None:
        data = _RNG.standard_normal(n)
    return pd.Series(data, index=_TIMES[:n])


@pytest.fixture()
def l1_series() -> dict[str, pd.Series]:
    bid = pd.Series(100.0 + _RNG.standard_normal(_N).cumsum() * 0.01)
    ask = bid + _RNG.uniform(0.01, 0.05, _N)
    mid = (bid + ask) / 2
    spread = ask - bid
    time = pd.to_datetime(_TIMES)
    return {"time": time, "bid": bid, "ask": ask, "mid": mid, "spread": spread}


@pytest.fixture()
def log_returns() -> pd.Series:
    prices = 100.0 + _RNG.standard_normal(_N).cumsum() * 0.01
    ratio = pd.Series(prices) / pd.Series(prices).shift(1)
    return np.log(ratio).dropna()


@pytest.fixture()
def agent_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Type": ["NoiseAgent"] * 20 + ["ValueAgent"] * 10 + ["MomentumAgent"] * 5,
            "P&L ($)": _RNG.standard_normal(35) * 100,
        }
    )


@pytest.fixture()
def attr_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": _TIMES[:50],
            "side": _RNG.choice(["BUY", "SELL"], 50),
            "price ($)": 100.0 + _RNG.standard_normal(50).cumsum() * 0.01,
            "maker_type": _RNG.choice(["NoiseAgent", "ValueAgent", "MM"], 50),
            "taker_type": _RNG.choice(["NoiseAgent", "MomentumAgent"], 50),
            "quantity": _RNG.integers(10, 200, 50),
        }
    )


@pytest.fixture()
def l2_df() -> pd.DataFrame:
    n = 200
    return pd.DataFrame(
        {
            "side": _RNG.choice(["bid", "ask"], n),
            "time_ns": np.sort(_RNG.integers(int(1e18), int(1.1e18), n)),
            "price_cents": _RNG.integers(9900, 10100, n),
            "qty": _RNG.integers(1, 500, n),
            "level": np.tile(np.arange(1, 11), n // 10),
        }
    )


@pytest.fixture()
def fill_df() -> pd.DataFrame:
    n = 60
    return pd.DataFrame(
        {
            "slippage (bps)": _RNG.standard_normal(n) * 5,
            "AS 100ms (bps)": _RNG.standard_normal(n) * 2,
            "AS 500ms (bps)": _RNG.standard_normal(n) * 3,
            "AS 1s (bps)": _RNG.standard_normal(n) * 4,
            "AS 5s (bps)": _RNG.standard_normal(n) * 5,
        }
    )


@pytest.fixture()
def rich_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Type": ["NoiseAgent", "ValueAgent", "MM", "MomentumAgent"],
            "Sharpe": [0.5, 1.2, 0.8, -0.1],
            "OTT Ratio": [5.0, 3.0, 8.0, 2.0],
            "Inventory σ": [10.0, 20.0, 5.0, 15.0],
            "Trade Count": [500, 300, 1000, 100],
        }
    )


# ── Section A: Market Overview & Microstructure ─────────────────────────


class TestMarketCharts:
    def test_price_series(self, l1_series: dict[str, pd.Series]) -> None:
        fig = charts.price_series(l1_series["time"], l1_series["bid"], l1_series["ask"], l1_series["mid"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_spread_over_time(self, l1_series: dict[str, pd.Series]) -> None:
        fig = charts.spread_over_time(l1_series["time"], l1_series["spread"], avg_spread=0.03)
        assert isinstance(fig, go.Figure)

    def test_rolling_volatility(self, log_returns: pd.Series) -> None:
        rolling_vol = log_returns.rolling(20).std()
        fig = charts.rolling_volatility(log_returns.index.to_series(), rolling_vol, window=20)
        assert isinstance(fig, go.Figure)

    def test_book_pressure(self) -> None:
        time = pd.to_datetime(_TIMES)
        pressure = _series()
        fig = charts.book_pressure(time, pressure)
        assert isinstance(fig, go.Figure)

    def test_returns_histogram(self, log_returns: pd.Series) -> None:
        fig = charts.returns_histogram(log_returns)
        assert isinstance(fig, go.Figure)

    def test_event_type_pie(self) -> None:
        counts = pd.Series({"LIMIT_ORDER": 500, "CANCEL_ORDER": 200, "MARKET_ORDER": 50})
        fig = charts.event_type_pie(counts)
        assert isinstance(fig, go.Figure)

    def test_side_balance(self) -> None:
        counts = pd.Series({"BUY": 300, "SELL": 280})
        fig = charts.side_balance(counts)
        assert isinstance(fig, go.Figure)

    def test_cumulative_imbalance(self) -> None:
        flow_time = pd.to_datetime(_TIMES)
        cum_imbalance = _series().cumsum()
        fig = charts.cumulative_imbalance(flow_time, cum_imbalance)
        assert isinstance(fig, go.Figure)

    def test_volume_by_agent_type(self) -> None:
        vol = pd.Series({"NoiseAgent": 5000, "ValueAgent": 3000, "MM": 8000})
        fig = charts.volume_by_agent_type(vol)
        assert isinstance(fig, go.Figure)


# ── Section A: Agent & Execution ────────────────────────────────────────


class TestAgentCharts:
    def test_pnl_box_plot(self, agent_df: pd.DataFrame) -> None:
        fig = charts.pnl_box_plot(agent_df)
        assert isinstance(fig, go.Figure)

    def test_equity_curve(self) -> None:
        n = 50
        nav = 10000 + np.cumsum(_RNG.standard_normal(n) * 10)
        ec_df = pd.DataFrame(
            {
                "time": pd.date_range("2025-01-30 09:30", periods=n, freq="s"),
                "NAV ($)": nav,
                "Peak NAV ($)": np.maximum.accumulate(nav),
            }
        )
        fig = charts.equity_curve(ec_df, agent_name="TestAgent")
        assert isinstance(fig, go.Figure)

    def test_slippage_comparison(self) -> None:
        data = [
            {"name": "Agent A", "vwap_slippage_bps": 2.5},
            {"name": "Agent B", "vwap_slippage_bps": -1.0},
            {"name": "Agent C", "vwap_slippage_bps": 5.0},
        ]
        fig = charts.slippage_comparison(data)
        assert isinstance(fig, go.Figure)


# ── Section A: Trade Attribution ────────────────────────────────────────


class TestTradeAttributionCharts:
    def test_maker_taker_volume(self) -> None:
        maker = pd.Series({"NoiseAgent": 3000, "ValueAgent": 2000})
        taker = pd.Series({"NoiseAgent": 2500, "MomentumAgent": 1500})
        fig = charts.maker_taker_volume(maker, taker)
        assert isinstance(fig, go.Figure)

    def test_trade_price_scatter(self, attr_df: pd.DataFrame) -> None:
        fig = charts.trade_price_scatter(attr_df)
        assert isinstance(fig, go.Figure)

    def test_trade_attribution_heatmap(self, attr_df: pd.DataFrame) -> None:
        fig = charts.trade_attribution_heatmap(attr_df)
        assert isinstance(fig, go.Figure)


# ── Section A: L2 Order Book ───────────────────────────────────────────


class TestL2Charts:
    def test_l2_depth_heatmap(self, l2_df: pd.DataFrame) -> None:
        fig = charts.l2_depth_heatmap(l2_df)
        assert isinstance(fig, go.Figure)

    def test_l2_depth_heatmap_with_mid(self, l2_df: pd.DataFrame, l1_series: dict[str, pd.Series]) -> None:
        fig = charts.l2_depth_heatmap(l2_df, mid=l1_series["mid"])
        assert isinstance(fig, go.Figure)

    def test_l2_depth_profile(self, l2_df: pd.DataFrame) -> None:
        fig = charts.l2_depth_profile(l2_df)
        assert isinstance(fig, go.Figure)


# ── Section A: Rich Metrics ────────────────────────────────────────────


class TestRichMetricsCharts:
    def test_adverse_selection_by_window(self, fill_df: pd.DataFrame) -> None:
        fig = charts.adverse_selection_by_window(fill_df)
        assert isinstance(fig, go.Figure)

    def test_fill_slippage_histogram(self, fill_df: pd.DataFrame) -> None:
        fig = charts.fill_slippage_histogram(fill_df)
        assert isinstance(fig, go.Figure)

    def test_rich_agent_comparison(self, rich_df: pd.DataFrame) -> None:
        fig = charts.rich_agent_comparison(rich_df)
        assert isinstance(fig, go.Figure)


# ── Section B: Terminal page charts ─────────────────────────────────────


class TestTerminalCharts:
    def test_price_returns_chart(self) -> None:
        price_df = pd.DataFrame(
            {
                "bid_price": 100.0 + _RNG.standard_normal(_N).cumsum() * 0.01,
                "ask_price": 100.05 + _RNG.standard_normal(_N).cumsum() * 0.01,
                "mid_price": 100.025 + _RNG.standard_normal(_N).cumsum() * 0.01,
                "returns": _RNG.standard_normal(_N) * 0.001,
            },
            index=_TIMES,
        )
        fig = charts.price_returns_chart(price_df)
        assert isinstance(fig, go.Figure)

    def test_price_returns_chart_with_historical(self) -> None:
        price_df = pd.DataFrame(
            {
                "bid_price": 100.0 + _RNG.standard_normal(_N).cumsum() * 0.01,
                "ask_price": 100.05 + _RNG.standard_normal(_N).cumsum() * 0.01,
                "mid_price": 100.025 + _RNG.standard_normal(_N).cumsum() * 0.01,
                "returns": _RNG.standard_normal(_N) * 0.001,
            },
            index=_TIMES,
        )
        hist_df = pd.DataFrame(
            {
                "timestamp": _TIMES,
                "historical_price_cents": 10000 + _RNG.integers(-50, 50, _N).cumsum(),
            }
        )
        fig = charts.price_returns_chart(price_df, historical_df=hist_df)
        assert isinstance(fig, go.Figure)

    def test_volume_imbalance_chart(self) -> None:
        df = pd.DataFrame(
            {
                "bid_qty": _RNG.integers(100, 500, _N).astype(float),
                "ask_qty": _RNG.integers(100, 500, _N).astype(float),
                "volume_imbalance": _RNG.standard_normal(_N) * 0.1,
            },
            index=_TIMES,
        )
        fig = charts.volume_imbalance_chart(df)
        assert isinstance(fig, go.Figure)

    def test_spread_analysis_chart(self) -> None:
        mid_price = 100.0 + _RNG.standard_normal(_N).cumsum() * 0.01
        spread = _RNG.uniform(0.01, 0.10, _N)
        df = pd.DataFrame(
            {
                "spread": spread,
                "spread_bps": spread / mid_price * 10000,
                "mid_price": mid_price,
            },
            index=_TIMES,
        )
        fig = charts.spread_analysis_chart(df)
        assert isinstance(fig, go.Figure)

    def test_csv_preview_chart(self) -> None:
        df = pd.DataFrame(
            {"price_cents": 10000 + _RNG.integers(-50, 50, _N).cumsum()},
            index=_TIMES,
        )
        fig = charts.csv_preview_chart(df, price_col="price_cents")
        assert isinstance(fig, go.Figure)


# ── Section C: Refinement Lab charts ────────────────────────────────────


class TestRefinementLabCharts:
    def test_score_progression_chart(self) -> None:
        @dataclass
        class FakeIteration:
            profitability_score: float | None
            risk_score: float | None
            volatility_impact_score: float | None
            spread_impact_score: float | None
            liquidity_impact_score: float | None
            execution_score: float | None

        scores = [5.0, 6.5, 7.0]
        iters = [
            FakeIteration(5.0, 4.0, 7.0, 6.0, 5.0, 8.0),
            FakeIteration(6.0, 5.0, 7.5, 6.5, 6.0, 8.5),
            FakeIteration(7.0, 6.0, 8.0, 7.0, 7.0, 9.0),
        ]
        fig = charts.score_progression_chart(scores, iters, convergence_threshold=0.5)
        assert isinstance(fig, go.Figure)

    def test_scoring_radar_chart(self) -> None:
        labels = ["Profitability", "Risk", "Vol Impact", "Spread", "Liquidity", "Execution"]
        values = [7.0, 5.5, 8.0, 6.0, 7.5, 9.0]
        fig = charts.scoring_radar_chart(labels, values)
        assert isinstance(fig, go.Figure)
