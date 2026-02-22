"""Tests for Explainer tools (Phase 2.4)."""

from unittest.mock import MagicMock

import pandas as pd

from rohan.llm.tools import make_explainer_tools


def _mock_output(
    l1_data: pd.DataFrame | None = None,
    logs_data: pd.DataFrame | None = None,
):
    """Create a mock SimulationOutput for tool testing."""
    output = MagicMock()

    if l1_data is None:
        l1_data = pd.DataFrame(
            {
                "time": [1_000_000_000, 2_000_000_000, 3_000_000_000],
                "bid_price": [10000, 10010, 10020],
                "bid_qty": [100, 150, 120],
                "ask_price": [10050, 10060, 10070],
                "ask_qty": [80, 90, 100],
                "timestamp": [1_000_000_000, 2_000_000_000, 3_000_000_000],
            }
        )

    if logs_data is None:
        logs_data = pd.DataFrame(
            {
                "AgentID": [0, 1, 1, 0],
                "AgentType": ["ExchangeAgent", "StrategicAgent", "StrategicAgent", "ExchangeAgent"],
                "EventType": ["ORDER_SUBMITTED", "ORDER_SUBMITTED", "FILL", "FILL"],
            }
        )

    output.get_order_book_l1.return_value = l1_data
    output.get_logs_df.return_value = logs_data
    return output


class TestMakeExplainerTools:
    def test_returns_list_of_tools(self):
        output = _mock_output()
        tools = make_explainer_tools(output)
        assert isinstance(tools, list)
        assert len(tools) == 8

    def test_tool_names(self):
        output = _mock_output()
        tools = make_explainer_tools(output)
        names = {t.name for t in tools}
        assert "get_order_book_snapshot" in names
        assert "get_agent_trades" in names
        assert "compute_pnl_curve" in names
        assert "get_price_stats" in names
        assert "get_spread_stats" in names
        assert "query_logs" in names
        assert "get_volume_profile" in names
        assert "get_microstructure_stats" in names


class TestOrderBookSnapshot:
    def test_returns_bid_ask(self):
        output = _mock_output()
        tools = make_explainer_tools(output)
        snapshot_tool = next(t for t in tools if t.name == "get_order_book_snapshot")
        result = snapshot_tool.invoke({"timestamp_ns": 1_500_000_000})
        assert "Bid:" in result
        assert "Ask:" in result
        assert "Mid:" in result

    def test_empty_order_book(self):
        output = _mock_output(l1_data=pd.DataFrame())
        tools = make_explainer_tools(output)
        snapshot_tool = next(t for t in tools if t.name == "get_order_book_snapshot")
        result = snapshot_tool.invoke({"timestamp_ns": 0})
        assert "No order book data" in result


class TestAgentTrades:
    def test_gets_fills(self):
        output = _mock_output()
        tools = make_explainer_tools(output)
        trades_tool = next(t for t in tools if t.name == "get_agent_trades")
        result = trades_tool.invoke({"agent_id": 1})
        assert "StrategicAgent" in result

    def test_no_trades(self):
        output = _mock_output()
        tools = make_explainer_tools(output)
        trades_tool = next(t for t in tools if t.name == "get_agent_trades")
        result = trades_tool.invoke({"agent_id": 999})
        assert "No trades found" in result

    def test_empty_logs(self):
        output = _mock_output(logs_data=pd.DataFrame())
        tools = make_explainer_tools(output)
        trades_tool = next(t for t in tools if t.name == "get_agent_trades")
        result = trades_tool.invoke({"agent_id": 1})
        assert "No log data" in result


class TestPriceStats:
    def test_returns_stats(self):
        output = _mock_output()
        tools = make_explainer_tools(output)
        stats_tool = next(t for t in tools if t.name == "get_price_stats")
        result = stats_tool.invoke({})
        assert "Mid-price stats" in result
        assert "mean" in result

    def test_empty_data(self):
        output = _mock_output(l1_data=pd.DataFrame())
        tools = make_explainer_tools(output)
        stats_tool = next(t for t in tools if t.name == "get_price_stats")
        result = stats_tool.invoke({})
        assert "No order book data" in result


class TestSpreadStats:
    def test_returns_stats(self):
        output = _mock_output()
        tools = make_explainer_tools(output)
        spread_tool = next(t for t in tools if t.name == "get_spread_stats")
        result = spread_tool.invoke({})
        assert "Spread stats" in result

    def test_empty_data(self):
        output = _mock_output(l1_data=pd.DataFrame())
        tools = make_explainer_tools(output)
        spread_tool = next(t for t in tools if t.name == "get_spread_stats")
        result = spread_tool.invoke({})
        assert "No order book data" in result


class TestQueryLogs:
    def test_no_filter(self):
        output = _mock_output()
        tools = make_explainer_tools(output)
        query_tool = next(t for t in tools if t.name == "query_logs")
        result = query_tool.invoke({})
        # Should return all 4 rows
        assert "ExchangeAgent" in result

    def test_filter_by_agent_type(self):
        output = _mock_output()
        tools = make_explainer_tools(output)
        query_tool = next(t for t in tools if t.name == "query_logs")
        result = query_tool.invoke({"agent_type": "Strategic"})
        assert "StrategicAgent" in result

    def test_filter_by_event_type(self):
        output = _mock_output()
        tools = make_explainer_tools(output)
        query_tool = next(t for t in tools if t.name == "query_logs")
        result = query_tool.invoke({"event_type": "FILL"})
        assert "FILL" in result

    def test_no_match(self):
        output = _mock_output()
        tools = make_explainer_tools(output)
        query_tool = next(t for t in tools if t.name == "query_logs")
        result = query_tool.invoke({"agent_type": "NonExistent"})
        assert "No logs matching" in result


class TestVolumeProfile:
    def test_returns_profile(self):
        output = _mock_output()
        tools = make_explainer_tools(output)
        vol_tool = next(t for t in tools if t.name == "get_volume_profile")
        result = vol_tool.invoke({})
        assert "fill events" in result.lower() or "Total" in result

    def test_empty_logs(self):
        output = _mock_output(logs_data=pd.DataFrame())
        tools = make_explainer_tools(output)
        vol_tool = next(t for t in tools if t.name == "get_volume_profile")
        result = vol_tool.invoke({})
        assert "No log data" in result


class TestPnlCurve:
    def test_returns_stats(self):
        output = _mock_output()
        tools = make_explainer_tools(output)
        pnl_tool = next(t for t in tools if t.name == "compute_pnl_curve")
        result = pnl_tool.invoke({})
        assert "PnL curve stats" in result

    def test_empty_data(self):
        output = _mock_output(l1_data=pd.DataFrame())
        tools = make_explainer_tools(output)
        pnl_tool = next(t for t in tools if t.name == "compute_pnl_curve")
        result = pnl_tool.invoke({})
        assert "No order book data" in result
