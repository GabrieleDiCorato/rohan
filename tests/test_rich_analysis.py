"""Tests for rich analysis models, charts, and integration.

Covers:
- FillRecord / PnLPoint / InventoryPoint / etc. serialisation round-trip
- Chart methods (plot_pnl_curve, plot_inventory, plot_fills_vs_mid)
- Integration test with a real 5-minute ABIDES simulation
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from rohan.framework.analysis_models import (
    CounterpartySummary,
    FillRecord,
    InventoryPoint,
    OrderLifecycleRecord,
    PnLPoint,
    RichAnalysisBundle,
)
from rohan.framework.analysis_service import AnalysisService

# ═══════════════════════════════════════════════════════════════════════════
# Model serialisation tests
# ═══════════════════════════════════════════════════════════════════════════


class TestModelSerialisation:
    """Verify Pydantic models round-trip through JSON."""

    def test_fill_record_round_trip(self):
        fr = FillRecord(timestamp_ns=100, side="BUY", price=10000, qty=5, mid_at_fill=9990.0, slippage_bps=1.0, counterparty_type="NoiseAgent")
        restored = FillRecord.model_validate_json(fr.model_dump_json())
        assert restored == fr

    def test_pnl_point_round_trip(self):
        p = PnLPoint(timestamp_ns=200, mark_to_market_pnl=-50.5)
        assert PnLPoint.model_validate_json(p.model_dump_json()) == p

    def test_inventory_point_round_trip(self):
        ip = InventoryPoint(timestamp_ns=300, position=-3)
        assert InventoryPoint.model_validate_json(ip.model_dump_json()) == ip

    def test_order_lifecycle_round_trip(self):
        ol = OrderLifecycleRecord(order_id=7, submitted_at_ns=100, status="filled", resting_time_ns=50, filled_qty=10, submitted_qty=10)
        assert OrderLifecycleRecord.model_validate_json(ol.model_dump_json()) == ol

    def test_counterparty_summary_round_trip(self):
        cs = CounterpartySummary(agent_type="NoiseAgent", trade_count=5, avg_size=10.0, total_volume=50)
        assert CounterpartySummary.model_validate_json(cs.model_dump_json()) == cs

    def test_rich_bundle_round_trip(self):
        bundle = RichAnalysisBundle(
            fills=[FillRecord(timestamp_ns=1, side="BUY", price=100, qty=1)],
            pnl_curve=[PnLPoint(timestamp_ns=1, mark_to_market_pnl=0.0)],
            inventory_trajectory=[InventoryPoint(timestamp_ns=0, position=0)],
            adverse_selection_bps=2.5,
            counterparty_breakdown=[CounterpartySummary(agent_type="X", trade_count=1, avg_size=1.0, total_volume=1)],
            order_lifecycle=[OrderLifecycleRecord(order_id=1, submitted_at_ns=0, status="filled")],
        )
        restored = RichAnalysisBundle.model_validate_json(bundle.model_dump_json())
        assert restored == bundle

    def test_empty_bundle_round_trip(self):
        bundle = RichAnalysisBundle()
        restored = RichAnalysisBundle.model_validate_json(bundle.model_dump_json())
        assert restored == bundle
        assert restored.fills == []
        assert restored.adverse_selection_bps is None


# ═══════════════════════════════════════════════════════════════════════════
# Chart tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCharts:
    """Tests for the 3 new chart methods."""

    def test_plot_pnl_curve_returns_figure(self):
        points = [PnLPoint(timestamp_ns=i * 1_000_000_000, mark_to_market_pnl=float(i * 100)) for i in range(5)]
        fig = AnalysisService.plot_pnl_curve(points)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_pnl_curve_empty(self):
        fig = AnalysisService.plot_pnl_curve([])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_inventory_returns_figure(self):
        points = [
            InventoryPoint(timestamp_ns=0, position=0),
            InventoryPoint(timestamp_ns=1_000_000_000, position=5),
            InventoryPoint(timestamp_ns=2_000_000_000, position=3),
            InventoryPoint(timestamp_ns=3_000_000_000, position=-2),
        ]
        fig = AnalysisService.plot_inventory(points)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_inventory_empty(self):
        fig = AnalysisService.plot_inventory([])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_fills_vs_mid_returns_figure(self):
        fills = [
            FillRecord(timestamp_ns=1_000_000_000, side="BUY", price=10050, qty=1, mid_at_fill=10000.0),
            FillRecord(timestamp_ns=2_000_000_000, side="SELL", price=9950, qty=1, mid_at_fill=10000.0),
        ]
        fig = AnalysisService.plot_fills_vs_mid(fills)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_fills_vs_mid_empty(self):
        fig = AnalysisService.plot_fills_vs_mid([])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_fills_no_mid(self):
        """Fills without mid_at_fill should still render."""
        fills = [FillRecord(timestamp_ns=1_000_000_000, side="BUY", price=10000, qty=1)]
        fig = AnalysisService.plot_fills_vs_mid(fills)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Integration test (real ABIDES simulation)
# ═══════════════════════════════════════════════════════════════════════════


class TestRichAnalysisIntegration:
    """Integration test with a real 5-minute ABIDES simulation."""

    @pytest.fixture
    def simulation_output(self):
        """Run a real simulation for integration testing."""
        from rohan.config import SimulationSettings
        from rohan.simulation import SimulationService

        settings = SimulationSettings()
        settings.start_time = "09:30:00"
        settings.end_time = "09:35:00"
        settings.seed = 42
        settings.agents.noise.num_agents = 100
        settings.agents.value.num_agents = 20
        settings.agents.momentum.num_agents = 2
        settings.agents.adaptive_market_maker.num_agents = 1

        service = SimulationService()
        result = service.run_simulation(settings)
        if result.error is not None:
            raise result.error
        return result.result

    def test_compute_metrics_still_works(self, simulation_output):
        """Existing compute_metrics must not break."""
        analyzer = AnalysisService()
        metrics = analyzer.compute_metrics(simulation_output)
        assert metrics is not None
        assert metrics.mean_spread is not None

    def test_compute_agent_metrics_still_works(self, simulation_output):
        """Existing compute_agent_metrics must not break."""
        analyzer = AnalysisService()
        # Agent 1 is typically the first trading agent (agent 0 is ExchangeAgent).
        # HasufelOutput lacks end_state, so compute_agent_metrics returns an
        # empty AgentMetrics via its hasattr guard — verify it doesn't crash.
        metrics = analyzer.compute_agent_metrics(simulation_output, 1)
        assert metrics is not None

    def test_plots_still_work(self, simulation_output):
        """Existing plot methods must not break."""
        analyzer = AnalysisService()
        fig = analyzer.plot_price_series(simulation_output)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        fig = analyzer.plot_spread(simulation_output)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
