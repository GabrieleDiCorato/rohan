"""Tests for AnalysisService delegation to hasufel v2.5.8 compute_rich_metrics().

These tests verify that when AnalysisService receives a HasufelOutput
(from a real simulation), all metric computations are delegated to
hasufel's rich metrics API rather than Rohan's legacy local code.
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
from rohan.simulation.abides_impl.hasufel_output import HasufelOutput


@pytest.fixture(scope="module")
def hasufel_output() -> HasufelOutput:
    """Run a real 5-minute simulation and return a HasufelOutput."""
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
    output = result.result
    assert isinstance(output, HasufelOutput)
    return output


@pytest.fixture(scope="module")
def strategic_agent_id(hasufel_output: HasufelOutput) -> int:
    """Return the strategic agent ID from the simulation."""
    aid = hasufel_output.strategic_agent_id
    assert aid is not None
    return aid


@pytest.fixture(scope="module")
def trading_agent_id(hasufel_output: HasufelOutput) -> int:
    """Return the ID of an agent that actually traded (has fills)."""
    rich = hasufel_output.rich_metrics
    for ra in sorted(rich.agents, key=lambda a: a.trade_count, reverse=True):
        if ra.trade_count > 0:
            return ra.agent_id
    pytest.skip("No agents with trades in simulation")
    return 0  # unreachable


# ═══════════════════════════════════════════════════════════════════════════
# rich_metrics caching
# ═══════════════════════════════════════════════════════════════════════════


class TestRichMetricsCache:
    """Verify HasufelOutput.rich_metrics is cached and populated."""

    def test_rich_metrics_is_cached(self, hasufel_output: HasufelOutput) -> None:
        rich1 = hasufel_output.rich_metrics
        rich2 = hasufel_output.rich_metrics
        assert rich1 is rich2  # same object (cached_property)

    def test_rich_metrics_has_markets(self, hasufel_output: HasufelOutput) -> None:
        rich = hasufel_output.rich_metrics
        assert len(rich.markets) > 0

    def test_rich_metrics_has_agents(self, hasufel_output: HasufelOutput) -> None:
        rich = hasufel_output.rich_metrics
        assert len(rich.agents) > 0

    def test_rich_metrics_has_fills(self, hasufel_output: HasufelOutput) -> None:
        rich = hasufel_output.rich_metrics
        assert rich.fills is not None
        assert len(rich.fills) > 0


# ═══════════════════════════════════════════════════════════════════════════
# compute_metrics delegation
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeMetricsDelegation:
    """Verify compute_metrics uses hasufel for HasufelOutput."""

    def test_returns_populated_metrics(self, hasufel_output: HasufelOutput) -> None:
        metrics = AnalysisService.compute_metrics(hasufel_output)
        assert metrics is not None
        assert metrics.mean_spread is not None
        assert metrics.mean_spread > 0

    def test_volatility_populated(self, hasufel_output: HasufelOutput) -> None:
        metrics = AnalysisService.compute_metrics(hasufel_output)
        assert metrics.volatility is not None
        assert metrics.volatility > 0

    def test_microstructure_populated(self, hasufel_output: HasufelOutput) -> None:
        metrics = AnalysisService.compute_metrics(hasufel_output)
        assert metrics.lob_imbalance_mean is not None
        assert metrics.lob_imbalance_std is not None

    def test_pct_time_two_sided_in_range(self, hasufel_output: HasufelOutput) -> None:
        metrics = AnalysisService.compute_metrics(hasufel_output)
        assert metrics.pct_time_two_sided is not None
        # Rohan convention: 0-1
        assert 0.0 <= metrics.pct_time_two_sided <= 1.0

    def test_traded_volume(self, hasufel_output: HasufelOutput) -> None:
        metrics = AnalysisService.compute_metrics(hasufel_output)
        assert metrics.traded_volume is not None
        assert metrics.traded_volume > 0

    def test_effective_spread(self, hasufel_output: HasufelOutput) -> None:
        metrics = AnalysisService.compute_metrics(hasufel_output)
        assert metrics.effective_spread is not None
        assert metrics.effective_spread > 0

    def test_all_non_none_are_correct_types(self, hasufel_output: HasufelOutput) -> None:
        metrics = AnalysisService.compute_metrics(hasufel_output)
        float_fields = [
            "volatility",
            "mean_spread",
            "effective_spread",
            "avg_bid_liquidity",
            "avg_ask_liquidity",
            "lob_imbalance_mean",
            "lob_imbalance_std",
            "vpin",
            "resilience_mean_ns",
            "market_ott_ratio",
            "pct_time_two_sided",
        ]
        for field in float_fields:
            val = getattr(metrics, field, None)
            if val is not None:
                assert isinstance(val, float), f"{field} should be float, got {type(val)}"

        if metrics.traded_volume is not None:
            assert isinstance(metrics.traded_volume, int)


# ═══════════════════════════════════════════════════════════════════════════
# compute_agent_metrics delegation
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeAgentMetricsDelegation:
    """Verify compute_agent_metrics uses hasufel for HasufelOutput."""

    def test_returns_agent_metrics(
        self,
        hasufel_output: HasufelOutput,
        strategic_agent_id: int,
    ) -> None:
        metrics = AnalysisService.compute_agent_metrics(
            hasufel_output,
            strategic_agent_id,
        )
        assert metrics.agent_id == strategic_agent_id

    def test_pnl_populated(
        self,
        hasufel_output: HasufelOutput,
        strategic_agent_id: int,
    ) -> None:
        metrics = AnalysisService.compute_agent_metrics(
            hasufel_output,
            strategic_agent_id,
        )
        assert metrics.total_pnl is not None

    def test_trade_count_positive(
        self,
        hasufel_output: HasufelOutput,
        trading_agent_id: int,
    ) -> None:
        metrics = AnalysisService.compute_agent_metrics(
            hasufel_output,
            trading_agent_id,
        )
        assert metrics.trade_count > 0

    def test_fill_rate_in_range(
        self,
        hasufel_output: HasufelOutput,
        trading_agent_id: int,
    ) -> None:
        metrics = AnalysisService.compute_agent_metrics(
            hasufel_output,
            trading_agent_id,
        )
        if metrics.fill_rate is not None:
            # fill_rate = fills / submissions; can exceed 1.0 with partial fills
            assert metrics.fill_rate >= 0.0

    def test_missing_agent_raises(self, hasufel_output: HasufelOutput) -> None:
        with pytest.raises(ValueError, match="Agent 999999 not found"):
            AnalysisService.compute_agent_metrics(hasufel_output, 999999)


# ═══════════════════════════════════════════════════════════════════════════
# Rich analysis methods delegation
# ═══════════════════════════════════════════════════════════════════════════


class TestGetFillAnalysisDelegation:
    """Verify get_fill_analysis uses hasufel fills."""

    def test_returns_fill_records(
        self,
        hasufel_output: HasufelOutput,
        trading_agent_id: int,
    ) -> None:
        fills = AnalysisService.get_fill_analysis(hasufel_output, trading_agent_id)
        assert len(fills) > 0
        assert all(isinstance(f, FillRecord) for f in fills)

    def test_fill_has_slippage(
        self,
        hasufel_output: HasufelOutput,
        trading_agent_id: int,
    ) -> None:
        fills = AnalysisService.get_fill_analysis(hasufel_output, trading_agent_id)
        fills_with_slippage = [f for f in fills if f.slippage_bps is not None]
        assert len(fills_with_slippage) > 0

    def test_fill_has_counterparty(
        self,
        hasufel_output: HasufelOutput,
        trading_agent_id: int,
    ) -> None:
        fills = AnalysisService.get_fill_analysis(hasufel_output, trading_agent_id)
        fills_with_cp = [f for f in fills if f.counterparty_type is not None]
        assert len(fills_with_cp) > 0

    def test_unknown_agent_returns_empty(self, hasufel_output: HasufelOutput) -> None:
        fills = AnalysisService.get_fill_analysis(hasufel_output, 999999)
        assert fills == []


class TestGetPnlCurveDelegation:
    """Verify get_pnl_curve uses hasufel EquityCurve."""

    def test_returns_pnl_points(
        self,
        hasufel_output: HasufelOutput,
        trading_agent_id: int,
    ) -> None:
        pnl = AnalysisService.get_pnl_curve(hasufel_output, trading_agent_id)
        assert len(pnl) > 0
        assert all(isinstance(p, PnLPoint) for p in pnl)

    def test_unknown_agent_returns_empty(self, hasufel_output: HasufelOutput) -> None:
        pnl = AnalysisService.get_pnl_curve(hasufel_output, 999999)
        assert pnl == []


class TestGetInventoryTrajectoryDelegation:
    """Verify get_inventory_trajectory uses hasufel fills."""

    def test_returns_inventory_points(
        self,
        hasufel_output: HasufelOutput,
        trading_agent_id: int,
    ) -> None:
        traj = AnalysisService.get_inventory_trajectory(hasufel_output, trading_agent_id)
        assert len(traj) > 0
        assert all(isinstance(p, InventoryPoint) for p in traj)

    def test_unknown_agent_returns_empty(self, hasufel_output: HasufelOutput) -> None:
        traj = AnalysisService.get_inventory_trajectory(hasufel_output, 999999)
        assert traj == []


class TestGetAdverseSelectionDelegation:
    """Verify get_adverse_selection uses hasufel pre-computed fills."""

    def test_returns_float_or_none(
        self,
        hasufel_output: HasufelOutput,
        trading_agent_id: int,
    ) -> None:
        result = AnalysisService.get_adverse_selection(
            hasufel_output,
            trading_agent_id,
            window_ns=500_000_000,
        )
        assert result is None or isinstance(result, float)

    def test_unknown_agent_returns_none(self, hasufel_output: HasufelOutput) -> None:
        result = AnalysisService.get_adverse_selection(
            hasufel_output,
            999999,
        )
        assert result is None


class TestGetCounterpartyBreakdownDelegation:
    """Verify get_counterparty_breakdown uses hasufel trades."""

    def test_returns_counterparty_summaries(
        self,
        hasufel_output: HasufelOutput,
        trading_agent_id: int,
    ) -> None:
        breakdown = AnalysisService.get_counterparty_breakdown(
            hasufel_output,
            trading_agent_id,
        )
        assert len(breakdown) > 0
        assert all(isinstance(c, CounterpartySummary) for c in breakdown)

    def test_total_volume_positive(
        self,
        hasufel_output: HasufelOutput,
        trading_agent_id: int,
    ) -> None:
        breakdown = AnalysisService.get_counterparty_breakdown(
            hasufel_output,
            trading_agent_id,
        )
        total = sum(c.total_volume for c in breakdown)
        assert total > 0


class TestComputeRichAnalysisDelegation:
    """Verify compute_rich_analysis works end-to-end with HasufelOutput."""

    def test_returns_bundle(
        self,
        hasufel_output: HasufelOutput,
        trading_agent_id: int,
    ) -> None:
        bundle = AnalysisService.compute_rich_analysis(
            hasufel_output,
            trading_agent_id,
        )
        assert isinstance(bundle, RichAnalysisBundle)
        assert len(bundle.fills) > 0
        assert len(bundle.pnl_curve) > 0
        assert len(bundle.inventory_trajectory) > 0

    def test_adverse_selection_by_window(
        self,
        hasufel_output: HasufelOutput,
        trading_agent_id: int,
    ) -> None:
        bundle = AnalysisService.compute_rich_analysis(
            hasufel_output,
            trading_agent_id,
        )
        # Should have at least some of the standard windows
        assert len(bundle.adverse_selection_by_window) > 0

    def test_json_round_trip(
        self,
        hasufel_output: HasufelOutput,
        trading_agent_id: int,
    ) -> None:
        bundle = AnalysisService.compute_rich_analysis(
            hasufel_output,
            trading_agent_id,
        )
        json_str = bundle.model_dump_json()
        restored = RichAnalysisBundle.model_validate_json(json_str)
        assert len(restored.fills) == len(bundle.fills)

    def test_charts_work_from_bundle(
        self,
        hasufel_output: HasufelOutput,
        trading_agent_id: int,
    ) -> None:
        bundle = AnalysisService.compute_rich_analysis(
            hasufel_output,
            trading_agent_id,
        )
        fig = AnalysisService.plot_pnl_curve(bundle.pnl_curve)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        fig = AnalysisService.plot_inventory(bundle.inventory_trajectory)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        fig = AnalysisService.plot_fills_vs_mid(bundle.fills)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Order lifecycle — hasufel v2.5.8 uses OrderLifecycle model
# ═══════════════════════════════════════════════════════════════════════════


class TestOrderLifecycleDelegation:
    """get_order_lifecycle returns populated records for HasufelOutput (v2.5.8)."""

    def test_returns_non_empty(
        self,
        hasufel_output: HasufelOutput,
        trading_agent_id: int,
    ) -> None:
        lifecycle = AnalysisService.get_order_lifecycle(
            hasufel_output,
            trading_agent_id,
        )
        assert len(lifecycle) > 0

    def test_returns_lifecycle_records(
        self,
        hasufel_output: HasufelOutput,
        trading_agent_id: int,
    ) -> None:
        lifecycle = AnalysisService.get_order_lifecycle(
            hasufel_output,
            trading_agent_id,
        )
        assert all(isinstance(r, OrderLifecycleRecord) for r in lifecycle)

    def test_has_filled_orders(
        self,
        hasufel_output: HasufelOutput,
        trading_agent_id: int,
    ) -> None:
        lifecycle = AnalysisService.get_order_lifecycle(
            hasufel_output,
            trading_agent_id,
        )
        filled = [r for r in lifecycle if r.status == "filled"]
        assert len(filled) > 0

    def test_record_fields_valid(
        self,
        hasufel_output: HasufelOutput,
        trading_agent_id: int,
    ) -> None:
        lifecycle = AnalysisService.get_order_lifecycle(
            hasufel_output,
            trading_agent_id,
        )
        for rec in lifecycle:
            assert rec.order_id >= 0
            assert rec.submitted_at_ns >= 0
            assert rec.status in ("filled", "cancelled", "resting")
            assert rec.filled_qty >= 0
            assert rec.submitted_qty >= 0

    def test_resting_time_non_negative(
        self,
        hasufel_output: HasufelOutput,
        trading_agent_id: int,
    ) -> None:
        lifecycle = AnalysisService.get_order_lifecycle(
            hasufel_output,
            trading_agent_id,
        )
        for rec in lifecycle:
            if rec.resting_time_ns is not None:
                assert rec.resting_time_ns > 0

    def test_unknown_agent_returns_empty(
        self,
        hasufel_output: HasufelOutput,
    ) -> None:
        lifecycle = AnalysisService.get_order_lifecycle(hasufel_output, 999999)
        assert lifecycle == []


# ═══════════════════════════════════════════════════════════════════════════
# Bug fix: PnL curve density (v2.5.8 dense L1-interpolated curve)
# ═══════════════════════════════════════════════════════════════════════════


class TestPnlCurveDensity:
    """Verify PnL curve uses dense L1 interpolation (Bug 1 fix)."""

    def test_curve_denser_than_sparse(
        self,
        hasufel_output: HasufelOutput,
        trading_agent_id: int,
    ) -> None:
        """Dense curve should have more points than the raw fill count."""
        pnl = AnalysisService.get_pnl_curve(hasufel_output, trading_agent_id)
        # The sparse fill-only curve typically has ~4 points for a 5-min sim;
        # the dense L1-interpolated curve should have many more.
        assert len(pnl) > 4

    def test_timestamps_are_ascending(
        self,
        hasufel_output: HasufelOutput,
        trading_agent_id: int,
    ) -> None:
        pnl = AnalysisService.get_pnl_curve(hasufel_output, trading_agent_id)
        for i in range(1, len(pnl)):
            assert pnl[i].timestamp_ns >= pnl[i - 1].timestamp_ns

    def test_timestamps_since_midnight(
        self,
        hasufel_output: HasufelOutput,
        trading_agent_id: int,
    ) -> None:
        pnl = AnalysisService.get_pnl_curve(hasufel_output, trading_agent_id)
        # Timestamps should be ns-since-midnight, not epoch ns
        one_day_ns = 24 * 3600 * 10**9
        for p in pnl:
            assert 0 <= p.timestamp_ns < one_day_ns
