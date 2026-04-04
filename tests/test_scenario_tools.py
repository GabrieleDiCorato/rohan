"""Tests for scenario tools (R11 build_scenario, R12 explain_metrics, §7.2 validate_scenario)."""

import json

import pytest

from rohan.llm.scenario_tools import make_scenario_tools
from rohan.simulation.models.simulation_metrics import (
    AgentMetrics,
    ComparisonResult,
    MarketImpact,
    MarketMetrics,
)


@pytest.fixture()
def tools():
    return {t.name: t for t in make_scenario_tools()}


# ── build_scenario ──────────────────────────────────────────────────


class TestBuildScenario:
    def test_default_template(self, tools):
        result = tools["build_scenario"].invoke({"ticker": "ABM", "seed": 99})
        parsed = json.loads(result)
        assert parsed["template"] == "rmsc04"
        assert "config" in parsed

    def test_select_by_regime_tags(self, tools):
        result = tools["build_scenario"].invoke({"regime_tags": ["volatile", "megashocks"]})
        parsed = json.loads(result)
        assert parsed["template"] == "volatile_day"
        assert "volatile" in parsed["regime_tags"]

    def test_explicit_template_name(self, tools):
        result = tools["build_scenario"].invoke({"template_name": "stress_test"})
        parsed = json.loads(result)
        assert parsed["template"] == "stress_test"

    def test_invalid_template_returns_error(self, tools):
        result = tools["build_scenario"].invoke({"template_name": "nonexistent"})
        assert "Error" in result

    def test_overlay_stacking(self, tools):
        result = tools["build_scenario"].invoke(
            {
                "template_name": "rmsc04",
                "overlays": ["with_momentum", "with_execution"],
            }
        )
        parsed = json.loads(result)
        assert parsed["template"] == "rmsc04"
        assert parsed["applied_overlays"] == ["with_momentum", "with_execution"]
        assert "config" in parsed

    def test_invalid_overlay_skipped(self, tools):
        result = tools["build_scenario"].invoke(
            {
                "template_name": "rmsc04",
                "overlays": ["with_momentum", "nonexistent_overlay"],
            }
        )
        parsed = json.loads(result)
        assert parsed["applied_overlays"] == ["with_momentum"]

    def test_override_allowlist_blocks_disallowed(self, tools):
        """Builder overrides not in the allowlist are silently skipped."""
        result = tools["build_scenario"].invoke(
            {
                "template_name": "rmsc04",
                "overrides": {"seed": 999, "compile": None},
            }
        )
        # Should succeed (compile is skipped, seed is applied)
        parsed = json.loads(result)
        assert parsed["template"] == "rmsc04"


# ── validate_scenario ───────────────────────────────────────────────


class TestValidateScenario:
    def test_valid_config(self, tools):
        # Build a valid config first
        build_result = tools["build_scenario"].invoke({"template_name": "rmsc04"})
        config = json.loads(build_result)["config"]
        result = tools["validate_scenario"].invoke({"config_json": json.dumps(config)})
        assert "Valid:" in result

    def test_invalid_json(self, tools):
        result = tools["validate_scenario"].invoke({"config_json": "not json"})
        assert "Invalid JSON" in result

    def test_empty_config(self, tools):
        result = tools["validate_scenario"].invoke({"config_json": "{}"})
        assert "Valid:" in result


# ── explain_metrics ─────────────────────────────────────────────────


def _make_comparison(
    pnl=100.0,
    sharpe=0.5,
    fill_rate=0.6,
    vol_delta=0.0,
    spread_delta=0.0,
    vpin=0.3,
    inventory_std=10.0,
    max_drawdown=-500.0,
) -> str:
    return ComparisonResult(
        strategy_metrics=AgentMetrics(
            agent_id=1,
            total_pnl=pnl,
            sharpe_ratio=sharpe,
            fill_rate=fill_rate,
            inventory_std=inventory_std,
            max_drawdown=max_drawdown,
            trade_count=20,
            vwap_cents=10050,
        ),
        strategy_market_metrics=MarketMetrics(volatility=0.15, vpin=0.3),
        baseline_metrics=MarketMetrics(volatility=0.14, vpin=vpin),
        market_impact=MarketImpact(
            volatility_delta_pct=vol_delta,
            spread_delta_pct=spread_delta,
        ),
    ).model_dump_json()


class TestExplainMetrics:
    def test_profitable_strategy(self, tools):
        result = tools["explain_metrics"].invoke({"comparison_json": _make_comparison(pnl=500)})
        assert "profitable" in result.lower()

    def test_destabilising_strategy(self, tools):
        result = tools["explain_metrics"].invoke({"comparison_json": _make_comparison(pnl=500, vol_delta=10.0)})
        assert "destabilising" in result.lower()
        assert "Reduce" in result

    def test_passive_strategy(self, tools):
        result = tools["explain_metrics"].invoke({"comparison_json": _make_comparison(pnl=-100, fill_rate=0.1)})
        assert "passive" in result.lower()

    def test_high_inventory_risk(self, tools):
        result = tools["explain_metrics"].invoke({"comparison_json": _make_comparison(inventory_std=100, max_drawdown=-10000)})
        assert "unhedged" in result.lower()

    def test_vwap_reported(self, tools):
        result = tools["explain_metrics"].invoke({"comparison_json": _make_comparison()})
        assert "VWAP" in result

    def test_invalid_json(self, tools):
        result = tools["explain_metrics"].invoke({"comparison_json": "invalid"})
        assert "Failed" in result
