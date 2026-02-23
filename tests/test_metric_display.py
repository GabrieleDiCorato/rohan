"""Tests for metric display configuration and delta helpers."""

from __future__ import annotations

import pytest

from rohan.simulation.models import SimulationMetrics
from rohan.ui.utils.metric_display import (
    METRIC_CONFIG,
    fmt_pct,
    get_delta_color,
    get_help,
    metric_delta,
    pct_delta,
)


# ---------------------------------------------------------------------------
# metric_delta
# ---------------------------------------------------------------------------
class TestMetricDelta:
    def test_both_present(self):
        assert metric_delta(10.0, 8.0) == 2.0

    def test_negative_delta(self):
        assert metric_delta(5.0, 10.0) == -5.0

    def test_current_none(self):
        assert metric_delta(None, 8.0) is None

    def test_previous_none(self):
        assert metric_delta(10.0, None) is None

    def test_both_none(self):
        assert metric_delta(None, None) is None

    def test_integers(self):
        assert metric_delta(100, 60) == 40


# ---------------------------------------------------------------------------
# pct_delta
# ---------------------------------------------------------------------------
class TestPctDelta:
    def test_increase(self):
        assert pct_delta(12.0, 10.0) == pytest.approx(0.2)

    def test_decrease(self):
        assert pct_delta(8.0, 10.0) == pytest.approx(-0.2)

    def test_zero_base_nonzero_current(self):
        assert pct_delta(5.0, 0.0) is None

    def test_both_zero(self):
        assert pct_delta(0.0, 0.0) == 0.0

    def test_none_current(self):
        assert pct_delta(None, 10.0) is None

    def test_none_previous(self):
        assert pct_delta(10.0, None) is None


# ---------------------------------------------------------------------------
# fmt_pct
# ---------------------------------------------------------------------------
class TestFmtPct:
    def test_positive(self):
        assert fmt_pct(0.123) == "+12.3%"

    def test_negative(self):
        assert fmt_pct(-0.05) == "-5.0%"

    def test_zero(self):
        assert fmt_pct(0.0) == "+0.0%"

    def test_none(self):
        assert fmt_pct(None) is None


# ---------------------------------------------------------------------------
# METRIC_CONFIG completeness
# ---------------------------------------------------------------------------
class TestMetricConfig:
    def test_all_simulation_metrics_fields_covered(self):
        """Every SimulationMetrics field must have a display config entry."""
        for field in SimulationMetrics.model_fields:
            assert field in METRIC_CONFIG, f"Missing display config for '{field}'"

    def test_delta_colors_valid(self):
        valid = {"normal", "inverse", "off"}
        for field, cfg in METRIC_CONFIG.items():
            assert cfg["delta_color"] in valid, f"{field} has invalid delta_color: {cfg['delta_color']}"

    def test_help_texts_non_empty(self):
        for field, cfg in METRIC_CONFIG.items():
            assert cfg["help"], f"{field} has empty help text"


# ---------------------------------------------------------------------------
# get_delta_color / get_help
# ---------------------------------------------------------------------------
class TestGetDeltaColor:
    def test_inverse_metrics(self):
        for f in ("volatility", "mean_spread", "effective_spread", "vpin", "resilience_mean_ns"):
            assert get_delta_color(f) == "inverse", f"{f} should be inverse"

    def test_normal_metrics(self):
        for f in ("avg_bid_liquidity", "avg_ask_liquidity", "traded_volume"):
            assert get_delta_color(f) == "normal", f"{f} should be normal"

    def test_off_metrics(self):
        for f in ("lob_imbalance_mean", "lob_imbalance_std", "market_ott_ratio"):
            assert get_delta_color(f) == "off", f"{f} should be off"

    def test_unknown_field_defaults_to_off(self):
        assert get_delta_color("nonexistent_field") == "off"


class TestGetHelp:
    def test_known_fields_have_help(self):
        for field in METRIC_CONFIG:
            h = get_help(field)
            assert h is not None
            assert len(h) > 10, f"{field} help text too short"

    def test_unknown_field_returns_none(self):
        assert get_help("nonexistent_field") is None
