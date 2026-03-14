"""Tests for Terminal baseline comparison helpers."""

from __future__ import annotations

import pytest

from rohan.config import SimulationSettings
from rohan.exceptions import BaselineComparisonError
from rohan.ui.utils.baseline_comparison import (
    build_baseline_context_table,
    ensure_baseline_comparable,
    get_baseline_compatibility_issues,
)


def _settings(**overrides) -> SimulationSettings:
    values = {
        "date": "20260130",
        "start_time": "09:30:00",
        "end_time": "11:30:00",
        "seed": 42,
    }
    values.update(overrides)
    return SimulationSettings(**values)


class TestBaselineCompatibility:
    def test_matching_configs_have_no_issues(self):
        issues = get_baseline_compatibility_issues(_settings(), _settings())
        assert issues == []

    def test_seed_mismatch_is_reported(self):
        issues = get_baseline_compatibility_issues(_settings(), _settings(seed=99))
        assert issues == ["Random Seed differs: current=42, baseline=99"]

    def test_ensure_baseline_comparable_raises(self):
        with pytest.raises(BaselineComparisonError, match="Start Time differs"):
            ensure_baseline_comparable(_settings(), _settings(start_time="10:00:00", end_time="12:00:00"))

    def test_context_table_marks_matches_and_mismatches(self):
        table = build_baseline_context_table(_settings(), _settings(seed=7))

        assert list(table.columns) == ["Setting", "Current Run", "Baseline", "Status"]
        assert len(table) == 5

        seed_row = table.loc[table["Setting"] == "Random Seed"].iloc[0]
        assert seed_row["Status"] == "Mismatch"

        duration_row = table.loc[table["Setting"] == "Duration"].iloc[0]
        assert duration_row["Current Run"] == "120 min"
        assert duration_row["Status"] == "Match"
