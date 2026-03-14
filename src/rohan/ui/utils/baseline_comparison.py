"""Helpers for fair baseline comparisons in the Terminal UI."""

from __future__ import annotations

from datetime import datetime

from rohan.config import SimulationSettings
from rohan.exceptions import BaselineComparisonError

_COMPARABILITY_FIELDS: list[tuple[str, str]] = [
    ("date", "Date"),
    ("start_time", "Start Time"),
    ("end_time", "End Time"),
    ("seed", "Random Seed"),
]


def _duration_minutes(settings: SimulationSettings) -> int:
    start = datetime.strptime(settings.start_time, "%H:%M:%S")
    end = datetime.strptime(settings.end_time, "%H:%M:%S")
    return int((end - start).total_seconds() // 60)


def get_baseline_compatibility_issues(
    current: SimulationSettings,
    baseline: SimulationSettings,
) -> list[str]:
    """Return human-readable reasons a baseline is not comparable."""
    issues: list[str] = []
    for field_name, label in _COMPARABILITY_FIELDS:
        current_value = getattr(current, field_name)
        baseline_value = getattr(baseline, field_name)
        if current_value != baseline_value:
            issues.append(f"{label} differs: current={current_value}, baseline={baseline_value}")
    return issues


def ensure_baseline_comparable(
    current: SimulationSettings,
    baseline: SimulationSettings,
) -> None:
    """Raise when baseline comparison would be unfair."""
    issues = get_baseline_compatibility_issues(current, baseline)
    if issues:
        raise BaselineComparisonError("; ".join(issues))


def build_baseline_context_table(
    current: SimulationSettings,
    baseline: SimulationSettings,
):
    """Build a compact current-vs-baseline fairness table."""
    import pandas as pd

    rows: list[dict[str, str]] = []
    for field_name, label in _COMPARABILITY_FIELDS:
        current_value = str(getattr(current, field_name))
        baseline_value = str(getattr(baseline, field_name))
        rows.append(
            {
                "Setting": label,
                "Current Run": current_value,
                "Baseline": baseline_value,
                "Status": "Match" if current_value == baseline_value else "Mismatch",
            }
        )

    current_duration = _duration_minutes(current)
    baseline_duration = _duration_minutes(baseline)
    rows.append(
        {
            "Setting": "Duration",
            "Current Run": f"{current_duration} min",
            "Baseline": f"{baseline_duration} min",
            "Status": "Match" if current_duration == baseline_duration else "Mismatch",
        }
    )

    return pd.DataFrame(rows)
