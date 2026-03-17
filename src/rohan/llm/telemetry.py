"""Lightweight structured telemetry helpers for the refinement loop."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("rohan.telemetry")
TELEMETRY_SCHEMA_VERSION = 1


def _safe_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Normalize field values so telemetry emission cannot break callers."""
    safe: dict[str, Any] = {}
    for key, value in fields.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


def emit_metric(event: str, *, component: str = "rohan.refinement", run_id: str | None = None, **fields: Any) -> None:
    """Emit a structured metric event as one JSON log line.

    This helper is intentionally failure-safe: telemetry issues should never
    affect the refinement control path.
    """
    payload: dict[str, Any] = {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "component": component,
        "kind": "metric",
        "event": event,
        "ts": datetime.now(UTC).isoformat(),
        **_safe_fields(fields),
    }
    if run_id is not None:
        payload["run_id"] = run_id

    try:
        logger.info("telemetry %s", json.dumps(payload, default=str, sort_keys=True))
    except Exception:
        logger.debug("Telemetry emit failed for event=%s", event, exc_info=True)


@dataclass(frozen=True)
class TelemetryContext:
    """Reusable context that applies stable fields to every emitted event."""

    component: str
    run_id: str | None = None
    base_fields: dict[str, Any] = field(default_factory=dict)

    def emit(self, event: str, **fields: Any) -> None:
        """Emit telemetry with context-level fields merged into event fields."""
        merged_fields = {**self.base_fields, **fields}
        emit_metric(event, component=self.component, run_id=self.run_id, **merged_fields)


def refinement_telemetry_context(
    run_id: str | None,
    *,
    iteration: int | None = None,
) -> TelemetryContext:
    """Build a standard telemetry context for refinement-loop events."""
    base_fields: dict[str, Any] = {}
    if iteration is not None:
        base_fields["iteration"] = iteration
    return TelemetryContext(
        component="rohan.llm.refinement",
        run_id=run_id,
        base_fields=base_fields,
    )
