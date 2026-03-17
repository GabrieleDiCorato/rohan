"""Lightweight structured telemetry helpers for the refinement loop."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("rohan.telemetry")
TELEMETRY_SCHEMA_VERSION = 1


def emit_metric(event: str, *, component: str = "rohan.refinement", run_id: str | None = None, **fields: Any) -> None:
    """Emit a structured metric event as one JSON log line."""
    payload: dict[str, Any] = {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "component": component,
        "event": event,
        "ts": datetime.now(UTC).isoformat(),
        **fields,
    }
    if run_id is not None:
        payload["run_id"] = run_id
    logger.info("telemetry %s", json.dumps(payload, default=str, sort_keys=True))
