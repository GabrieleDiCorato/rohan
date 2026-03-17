"""Lightweight structured telemetry helpers for the refinement loop."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("rohan.telemetry")


def emit_metric(event: str, **fields: Any) -> None:
    """Emit a structured metric event as one JSON log line."""
    payload: dict[str, Any] = {
        "event": event,
        "ts": datetime.now(UTC).isoformat(),
        **fields,
    }
    logger.info("telemetry %s", json.dumps(payload, default=str, sort_keys=True))
