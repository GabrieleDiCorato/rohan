import json
from unittest.mock import patch

from rohan.llm.telemetry import emit_metric, refinement_telemetry_context


def test_emit_metric_logs_json_payload():
    with patch("rohan.llm.telemetry.logger.info") as mock_info:
        emit_metric(
            "writer_success",
            component="rohan.llm.refinement",
            run_id="run-123",
            iteration=2,
            attempt=1,
        )

    mock_info.assert_called_once()
    _, payload = mock_info.call_args.args
    assert payload.startswith("{")
    parsed = json.loads(payload)
    assert parsed["schema_version"] == 1
    assert parsed["component"] == "rohan.llm.refinement"
    assert parsed["kind"] == "metric"
    assert parsed["event"] == "writer_success"
    assert parsed["run_id"] == "run-123"
    assert parsed["iteration"] == 2
    assert parsed["attempt"] == 1
    assert "ts" in parsed


def test_emit_metric_normalizes_non_primitive_fields():
    class _Value:
        def __str__(self):
            return "value-object"

    with patch("rohan.llm.telemetry.logger.info") as mock_info:
        emit_metric("sample_event", component="rohan.test", nested=_Value())

    _, payload = mock_info.call_args.args
    parsed = json.loads(payload)
    assert parsed["nested"] == "value-object"


def test_refinement_telemetry_context_applies_defaults():
    with patch("rohan.llm.telemetry.logger.info") as mock_info:
        ctx = refinement_telemetry_context("run-xyz", iteration=7)
        ctx.emit("writer_success", attempt=2)

    _, payload = mock_info.call_args.args
    parsed = json.loads(payload)
    assert parsed["component"] == "rohan.llm.refinement"
    assert parsed["run_id"] == "run-xyz"
    assert parsed["iteration"] == 7
    assert parsed["attempt"] == 2
