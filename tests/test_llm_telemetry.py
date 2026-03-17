import json
from unittest.mock import patch

from rohan.llm.telemetry import emit_metric


def test_emit_metric_logs_json_payload():
    with patch("rohan.llm.telemetry.logger.info") as mock_info:
        emit_metric("writer_success", iteration=2, attempt=1)

    mock_info.assert_called_once()
    _, payload = mock_info.call_args.args
    assert payload.startswith("{")
    parsed = json.loads(payload)
    assert parsed["event"] == "writer_success"
    assert parsed["iteration"] == 2
    assert parsed["attempt"] == 1
    assert "ts" in parsed
