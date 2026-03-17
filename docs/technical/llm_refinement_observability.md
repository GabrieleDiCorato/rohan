# LLM Refinement Observability and Operating Assumptions

This document records the current operating assumptions for the refinement loop and the observability choices built on top of LangGraph.

## Agreed Operating Assumptions

- The refinement loop is currently single-user. We do not need multi-tenant cache isolation or per-user trace partitioning yet.
- Feature flags are deploy-time static. They are read from environment-backed settings and cached for the process lifetime.
- We want stronger telemetry and observability, but we do not want to replace LangGraph-native tracing with a parallel bespoke platform.

## Observability Approach

LangGraph is compatible with existing tracing solutions such as LangSmith. ROHAN keeps that compatibility and adds a small local telemetry layer for refinement-specific events.

The local telemetry is intended to complement graph tracing, not replace it:

- LangGraph and LangSmith are still the right tools for graph execution traces, node timing, and step-by-step debugging.
- ROHAN structured telemetry is used for stable product and engineering signals such as writer retries, explainer fallback tier usage, aggregator outcomes, and baseline-cache reuse.
- Telemetry events are emitted as JSON log lines so they can be forwarded later into standard log/metrics pipelines without changing node logic again.

## Current Telemetry Contract

Structured telemetry events now include a consistent envelope:

- `schema_version`: event schema version for downstream parsing.
- `component`: logical producer such as `rohan.llm.refinement` or `rohan.simulation`.
- `event`: stable event name.
- `ts`: UTC timestamp.
- `run_id`: refinement run correlation id when the event originates from graph state.

## How To Use Telemetry

### 1) Toggle telemetry on or off

Telemetry emission is controlled by the deploy-time feature flag:

```env
FEATURE_LLM_TELEMETRY_V1=true
```

### 2) Emit events from refinement nodes

Use the refinement context helper to avoid repeating `component`, `run_id`, and `iteration` boilerplate:

```python
from rohan.llm.telemetry import refinement_telemetry_context

ctx = refinement_telemetry_context(run_id="abc123", iteration=4)
ctx.emit("writer_success", attempt=1, max_retries=3)
```

### 3) Emit events from non-graph services

Use `emit_metric` directly and provide a stable component name:

```python
from rohan.llm.telemetry import emit_metric

emit_metric(
	"baseline_cache_hit",
	component="rohan.simulation",
	cache_key="3ad1f7b8d2a1",
)
```

### 4) Parse telemetry downstream

Events are emitted as single-line JSON messages prefixed by `telemetry ` in logger output. Filter by `schema_version`, `component`, and `event` in your log pipeline.

## Current Rollout Behavior

- `FEATURE_LLM_EXPLAINER_TIERS_V1` controls whether the deterministic template explainer tier is enabled after ReAct and structured fallback failures.
- `FEATURE_EXPLICIT_TERMINAL_REASONS_V1` controls whether terminal reason metadata is populated in graph state.
- `FEATURE_BASELINE_CACHE_V1` controls baseline cache reuse.
- `FEATURE_LLM_TELEMETRY_V1` controls refinement telemetry emission.

## Follow-up Work

The current implementation intentionally stops short of replacing the logging stack or introducing external observability dependencies. The next sensible step, if deeper operational visibility is needed, is to wire these structured events into the existing logging pipeline or a LangSmith/OpenTelemetry sink rather than build a second tracing system.
