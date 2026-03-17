# LLM Refinement Resilience Plan

This plan targets six reliability and effectiveness issues observed in production refinement runs:

1. Writer structured-output misses causing stalled iterations.
2. Weak diagnostics when function-calling parsing fails.
3. Explainer recursion-limit failures.
4. Retry-budget semantics that are hard to reason about in logs.
5. Repeated baseline simulation work per iteration.
6. Limited observability for engineering and product decisions.

## Delivery Approach

We execute one phase at a time with strict validation gates:

- Documentation updated for any behavior/config change.
- `pre-commit run --all-files` passes.
- Simplicity and elegance review: minimal branching, clear defaults, predictable failure behavior.

## Phases

### Phase 1: Writer Reliability Guardrails

Goal: maximize strategy-generation success under schema/tool-call instability.

Scope:

- Move writer retry policy to configuration.
- Add optional fallback model for final writer retry.
- Improve structured-output parse diagnostics.
- Preserve a known-good code snapshot when writer exhausts retries.

Design:

1. Add new `LLMSettings` fields:
   - `writer_max_retries` (default: 3)
   - `writer_retry_prompt_trim` (default: true)
   - `writer_fallback_model` (default: null)
2. In `writer_node`:
   - read retry policy from settings instead of hardcoded local constants,
   - optionally use `writer_fallback_model` on final attempt,
   - on full retry exhaustion, preserve `best_code` (or current code) instead of hard-resetting to `None`.
3. In `get_structured_model` parsing path:
   - log parse-error class/message,
   - include tool-call count and a compact raw-content preview.

Success criteria:

- Fewer terminal failures caused by writer schema misses.
- Clear logs for parser failures and model behavior.
- No regressions in existing writer/validator/explainer flow.

Validation checklist:

- [x] Update docs for new environment variables.
- [x] `pre-commit run --all-files` passes.
- [x] Reviewer confirms fallback path is simple, deterministic, and reversible.

### Phase 1 Detailed Plan and Execution Notes

Detailed plan:

1. Externalize writer retry knobs in `LLMSettings` to remove hidden constants.
2. Keep existing structured-output contract (`T | None`) for compatibility, but emit richer diagnostics in the parser path.
3. Add controlled final-attempt model fallback in writer node.
4. Replace destructive `current_code=None` behavior with preservation of a known-good code snapshot (`best_code` preferred, otherwise current code).
5. Keep validator in the loop after fallback to maintain safety.

Executed changes:

- Added `writer_max_retries`, `writer_retry_prompt_trim`, and `writer_fallback_model` to `LLMSettings`.
- Updated writer loop to use settings-driven retry behavior and optional final-attempt model override.
- Enhanced parse-failure logs with error type, tool-call count, and compact raw preview.
- Implemented resilient writer exhaustion path that preserves known-good code and surfaces explicit validation error context.
- Updated README with operational environment knobs.
- Validation complete: targeted node/settings tests passed and `pre-commit run --all-files` passed.

### Phase 2: Explainer Recursion Control

Move ReAct recursion/tool budget to config and add a tiered fallback ladder.

Validation checklist:

- [x] Settings added for recursion and tool-budget controls.
- [x] Tiered fallback implemented (ReAct -> structured -> deterministic template).
- [x] Tests added/updated for recursion config usage and fallback behavior.
- [x] `pre-commit run --all-files` passes after Phase 2 changes.

Execution notes:

- Added `explainer_react_recursion_limit` and `explainer_max_tool_calls` to `LLMSettings`.
- Replaced hardcoded explainer recursion constant with settings-driven derived limit.
- Added deterministic template fallback to guarantee a usable `ScenarioExplanation`.

### Phase 3: Retry/Termination Semantics

Make retry counters and terminal status reasons explicit and consistent.

Validation checklist:

- [x] Explicit terminal metadata fields added to refinement state.
- [x] Validation-budget exhaustion now routes through a terminalization node.
- [x] Aggregator sets terminal reason for converged/plateau/max-iterations exits.
- [x] `pre-commit run --all-files` passes after Phase 3 changes.

Execution notes:

- Added `terminal_reason`, `terminal_iteration`, and `terminal_context` to graph state.
- Added `terminalize_validation_failure_node` in the graph to make validation-stop reasons explicit.
- Updated aggregator to emit terminal reasons for deterministic stop conditions.
- Added unit tests for terminalization node and aggregator terminal reason fields.

### Phase 4: Baseline Reuse Caching

Cache deterministic baseline runs to cut iteration latency.

### Phase 5: Observability & Product KPIs

Emit structured node-level metrics for quality/cost/time decisions.

### Phase 6: Rollout Hardening

Feature flags, integration benchmarks, and staged enablement.

## Execution Log

- 2026-03-17: Phase 1 started.
