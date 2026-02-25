# Agentic Simulation Framework - Implementation Plan

This document outlines the remaining planned features and future extensions for the `abides-rohan` framework. For the current technical architecture and implemented features, please refer to [technical_architecture.md](technical_architecture.md).

## Phase 3 - Docker Sandbox
**Status:** Deferred.

*   **Docker Container Integration:** Implement Docker container isolation for executing untrusted LLM-generated code securely. This will replace the current in-process restricted `exec()` environment to provide robust CPU/memory limits, filesystem isolation, and network isolation.

## Phase 4 - Production Features
**Status:** Deferred.

*   **Web UI:** Develop a full-featured web application.
*   **Dashboards:** Create comprehensive dashboards for monitoring simulation runs and agent performance.
*   **Leaderboards:** Implement leaderboards to rank generated strategies based on KPIs (PnL, Sharpe, Max Drawdown).

## Future Extensions

*   **Parallel Scenarios:** Support for running multiple scenarios in parallel to speed up the evaluation phase.
*   **Agent-Driven Scenario Selection:** Allow the LLM to autonomously select or generate new scenarios based on the weaknesses identified in previous iterations.
*   **Advanced Tool Suite:** Expand the tools available to the Explainer agent for deeper market microstructure analysis.
*   **Persistent Checkpointing:** Enhance LangGraph state persistence to allow pausing, resuming, and rewinding refinement sessions.
*   **Advanced Order Management API:**
    *   `modify_order()` — Change price/quantity of existing orders without canceling.
    *   `partial_cancel_order()` — Reduce order quantity while keeping the remainder active.
    *   *Note:* Not exposing these from ABIDES in the current implementation to avoid complexity. Can be added later without breaking changes to the `StrategicAgent` protocol.
