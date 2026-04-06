# ROHAN — Target Architecture: Agentic Strategy Validation Platform

> **Version:** 1.0 draft — April 2026
> **Audience:** Academic review panel, technical stakeholders, cost/benefit analysis input.
> **Status:** Architecture proposal. Maps the PoC (v0.3.1) to a cloud-native production system.

---

## 1. Executive Summary

ROHAN evolves from an autonomous strategy-*generation* loop into an agentic
strategy-*validation* platform. The system accepts user-supplied trading
strategies, applies AI-driven adversarial scenario design, runs high-fidelity
market simulations with full microstructure observability, and produces
forensic-quality validation reports.

**Target user:** Risk management offices and model validation teams within
quantitative investment firms. The platform helps them comply with SR 11-7
(OCC/FRB model risk management guidance), MiFID II Article 17 (algorithmic
trading stress testing), and EBA guidelines on model risk by providing
reproducible, auditable, and explainable strategy stress-testing.

### Core value proposition

| Dimension | Description |
|---|---|
| **Forensic observability** | Every fill, every order lifecycle event, every L2 snapshot — captured with nanosecond precision. The AI explains *why* the strategy failed, not just *that* it failed. |
| **Adversarial intelligence** | AI designs stress scenarios targeting the specific weaknesses of each strategy, going beyond standard regulatory shock scenarios. |
| **Audit-ready output** | Immutable, versioned reports with full traceability from findings back to simulation data, scenario configuration, and strategy code. |
| **Reproducibility** | Deterministic simulations (SHA-256 seeded) guarantee byte-identical reruns for any auditor challenge. |

---

## 2. System Context

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│                          ROHAN Platform                              │
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │
│  │ Strategy  │  │Scenario  │  │Simulation│  │  AI Analysis &   │    │
│  │ Ingestion │→ │ Planning │→ │ Engine   │→ │  Report Engine   │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘    │
│       ↑                                              │               │
└───────┼──────────────────────────────────────────────┼───────────────┘
        │                                              │
        │                                              ▼
┌───────┴──────┐                              ┌─────────────────┐
│  Consumers   │                              │   Deliverables  │
│              │                              │                 │
│ • Web UI     │                              │ • Interactive   │
│ • REST API   │                              │   dashboard     │
│ • CI/CD hook │                              │ • PDF report    │
│ • Scheduler  │                              │ • JSON/API      │
└──────────────┘                              │ • Email digest  │
                                              └─────────────────┘
```

### External actors

| Actor | Interaction | Protocol |
|---|---|---|
| **Quant analyst** | Submits strategy, triggers on-demand validation, explores results | Web UI, REST API |
| **CI/CD pipeline** | Triggers validation on strategy code change, gates deployment | REST API (webhook) |
| **Scheduler** | Triggers periodic re-validation of production strategies | Cron → REST API |
| **Risk committee** | Reviews reports, manages mandatory scenario registry | Web UI (admin) |
| **Auditor** | Retrieves historical reports, verifies reproducibility | PDF export, REST API |
| **Downstream risk systems** | Consumes structured results for VaR/PnL aggregation | JSON API, event stream |

---

## 3. Validation Pipeline (Core Data Flow)

The pipeline is a **linear DAG** — no refinement loop. Each strategy goes
through five stages exactly once per validation run.

```
┌─────────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ INGEST  │───→│ ANALYZE │───→│  PLAN    │───→│ EXECUTE  │───→│ EXPLAIN  │
│         │    │         │    │          │    │          │    │          │
│Strategy │    │AI reads │    │AI designs│    │Parallel  │    │AI inter- │
│code +   │    │code,    │    │adversar- │    │simulation│    │prets per │
│metadata │    │infers   │    │ial +     │    │across all│    │scenario +│
│+ stated │    │intent,  │    │merge     │    │scenarios │    │aggregate │
│objective│    │classif- │    │mandatory │    │          │    │report    │
│         │    │ies type │    │scenarios │    │          │    │          │
└─────────┘    └─────────┘    └──────────┘    └──────────┘    └──────────┘
     │              │              │                │               │
     ▼              ▼              ▼                ▼               ▼
 strategy_id   StrategyProfile  ScenarioPlan    ScenarioResult[]  ValidationReport
 (DB record)   (Pydantic)       (versioned)     (per scenario)    (immutable)
```

### 3.1 Stage 1 — Ingest

**Input:** Strategy Python code + metadata (asset class, style, risk limits) + user-declared objective.

**Operations:**
1. AST safety validation (existing `strategy_validator.py` sandbox).
2. Compatibility check: static analysis flags constructs that cannot be faithfully translated to discrete-time simulation (network I/O, wall-clock timers, threading). The system rejects with clear explanation or proposes adapter rewrites.
3. Strategy adapter layer translates real-time idioms → discrete-event callbacks (see §5.1).
4. Persists strategy record with code hash, version, and ingestion timestamp.
5. If a `StrategicAgent`-compliant strategy is provided, skip adaptation and inject directly.

**Output:** `Strategy` record (DB) + adapted `StrategicAgent` code ready for simulation.

### 3.2 Stage 2 — Analyze

**Input:** Strategy code, user-declared objective, strategy metadata.

**Operations:**
1. **Code comprehension:** LLM reads the adapted strategy source code and produces a `StrategyProfile`:
   - Inferred style classification (market-making, momentum, stat-arb, execution, etc.)
   - Identified risk dimensions (inventory risk, adverse selection, crowding, latency sensitivity, etc.)
   - Behavioral summary: what triggers entry/exit, how positions are sized, how risk is managed
2. **Intent cross-validation:** Compares LLM-inferred intent against user-declared objective. Mismatches become findings in the final report (e.g., "User describes a market-making strategy, but the code exhibits trend-following behavior in the `on_tick` callback").
3. **Weakness hypothesis:** LLM proposes 3–5 weakness axes specific to this strategy, which feed Stage 3.

**Output:** `StrategyProfile` (Pydantic model, persisted on the validation run).

### 3.3 Stage 3 — Plan Scenarios

**Input:** `StrategyProfile`, mandatory scenario registry, user-selected scenarios (optional).

**Operations:**
1. **Load mandatory scenarios** from the three-tier registry:
   - **Regulatory tier** (global): flash crash, liquidity drought, extreme volatility, fat-finger, halt/resume
   - **Strategy-type tier**: scenarios matched to the inferred style classification
   - **Organization tier**: firm-specific stress scenarios configured by the risk committee
2. **AI adversarial planning** (existing `planner.py` with 3-tier fallback):
   - ReAct agent with hasufel template-discovery tools proposes 3–5 adversarial scenarios targeting the weakness axes from Stage 2
   - Each adversarial scenario gets a name, rationale, and concrete config override
3. **Scenario set assembly:** Merge mandatory + user-selected + adversarial. Deduplicate. Assign deterministic seeds (SHA-256 of scenario name + validation run ID).
4. **Version-stamp** the assembled scenario set. Once referenced by an execution, the scenario version is immutable.

**Output:** `ScenarioPlan` — ordered list of `ScenarioConfig` records with seeds, rationale, and provenance (mandatory vs. adversarial vs. user-selected).

### 3.4 Stage 4 — Execute

**Input:** Adapted strategy code, `ScenarioPlan`.

**Operations:**
1. **Baseline runs** (no strategy injected) for each scenario — establishes market microstructure reference values. Cached when scenario config + seed match a prior baseline (existing `baseline_cache`).
2. **Strategy runs** — strategy injected into each scenario via `StrategicAgent` protocol and `@register_agent("rohan_strategy")`.
3. **Parallel execution** — scenarios execute concurrently across simulation workers (see §6.2). Each worker uses hasufel's `run_simulation()` or `run_batch()`.
4. **Rich data capture** per scenario:
   - `HasufelOutput` → `compute_rich_analysis()` → `RichAnalysisBundle` (fills, PnL curve, inventory trajectory, multi-window adverse selection, counterparty breakdown, order lifecycle, L2 snapshots)
   - 6 charts generated (Price, Spread, Volume, PnL, Inventory, Fill Scatter)
   - Deterministic 6-axis scoring (Profitability, Risk, Volatility Impact, Spread Impact, Liquidity Impact, Execution Quality)
5. **Serialize** `RichAnalysisBundle` to JSON for checkpoint safety.

**Output:** `ScenarioResult[]` — one per scenario, each carrying scores, metrics, `rich_analysis_json`, and charts.

### 3.5 Stage 5 — Explain & Report

**Input:** `StrategyProfile`, `ScenarioPlan`, `ScenarioResult[]`.

**Operations:**
1. **Per-scenario explanation** — ReAct Explainer agent with 8 investigation tools (existing `tools.py`):
   - Drills into fills, PnL trajectory, order lifecycle, L2 snapshots
   - Produces `ScenarioExplanation` with findings, root causes, and recommendations
   - Fallback to structured-output single call on agent failure
2. **Cross-scenario aggregation** — Aggregator synthesizes findings across all scenarios:
   - Identifies systematic weaknesses (patterns that appear across multiple scenarios)
   - Identifies scenario-specific vulnerabilities
   - Ranks findings by severity and frequency
   - Produces actionable improvement recommendations with code-level specificity
3. **Report assembly:**
   - Executive summary (pass/fail with confidence, headline risks)
   - Strategy profile (AI-inferred intent vs. declared intent, behavioral summary)
   - Per-scenario drill-down (scores, charts, findings, root causes)
   - Aggregated findings (cross-scenario patterns, systematic weaknesses)
   - Improvement recommendations (ranked, code-referencing)
   - Reproducibility attestation (run ID, seeds, config versions, model versions)
4. **Report persistence** — immutable, stamped with unique `validation_run_id`. Retained per configurable policy (min. 5 years for regulatory compliance).
5. **Multi-format rendering** — Interactive UI, PDF, structured JSON, email digest.

**Output:** `ValidationReport` — the primary deliverable.

---

## 4. LangGraph Orchestration (Revised)

The PoC's cyclic refinement graph becomes a **linear validation DAG**.
The Writer node is removed. The loop is eliminated.

```
                    ┌─────────────────────────────────────────────┐
                    │          LangGraph Validation DAG           │
                    │                                             │
START ──→ [Analyzer] ──→ [Planner] ──→ [Executor] ──→ fan-out   │
                                            │                     │
                                   ┌────────┼────────┐           │
                                   ▼        ▼        ▼           │
                              [Explainer][Explainer][Explainer]   │
                              (Scenario1)(Scenario2)(ScenarioN)   │
                                   │        │        │           │
                                   └────────┼────────┘           │
                                            ▼                     │
                                      [Aggregator]                │
                                            │                     │
                                            ▼                     │
                                     [ReportBuilder]              │
                                            │                     │
                                            ▼                     │
                                          END                     │
                    └─────────────────────────────────────────────┘
```

### 4.1 State Schema (revised `ValidationState`)

```python
class ValidationState(TypedDict):
    # ── Inputs (set once at start) ──
    strategy_code: str              # User-supplied strategy source
    strategy_metadata: dict         # Asset class, style, risk limits
    declared_objective: str         # User's description of intent
    validation_run_id: str          # UUID, immutable per run
    trigger: str                    # "on_demand" | "scheduled" | "ci"

    # ── Stage 2: Analyze ──
    strategy_profile: dict          # StrategyProfile serialized
    intent_mismatch_findings: list  # Cross-validation results

    # ── Stage 3: Plan ──
    scenario_plan: list             # ScenarioConfig[] with provenance
    scenario_plan_reasoning: str    # AI rationale for adversarial choices

    # ── Stage 4: Execute (map-reduce via Send) ──
    scenario_results: Annotated[list, _concat_or_clear]
    baseline_results: Annotated[list, _concat_or_clear]

    # ── Stage 5: Explain ──
    explanations: Annotated[list, _concat_or_clear]
    aggregated_report: dict         # Cross-scenario synthesis
    improvement_recommendations: list

    # ── Metadata ──
    feature_flags: FeatureFlagState
    terminal_reason: str            # "completed" | "execution_failed" | ...
    messages: Annotated[list, add_messages]  # For agent tool calls
```

### 4.2 Node Mapping (PoC → Target)

| PoC Node | Target Node | Change |
|---|---|---|
| `writer_node` | **Removed** | Strategy is user-supplied, not LLM-generated |
| `validator_node` | `ingest_node` | Still runs AST sandbox; adds adapter translation |
| — | `analyzer_node` | **New**: LLM reads strategy, produces StrategyProfile |
| — | `planner_node` | Promoted from pre-graph call to first-class graph node |
| `process_scenario_node` | `executor_node` | Unchanged: fan-out via `Send`, runs simulations |
| Explainer (inside scenario) | `explainer_node` | Unchanged: ReAct agent per scenario |
| `aggregator_node` | `aggregator_node` | Simplified: no convergence/rollback logic; pure synthesis |
| — | `report_builder_node` | **New**: assembles multi-format report |

### 4.3 Key Simplifications

1. **No loop edges.** Aggregator → END, always. This simplifies checkpointing, makes cost predictable, and eliminates convergence/plateau/rollback logic.
2. **No code generation.** The AST sandbox still validates user code, but there are no retry/feedback cycles for LLM-generated code.
3. **Deterministic scoring is unchanged.** The 6-axis scoring in `scoring.py` is preserved as-is — it's already LLM-free.
4. **Tool-equipped explainer is unchanged.** The ReAct agent with 8 investigation tools is the core analytical investment and carries forward directly.

---

## 5. Strategy Decoupling Layer

### 5.1 The Discrete-Time Problem

ABIDES is a discrete-event simulator. Time advances message-by-message through
a priority queue — there is no wall clock. Real-time strategies may rely on:

- Wall-clock timers (`time.sleep`, `asyncio` timers, scheduler-based rebalancing)
- Continuous subscriptions (WebSocket feeds, streaming APIs)
- System clock queries (`time.time()`, `datetime.now()`)

These constructs have no meaning in a discrete-event simulation.

### 5.2 Strategy Adapter Layer

```
┌─────────────────────────────────────┐
│         Strategy Code               │  ← User's real-time strategy
│  (may use timers, subscriptions)    │
└────────────────┬────────────────────┘
                 │
    ┌────────────▼────────────────┐
    │   Strategy Adapter Layer    │  ← Translates time model
    │  • Timer → scheduled wakeup │
    │  • Subscription → on_tick   │
    │  • Real clock → sim clock   │
    │  • Incompatible → reject    │
    └────────────┬────────────────┘
                 │
    ┌────────────▼────────────────┐
    │   StrategicAgent Protocol   │  ← Existing event-driven contract
    │   (on_tick, on_market_data, │
    │    on_order_update, etc.)   │
    └────────────┬────────────────┘
                 │
    ┌────────────▼────────────────┐
    │   ABIDES/Hasufel Discrete   │
    │   Event Engine              │
    └─────────────────────────────┘
```

**Adaptation rules:**

| Real-Time Construct | Discrete-Time Translation | Fidelity Impact |
|---|---|---|
| `time.sleep(n)` / periodic timer | `setWakeup(current_time + n_ns)` — scheduled wakeup | **Exact**: timer fires at the correct simulation time |
| Market data subscription callback | `on_market_data()` callback | **Exact**: same event semantics |
| `time.time()` / `datetime.now()` | Injected simulation clock (`state.timestamp_ns`) | **Exact**: returns simulation time |
| `asyncio` event loop | **Rejected** — cannot faithfully translate | Strategy must be refactored |
| Network I/O, file I/O, threading | **Rejected** — side effects incompatible with sim | Strategy must be refactored |

**Key design decision:** The adapter is also a validation artifact. The adaptation
log documents exactly what was translated and how, making it auditable. If the
system rejects a strategy, the rejection reason is itself a finding.

### 5.3 Direct Protocol Support

Strategies that directly implement the `StrategicAgent` protocol bypass the
adapter entirely. This is the high-fidelity path — zero translation, zero
fidelity loss. The PoC already supports this path end-to-end.

### 5.4 Ingestion Modes

| Mode | Input | Adapter Path | Use Case |
|---|---|---|---|
| **Protocol-native** | Python class implementing `StrategicAgent` | None (direct injection) | Strategies built for ROHAN |
| **Adapted** | Python code with real-time constructs | Full adapter translation | Existing production strategies |
| **Rejected** | Code with incompatible constructs | Rejection with explanation | Strategies requiring refactoring |

---

## 6. Component Architecture

### 6.1 Service Decomposition

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Cloud Infrastructure                         │
│                                                                     │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────┐    │
│  │  API Gateway │    │  Validation     │    │  Simulation      │    │
│  │  (FastAPI)   │───→│  Orchestrator   │───→│  Worker Pool     │    │
│  │             │    │  (LangGraph)    │    │  (hasufel)       │    │
│  │  • Auth     │    │                 │    │                  │    │
│  │  • Rate     │    │  • DAG exec     │    │  • N parallel    │    │
│  │    limiting │    │  • Checkpointing│    │    workers       │    │
│  │  • Routing  │    │  • LLM calls    │    │  • CPU-bound     │    │
│  └──────┬──────┘    └────────┬────────┘    └──────────────────┘    │
│         │                    │                                      │
│  ┌──────▼──────┐    ┌────────▼────────┐    ┌──────────────────┐    │
│  │  Web UI     │    │  PostgreSQL     │    │  Object Storage  │    │
│  │ (Streamlit  │    │                 │    │  (S3 / GCS)      │    │
│  │  or React)  │    │  • Sessions     │    │                  │    │
│  │             │    │  • Strategies   │    │  • Reports (PDF) │    │
│  │  • Dashboard│    │  • Scenarios    │    │  • Charts (PNG)  │    │
│  │  • Reports  │    │  • Results      │    │  • Rich analysis │    │
│  │  • Admin    │    │  • Audit trail  │    │  • Strategy code │    │
│  └─────────────┘    └─────────────────┘    └──────────────────┘    │
│                                                                     │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────┐    │
│  │  Task Queue │    │  Observability  │    │  Scenario        │    │
│  │  (Celery /  │    │                 │    │  Registry        │    │
│  │   Cloud     │    │  • OpenTelemetry│    │                  │    │
│  │   Tasks)    │    │  • Metrics      │    │  • Versioned     │    │
│  │             │    │  • Traces       │    │  • 3-tier        │    │
│  │  • Sim jobs │    │  • Dashboards   │    │  • Audit trail   │    │
│  │  • Reports  │    │  • Alerts       │    │  • Immutable     │    │
│  │  • Scheduled│    │                 │    │    once           │    │
│  └─────────────┘    └─────────────────┘    │    referenced    │    │
│                                             └──────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Component Responsibilities

#### API Gateway (FastAPI)

| Responsibility | Detail |
|---|---|
| Authentication | JWT/OAuth2 — user identity for session isolation and audit trail |
| Authorization | RBAC: analyst (submit/view), risk manager (manage scenarios), admin (platform config) |
| Rate limiting | Per-user and per-organization LLM token budgets |
| API versioning | `/v1/validations`, `/v1/strategies`, `/v1/scenarios`, `/v1/reports` |
| Webhook ingestion | CI/CD triggers on strategy code change |

**Key endpoints:**

```
POST   /v1/validations              # Submit a validation run
GET    /v1/validations/{run_id}     # Poll status / retrieve results
GET    /v1/validations/{run_id}/report?format=pdf|json
POST   /v1/strategies               # Upload strategy code + metadata
GET    /v1/strategies/{id}/versions  # Version history
GET    /v1/scenarios                 # List scenario registry
POST   /v1/scenarios                # Add/update scenario (admin)
GET    /v1/comparisons              # Cross-run comparison view
POST   /v1/validations/schedule     # Create scheduled validation
```

#### Validation Orchestrator (LangGraph)

The orchestrator is the heart of the system. It runs the linear DAG from §4
as a stateful LangGraph execution with checkpointing.

| Responsibility | Detail |
|---|---|
| DAG execution | Analyzer → Planner → Executor (fan-out) → Explainer (fan-out) → Aggregator → ReportBuilder |
| LLM management | Model factory (existing `factory.py`), provider routing, token tracking |
| Checkpointing | LangGraph native checkpointing to PostgreSQL — resume from any node on failure |
| Simulation dispatch | Sends scenario execution jobs to the task queue / worker pool |
| Result collection | Gathers `ScenarioResult[]` from workers, feeds to Explainer |

**Deployment:** Runs as a long-lived service (not ephemeral). One orchestrator
instance can handle multiple concurrent validation runs via async LangGraph
execution. Horizontal scaling via additional orchestrator instances if needed.

#### Simulation Worker Pool

| Responsibility | Detail |
|---|---|
| Simulation execution | Runs hasufel `run_simulation()` with injected strategy |
| Baseline caching | Reuses baseline results when config + seed match (existing cache logic) |
| Rich data capture | `compute_rich_analysis()` → `RichAnalysisBundle` → JSON serialization |
| Chart generation | 6 Matplotlib charts per scenario |
| Resource isolation | Each worker is a separate process; strategies run in sandbox |

**Deployment:** Pool of N workers (auto-scaled by task queue depth). Each worker
is CPU-bound (simulation is compute-intensive, no GPU needed). Workers are
stateless — all state flows through the task queue and result store.

**Scaling model:** A validation run with 12 scenarios and N workers completes
the execution stage in `ceil(12 / N)` batches. With 4 workers, 3 batches.
With 12 workers, 1 batch. Workers are the primary cost lever.

#### PostgreSQL

Central relational store. Evolved schema from the PoC:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Strategy    │     │  Validation  │     │  Scenario    │
│              │     │  Run         │     │  Registry    │
│ • id (UUID)  │←───┐│ • run_id     │     │              │
│ • code_hash  │    ││ • strategy_id│     │ • id (UUID)  │
│ • version    │    ││ • trigger    │  ┌──│ • tier       │
│ • metadata   │    ││ • status     │  │  │ • name       │
│ • created_at │    ││ • profile    │  │  │ • config     │
│ • created_by │    ││ • created_at │  │  │ • version    │
└──────────────┘    │└──────┬───────┘  │  │ • approved_by│
                    │       │          │  │ • created_at │
                    │       ▼          │  └──────────────┘
                    │┌──────────────┐  │
                    ││ Scenario Run │  │
                    ││              │  │
                    ││ • run_id(FK) │  │
                    ││ • scenario_id│──┘
                    ││ • scores     │
                    ││ • metrics    │
                    ││ • duration   │
                    ││ • error      │
                    │└──────┬───────┘
                    │       │
                    │       ▼
                    │┌──────────────┐     ┌──────────────┐
                    ││ Explanation  │     │   Artifact   │
                    ││              │     │              │
                    ││ • findings   │     │ • type (chart│
                    ││ • root_causes│     │   /report/   │
                    ││ • recomm.    │     │   analysis)  │
                    │└──────────────┘     │ • storage_uri│
                    │                     │ • content_hash│
                    │                     └──────────────┘
          ┌─────────┘
          │
          ▼
   ┌──────────────┐
   │  Report      │
   │              │
   │ • run_id(FK) │
   │ • format     │
   │ • storage_uri│
   │ • generated  │
   │ • immutable  │
   └──────────────┘
```

**Key differences from PoC schema:**
- `Strategy` table (new) — versioned strategy registry, decoupled from validation runs
- `ScenarioRegistry` table (new) — versioned, tiered, approval-tracked mandatory scenarios
- `Explanation` table (new) — structured findings, separated from scenario run metrics
- `Report` table (new) — immutable report records with object storage references
- `Artifact` table — now references object storage URIs instead of inline base64
- `Session` → `ValidationRun` — renamed to reflect the new domain language
- `Iteration` → removed — no refinement loop

#### Object Storage (S3 / GCS)

| Content | Format | Retention |
|---|---|---|
| PDF reports | PDF | 5+ years (regulatory) |
| Charts | PNG | Lifecycle of validation run |
| Rich analysis bundles | JSON | Lifecycle of validation run |
| Strategy code versions | Python source | Indefinite |
| Scenario registry snapshots | JSON | Indefinite |

**Critical:** Object storage provides immutability guarantees (S3 Object Lock /
GCS retention policies) for audit-grade report retention.

#### Task Queue

Decouples the orchestrator from simulation workers. Options:

| Option | Pros | Cons |
|---|---|---|
| **Celery + Redis** | Mature, Python-native, rich monitoring (Flower) | Self-managed Redis, operational overhead |
| **Cloud Tasks** (GCP) / **SQS + Lambda** (AWS) | Fully managed, auto-scales to zero | Vendor lock-in, cold start latency for workers |
| **Temporal** | Durable execution, built-in retry/timeout, visibility | Heavier infra, learning curve |

**Recommendation:** Start with Celery + Redis for portability and Python ecosystem
compatibility. Migrate to managed cloud tasks if operational burden justifies it.

#### Observability Stack

| Layer | Tool | Purpose |
|---|---|---|
| **Tracing** | OpenTelemetry → Jaeger/Tempo | End-to-end request traces through DAG, LLM calls, simulations |
| **Metrics** | Prometheus | Simulation duration, LLM token usage, queue depth, error rates |
| **Logs** | Structured JSON → Loki/CloudWatch | Existing `rohan.telemetry` structured events + service logs |
| **Dashboards** | Grafana | Operational health, cost tracking, SLA monitoring |
| **Alerts** | Grafana Alerting | Failed runs, SLA breaches, budget overruns |

**LLM cost attribution:** Every LLM call is tagged with `validation_run_id`,
`node_name`, and `user_id`. Token counts (prompt + completion) are emitted as
OpenTelemetry metrics and persisted. This enables per-run, per-user, and per-node
cost breakdowns.

---

## 7. Mandatory Scenario Registry

### 7.1 Three-Tier Governance Model

```
┌─────────────────────────────────────────────────────────┐
│                   Scenario Registry                      │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Tier 1: REGULATORY (global, compliance-owned)  │    │
│  │                                                  │    │
│  │  • Flash crash (price drops 5% in 30s)          │    │
│  │  • Liquidity drought (bid/ask qty drops 90%)    │    │
│  │  • Extreme volatility (3× normal σ)             │    │
│  │  • Fat-finger event (10× normal volume spike)   │    │
│  │  • Market halt & resume                         │    │
│  │                                                  │    │
│  │  Change policy: Compliance sign-off required     │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Tier 2: STRATEGY-TYPE (per style class)        │    │
│  │                                                  │    │
│  │  Market Making:                                  │    │
│  │    • Toxic flow (high informed trader ratio)     │    │
│  │    • Inventory squeeze (directional bias)        │    │
│  │    • Spread compression (tight competition)      │    │
│  │                                                  │    │
│  │  Momentum:                                       │    │
│  │    • Regime reversal (trend breakdown)           │    │
│  │    • Crowded exit (correlated unwind)            │    │
│  │    • Low-volume drift (false signal)             │    │
│  │                                                  │    │
│  │  Change policy: Risk desk approval per class     │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Tier 3: ORGANIZATION (firm-specific)           │    │
│  │                                                  │    │
│  │  • Correlated position unwinds                   │    │
│  │  • Counterparty default proxy                    │    │
│  │  • Firm-specific historical stress dates         │    │
│  │                                                  │    │
│  │  Change policy: Risk committee approval          │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  Version control: Every change creates a new version.   │
│  Referenced versions are immutable.                      │
│  Full audit trail: who, when, why, approval chain.      │
└─────────────────────────────────────────────────────────┘
```

### 7.2 Scenario Configuration Model

Each scenario in the registry maps to a hasufel template + overlay composition:

```python
class RegisteredScenario(BaseModel):
    """A scenario in the mandatory registry."""

    scenario_id: UUID
    tier: Literal["regulatory", "strategy_type", "organization"]
    name: str                          # e.g., "flash_crash_5pct_30s"
    display_name: str                  # e.g., "Flash Crash (5% / 30s)"
    description: str
    strategy_types: list[str] | None   # None = applies to all
    template: str                      # hasufel template name
    overlays: list[str]                # hasufel overlay stack
    config_override: dict[str, Any]    # Explicit parameter overrides
    version: int
    approved_by: str | None
    approved_at: datetime | None
    created_at: datetime
```

---

## 8. Execution Triggers & Scheduling

### 8.1 Three Trigger Modes

```
┌────────────┐     ┌────────────┐     ┌────────────┐
│  On-Demand │     │  Scheduled │     │ CI-Trigger │
│            │     │            │     │            │
│ User clicks│     │ Cron job   │     │ Git push   │
│ "Validate" │     │ fires      │     │ webhook    │
│ in UI or   │     │ weekly /   │     │ triggers   │
│ calls API  │     │ monthly    │     │ validation │
└─────┬──────┘     └─────┬──────┘     └─────┬──────┘
      │                  │                   │
      └──────────┬───────┘───────────────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  Validation API  │
        │  POST /v1/       │
        │  validations     │
        │                  │
        │  • strategy_id   │
        │  • trigger_type  │
        │  • scenario_set  │
        │   (override opt.)│
        └────────┬─────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  Task Queue      │───→ Orchestrator ───→ Workers
        └─────────────────┘
```

### 8.2 Scheduled Validation

Scheduled validation addresses SR 11-7's ongoing monitoring requirement.
A strategy that passed validation last quarter may fail under new market
conditions.

| Parameter | Configuration |
|---|---|
| Frequency | Per strategy: daily / weekly / monthly / quarterly |
| Scenario set | Mandatory (all tiers) + last adversarial set |
| Notification | Email digest to strategy owner + risk manager |
| Escalation | Automatic ticket creation if score drops below threshold |
| Historical comparison | Report includes delta from previous scheduled run |

---

## 9. Report Architecture

### 9.1 Report Data Model

```python
class ValidationReport(BaseModel):
    """The primary deliverable of a validation run."""

    # ── Identity & Traceability ──
    report_id: UUID
    validation_run_id: UUID
    strategy_id: UUID
    strategy_code_hash: str           # SHA-256 of source code
    scenario_set_version: int         # Immutable snapshot reference
    generated_at: datetime
    model_versions: dict[str, str]    # {"analyzer": "gpt-4o", "planner": "...", ...}

    # ── Executive Summary ──
    overall_verdict: Literal["pass", "conditional_pass", "fail"]
    confidence: float                 # 0.0–1.0
    headline_risks: list[str]
    aggregate_score: float            # Weighted 6-axis mean

    # ── Strategy Profile ──
    strategy_profile: StrategyProfile
    intent_mismatch_findings: list[Finding]

    # ── Per-Scenario Results ──
    scenario_results: list[ScenarioReportSection]

    # ── Cross-Scenario Synthesis ──
    systematic_weaknesses: list[Finding]
    scenario_specific_vulnerabilities: list[Finding]

    # ── Recommendations ──
    improvement_recommendations: list[Recommendation]

    # ── Reproducibility ──
    seeds: dict[str, int]            # scenario_name → seed
    simulation_config_snapshot: dict  # Full config for replay
```

### 9.2 Report Formats

| Format | Rendering | Storage |
|---|---|---|
| **Interactive** | Streamlit dashboard with drill-down | Real-time from DB |
| **PDF** | Jinja2 template → WeasyPrint / ReportLab | S3, immutable |
| **JSON** | Pydantic `.model_dump_json()` | S3 + API endpoint |
| **Email** | HTML template with summary + link to full report | Sent via SES/SendGrid |

### 9.3 Immutability & Audit Trail

- Reports are **write-once.** No update or delete operations.
- Each report carries a **content hash** (SHA-256 of the serialized JSON). Downstream consumers can verify integrity.
- S3 Object Lock (Compliance mode) or GCS Retention Policy prevents deletion even by administrators.
- Report access is logged (who viewed/downloaded, when).

---

## 10. Security & Compliance Architecture

### 10.1 Authentication & Authorization

```
┌─────────┐     ┌────────────┐     ┌─────────────────────┐
│  User   │────→│  Identity  │────→│  RBAC Authorization  │
│         │     │  Provider  │     │                      │
│         │     │ (Auth0/    │     │  Roles:              │
│         │     │  Keycloak/ │     │  • analyst           │
│         │     │  Azure AD) │     │  • risk_manager      │
│         │     │            │     │  • compliance        │
│         │     │            │     │  • admin             │
└─────────┘     └────────────┘     └─────────────────────┘
```

| Role | Permissions |
|---|---|
| **analyst** | Submit strategies, trigger validations, view own reports |
| **risk_manager** | All analyst + manage strategy-type scenarios, view all reports, approve strategies |
| **compliance** | All risk_manager + manage regulatory scenarios, export audit data |
| **admin** | All compliance + platform configuration, user management |

### 10.2 Strategy Sandbox (Enhanced)

The PoC's AST sandbox is preserved and hardened:

| Control | PoC | Target |
|---|---|---|
| AST forbidden constructs | `eval`, `exec`, `compile`, `getattr`, etc. | Same + `ctypes`, `cffi`, `subprocess` |
| Import whitelist | `numpy`, `math`, `collections`, etc. | Same, configurable per deployment |
| Execution timeout | `ThreadPoolExecutor` with timeout | Same + per-worker cgroup limits (CPU, memory) |
| Network isolation | None (trusted local) | Worker containers have no network access |
| Resource limits | None | cgroup: 2GB RAM, 2 CPU cores per strategy execution |

### 10.3 Data Classification

| Data Type | Classification | Handling |
|---|---|---|
| Strategy source code | **Confidential** | Encrypted at rest (AES-256), transit (TLS 1.3), access-logged |
| Simulation results | **Internal** | Encrypted at rest, standard access controls |
| Reports | **Internal** (may be shared with regulators) | Immutable, integrity-verified, retention-controlled |
| API keys (LLM providers) | **Secret** | Vault/KMS, never logged, rotated quarterly |
| User credentials | **Secret** | Delegated to identity provider |

### 10.4 Regulatory Compliance Mapping

| Requirement | Regulation | ROHAN Implementation |
|---|---|---|
| Independent model validation | SR 11-7 §§5-8 | AI-inferred intent cross-validated against declared objective |
| Stress testing of algo systems | MiFID II Art. 17(1) | Mandatory 3-tier scenario registry |
| Documentation of algo characteristics | MiFID II Art. 17(2) | StrategyProfile auto-generated and persisted |
| Audit trail | SR 11-7 §11 | Immutable reports, versioned scenarios, access logging |
| Ongoing monitoring | SR 11-7 §9 | Scheduled re-validation with delta comparison |
| Model inventory | EBA GL 2017/11 | Strategy registry with version history |
| Reproducibility | SR 11-7 §6 | Deterministic seeds, config snapshots, replay capability |

---

## 11. Cloud Infrastructure Topology

### 11.1 Reference Deployment (AWS)

```
┌─────────────────────────────────────────────────────────────────────┐
│  VPC                                                                │
│                                                                     │
│  ┌──────────────────────────────────────────────────┐              │
│  │  Public Subnet                                    │              │
│  │                                                   │              │
│  │  ┌────────────┐    ┌────────────┐                │              │
│  │  │ ALB        │    │ CloudFront │                │              │
│  │  │ (API GW)   │    │ (UI static)│                │              │
│  │  └─────┬──────┘    └────────────┘                │              │
│  └────────┼──────────────────────────────────────────┘              │
│           │                                                         │
│  ┌────────▼──────────────────────────────────────────┐              │
│  │  Private Subnet — Application Tier                │              │
│  │                                                   │              │
│  │  ┌────────────┐    ┌────────────────────────┐     │              │
│  │  │ ECS/EKS    │    │  ECS/EKS               │     │              │
│  │  │            │    │                        │     │              │
│  │  │ API Service│    │  Orchestrator Service  │     │              │
│  │  │ (FastAPI)  │    │  (LangGraph)           │     │              │
│  │  │ 2 tasks    │    │  2 tasks               │     │              │
│  │  └────────────┘    └────────────────────────┘     │              │
│  │                                                   │              │
│  │  ┌────────────────────────────────────────────┐   │              │
│  │  │  ECS/EKS — Simulation Workers              │   │              │
│  │  │                                            │   │              │
│  │  │  Auto-scaled: 2–16 tasks (CPU-bound)       │   │              │
│  │  │  No network access (strategy sandbox)      │   │              │
│  │  │  2 vCPU + 4GB RAM per task                 │   │              │
│  │  └────────────────────────────────────────────┘   │              │
│  └───────────────────────────────────────────────────┘              │
│                                                                     │
│  ┌───────────────────────────────────────────────────┐              │
│  │  Private Subnet — Data Tier                       │              │
│  │                                                   │              │
│  │  ┌────────────┐  ┌──────────┐  ┌──────────────┐  │              │
│  │  │ RDS Postgr.│  │ ElastiC. │  │ S3           │  │              │
│  │  │ (db.r6g.lg)│  │ (Redis)  │  │ (reports,    │  │              │
│  │  │ Multi-AZ   │  │ Task     │  │  artifacts,  │  │              │
│  │  │            │  │ queue +  │  │  Object Lock)│  │              │
│  │  │            │  │ cache    │  │              │  │              │
│  │  └────────────┘  └──────────┘  └──────────────┘  │              │
│  └───────────────────────────────────────────────────┘              │
│                                                                     │
│  ┌───────────────────────────────────────────────────┐              │
│  │  Observability                                    │              │
│  │  CloudWatch Logs │ X-Ray (traces) │ CloudWatch    │              │
│  │  Metrics │ Grafana Cloud (dashboards + alerts)    │              │
│  └───────────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

### 11.2 Scaling Characteristics

| Component | Scaling Model | Trigger |
|---|---|---|
| API Service | Horizontal (2–4 tasks) | Request rate |
| Orchestrator | Horizontal (2–4 tasks) | Concurrent validation runs |
| Simulation Workers | Horizontal (2–16 tasks) | Task queue depth |
| PostgreSQL | Vertical (instance size) + read replicas for reporting | Query load |
| Redis | Single node (task queue is lightweight) | N/A |

**Cost-dominant component:** Simulation workers. A 12-scenario validation run
with 4 vCPU workers takes ~3 batches. This is the primary cost lever and the
primary scaling target.

### 11.3 GCP / Azure Equivalent Mapping

| AWS | GCP | Azure |
|---|---|---|
| ECS/EKS | Cloud Run / GKE | AKS / Container Apps |
| RDS PostgreSQL | Cloud SQL | Azure DB for PostgreSQL |
| ElastiCache | Memorystore | Azure Cache for Redis |
| S3 + Object Lock | GCS + Retention Policy | Blob Storage + Immutability Policy |
| ALB | Cloud Load Balancing | Application Gateway |
| CloudWatch + X-Ray | Cloud Logging + Cloud Trace | Azure Monitor + App Insights |
| SES | SendGrid (third-party) | SendGrid / Communication Services |

---

## 12. Data Flow Summary (End-to-End)

```
User/CI/Cron
    │
    ▼
POST /v1/validations
    │  {strategy_id, trigger, scenario_overrides}
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  API Gateway                                                     │
│  • Authenticate (JWT)                                            │
│  • Authorize (RBAC)                                              │
│  • Validate request                                              │
│  • Create ValidationRun record (status: "queued")                │
│  • Enqueue to task queue                                         │
│  • Return 202 Accepted + run_id                                  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Orchestrator picks up job                                       │
│                                                                  │
│  1. INGEST                                                       │
│     • Load strategy from DB                                      │
│     • AST validation + adapter translation                       │
│     • Status → "analyzing"                                       │
│                                                                  │
│  2. ANALYZE (LLM)                                                │
│     • Code comprehension → StrategyProfile                       │
│     • Intent cross-validation                                    │
│     • Weakness hypothesis                                        │
│     • Status → "planning"                                        │
│                                                                  │
│  3. PLAN (LLM + tools)                                           │
│     • Load mandatory scenarios from registry                     │
│     • AI adversarial planning (3-tier fallback)                  │
│     • Assemble + deduplicate + seed                              │
│     • Version-stamp scenario set                                 │
│     • Status → "executing"                                       │
│                                                                  │
│  4. EXECUTE (parallel workers)                                   │
│     • Fan-out: N scenario jobs → task queue                      │
│     • Workers run simulations (baseline + strategy)              │
│     • Workers return ScenarioResult + RichAnalysisBundle         │
│     • Fan-in: collect all results                                │
│     • Status → "analyzing_results"                               │
│                                                                  │
│  5. EXPLAIN (LLM, per scenario + aggregate)                      │
│     • Per-scenario ReAct explainer with 8 investigation tools    │
│     • Cross-scenario aggregation                                 │
│     • Improvement recommendations                                │
│     • Status → "generating_report"                               │
│                                                                  │
│  6. REPORT                                                       │
│     • Assemble ValidationReport                                  │
│     • Render PDF → S3                                            │
│     • Render JSON → S3                                           │
│     • Persist to DB (immutable)                                  │
│     • Status → "completed"                                       │
│     • Notify (email/webhook)                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                   GET /v1/validations/{run_id}/report
```

---

## 13. Technology Stack Summary

| Layer | Technology | Justification |
|---|---|---|
| **API** | FastAPI | Async, OpenAPI schema generation, Pydantic-native |
| **Orchestration** | LangGraph | Already proven in PoC; checkpointing, fan-out/fan-in, state management |
| **Simulation** | abides-hasufel | LOB simulator with full microstructure; already integrated |
| **LLM** | LangChain + OpenRouter | Model-agnostic; any provider works |
| **Task Queue** | Celery + Redis | Python-native, proven at scale, rich monitoring |
| **Database** | PostgreSQL (RDS/Cloud SQL) | ACID, JSON columns, mature tooling |
| **Object Storage** | S3 / GCS | Immutable retention, cost-effective, lifecycle policies |
| **Auth** | Auth0 / Keycloak / Azure AD | Delegated identity, OIDC/SAML, MFA |
| **Observability** | OpenTelemetry + Grafana | Vendor-neutral tracing, metrics, dashboards |
| **PDF Generation** | WeasyPrint or ReportLab | Python-native, template-driven |
| **CI/CD** | GitHub Actions | Already in use; extends to scheduled validations |
| **Container Runtime** | ECS Fargate / Cloud Run / AKS | Serverless containers, auto-scaling |

---

## 14. Migration Path from PoC

### 14.1 What Carries Forward (Directly Reusable)

| Component | PoC Location | Target Role |
|---|---|---|
| `StrategicAgent` protocol | `strategy_api.py` | Strategy contract (unchanged) |
| Strategy sandbox | `strategy_validator.py` | Ingest stage validation |
| `config_builder.py` | `simulation/` | Simulation worker config pipeline |
| `run_simulation()` / `run_batch()` | `simulation_service.py` | Worker execution path |
| `HasufelOutput` + `RichAnalysisBundle` | `hasufel_output.py`, `analysis_models.py` | Data capture pipeline |
| `compute_rich_analysis()` | `analysis_service.py` | Worker analysis step |
| 6-axis deterministic scoring | `scoring.py` | Unchanged in the aggregator |
| 8 investigation tools | `tools.py` | Explainer ReAct agent toolkit |
| 3 scenario tools | `scenario_tools.py` | Planner agent toolkit |
| Planner (3-tier fallback) | `planner.py` | Planner node |
| Explainer (ReAct + fallback) | `nodes.py` | Explainer node |
| Prompt templates | `prompts.py` (both framework + llm) | Adapted for new flow |
| Pandera schemas | `schemas.py` | Data validation at boundaries |
| Feature flags | `feature_flags.py` | Staged rollout (extended) |
| Structured telemetry | `telemetry.py` | Preserved, extended with OpenTelemetry export |

### 14.2 What Changes

| PoC Component | Change | Rationale |
|---|---|---|
| Writer node | **Remove** | Strategies are user-supplied |
| Refinement loop (graph cycle) | **Remove** | Linear DAG, single pass |
| Convergence/rollback logic | **Remove** | No iteration to converge/rollback |
| Aggregator node | **Simplify** | Pure synthesis, no loop control |
| Streamlit as app server | **Decouple** | UI calls API; orchestration is separate |
| SQLite default | **Replace** | PostgreSQL as primary |
| Inline base64 artifacts | **Migrate** | Object storage with URI references |
| DB schema | **Evolve** | New tables (Strategy, ScenarioRegistry, Report, Explanation) |
| Single-process execution | **Decouple** | API + Orchestrator + Workers |

### 14.3 What's New

| Component | Purpose |
|---|---|
| Analyzer node | LLM reads strategy, produces StrategyProfile |
| Strategy Adapter Layer | Translates real-time constructs to discrete-time |
| ReportBuilder node | Assembles multi-format immutable reports |
| Scenario Registry (DB + Admin UI) | Versioned, tiered, approval-controlled |
| Scheduled validation | Cron-triggered re-validation |
| CI webhook trigger | Validates on strategy code change |
| FastAPI service | REST API for all consumers |
| PDF report generation | Audit-ready output |
| RBAC | Role-based access control |
| LLM cost metering | Per-run, per-user token tracking |

---

## 15. Open Questions & Risks

| # | Question/Risk | Impact | Mitigation |
|---|---|---|---|
| 1 | **Adapter fidelity** — How many real-time strategies can be faithfully translated? | If most strategies require refactoring, adoption suffers | Start with protocol-native path; build adapter incrementally based on actual strategy patterns |
| 2 | **LLM cost per run** — Analyzer + Planner + N Explainers + Aggregator is 4+ LLM calls minimum | Cost scales with scenario count | Token budgets per run; smaller models for lower-stakes calls; caching for repeated analyses |
| 3 | **Simulation fidelity vs. real markets** — ABIDES is a stylized model, not a real exchange | Users may over-trust results | Reports must include a model limitations section; scenario descriptions must be explicit about what is and isn't modeled |
| 4 | **Report generation latency** — PDF generation with many charts may be slow | User experience for on-demand runs | Generate PDF asynchronously; serve interactive view first |
| 5 | **Scenario registry governance** — Who approves changes? What's the workflow? | Regulatory risk if poorly governed | Design approval workflow as part of the admin UI from day one |
| 6 | **Multi-asset support** — PoC is single-ticker (ABM); real strategies may be multi-asset | Limits applicability | Document as known limitation; plan multi-ticker hasufel support as future work |

---

## 16. Glossary

| Term | Definition |
|---|---|
| **Validation Run** | A single end-to-end execution of the validation pipeline for one strategy |
| **Scenario** | A market environment configuration (template + overlays + overrides) |
| **Mandatory Scenario** | A scenario required by the registry (regulatory / strategy-type / organization tier) |
| **Adversarial Scenario** | An AI-designed scenario targeting a specific strategy weakness |
| **StrategyProfile** | AI-generated analysis of a strategy's behavior, style, and risk dimensions |
| **Finding** | A specific issue or vulnerability identified during validation |
| **Recommendation** | An actionable suggestion for improving the strategy, with code-level specificity |
| **RichAnalysisBundle** | Serialized simulation data: fills, PnL curve, inventory, adverse selection, L2 snapshots |
| **6-Axis Score** | Deterministic scoring across Profitability, Risk, Volatility Impact, Spread Impact, Liquidity Impact, Execution Quality |
