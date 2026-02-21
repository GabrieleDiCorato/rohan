# Agentic Simulation Framework - Technical Specification (SPEC_DEV.md)

## 1. Overview
This framework provides an autonomous loop for generating, testing, and refining specific trading strategies using `abides-rohan`. It leverages **LangGraph** for state management, **PostgreSQL** for relational data storage, and **Docker** for isolated execution.

## 2. Core Architecture

### 2.1 Technology Stack (subject to change)
*   **Orchestration:** LangGraph (Python)
*   **Simulation Engine:** abides-rohan (Existing)
*   **Database:** PostgreSQL
    *   *Features:* Relational Tables for Metrics/Logs/Series, BYTEA for Images/Code.
*   **LLM Integration:** Via `langchain` connectors or `OpenRouter` API. For example:
    *   *Analysis:* Google Gemini 1.5 Pro (via `langchain-google-genai`)
    *   *Code Generation:* Claude 3.5 (via `langchain-openai` / OpenRouter)
*   **Execution:** LLM-generated code is executed in Docker containers (via `docker-py`) or using secure Python interpreters (e.g. [`pydantic/monty`](https://github.com/pydantic/monty)).

### 2.2 System Components & Communication

The system follows a directed graph architecture where the Manager acts as the central orchestrator.

**Communication Protocol:**
*   **Control Flow (LangGraph):**
    *   **Persistence:** Uses `PostgresSaver` to checkpoint agent state (messages, current code, decision processing) synchronously. This allows the UI to query the exact status of any session.
    *   **Real-time Streaming:** The Manager API exposes a **WebSocket/SSE Endpoint**. It streams generic `LangGraph` events (Node Start/End) and custom application events (e.g., "Validation Error", "Simulation Progress 50%") directly to the UI.
*   **Data Flow (Bulk Storage):**
    *   **Simulation Artifacts:** Heavy data (ticks, order books) is handled via **Async Persistence**. The `SimulationNode` triggers a background write to PostgreSQL.
    *   **Lazy Loading:** The UI/Agents receive a `SimulationResult` containing *aggregate metrics* (Pydantic) and a *RunID*. If they need detailed series data, they request it via the RunID (Lazy Load).

## 3. Data Models

### 3.1 Strategic Agent Protocol (Framework Agnostic)
Defined in [src/rohan/simulation/models/strategy_api.py](../src/rohan/simulation/models/strategy_api.py). This is the **ONLY** interface the LLM interacts with.

**Units & Conventions (matching ABIDES):**
*   **Prices:** `int`, in **cents** (e.g. `18550` = $185.50).
*   **Quantities:** `int`, in **shares**.
*   **Cash:** `int`, in **cents**.
*   **Timestamps:** `int`, nanoseconds since epoch.

It defines:
*   `MarketState`: The agent's view of the market (prices, inventory, orders). All monetary fields are `int` cents.
*   `OrderAction`: The actions the agent can take. Includes a `@model_validator` enforcing:
    *   `LIMIT` orders **must** specify a `price`.
    *   `MARKET` orders **must not** specify a `price`.
*   `StrategicAgent` Protocol: The interface (`initialize`, `on_market_data`, `on_order_update`) that generated strategies must implement.

### 3.2 Data Exchange Objects (DXOs)
To balance type safety within the Agent Logic and performance for Simulation Data, we use a tiered approach:

1.  **Metadata & Summaries (Pydantic):**
    *   `SimulationSummary`: Lightweight object containing aggregate KPIs (PnL, Sharpe, Max Drawdown). Safe to serialize and pass to LLMs/UI.
    *   `StrategyEvaluation`: Contains the code, the summary, and the LLM's reasoning.
2.  **Bulk Data (Typed DataFrames via Pandera):**
    *   `SimulationOutput` (DAO): An abstract interface (`ABC`) that provides access to bulk data (Order Books, Logs). Return types are annotated with `pandera.typing.DataFrame[Schema]`.
    *   **Schema Definitions:** Defined in [src/rohan/simulation/models/schemas.py](../src/rohan/simulation/models/schemas.py) using `pandera.DataFrameModel`:
        *   `OrderBookL1Schema`: `time`, `bid_price`, `bid_qty`, `ask_price`, `ask_qty`, `timestamp`. `strict=False` to allow downstream-computed columns (e.g. `mid_price`).
        *   `OrderBookL2Schema`: `time`, `level`, `side`, `price`, `qty`, `timestamp`. `side` is constrained to `{"bid", "ask"}`, `level >= 1`.
        *   `AgentLogsSchema`: `AgentID`, `AgentType`, `EventType`. `strict=False` because upstream `parse_logs_df` may add extra columns.
    *   **Validation Strategy:** Schemas are validated at the *production boundary* — i.e. in `AbidesOutput` (concrete `SimulationOutput`) right after data is computed and before it is cached. Consumers (e.g. `AnalysisService`) rely on annotations for documentation without re-validating.
    *   **Transport:** Internally passed as `pd.DataFrame`. Over network/API, served as **Parquet** or **Arrow** streams, referenced by `RunID`.

### 3.3 Database Schema (PostgreSQL/SQLite)
Defined in [src/rohan/framework/database/models.py](../src/rohan/framework/database/models.py).

**Hierarchy:**
`StrategySession` (User goal) -> `StrategyIteration` (One code version) -> `SimulationScenario` (Conditions) -> `SimulationRun` (Execution).

**Key Tables:**
*   `strategy_sessions`: High-level user goal.
*   `simulation_scenarios`: Configuration overrides for different test cases.
*   `strategy_iterations`: Generated Python code versions.
*   `simulation_runs`: Execution results linked to iteration and scenario.
*   `market_data_l1`: High-frequency L1 order book data.
*   `agent_logs`: Detailed agent actions and events.
*   `artifacts`: Binary storage for plots and log files.

## 4. Workflows

### 4.1 Phase 1: The "Observer" (Running Baselines on Existing Agents)
**Goal:** Verify the system can run simulations, capture data, and provide intelligent analysis without injecting new strategy code yet.

1.  **Session Initialization**: User requests analysis. System creates session and scenarios.
2.  **Orchestration**: Manager triggers generic runs.
3.  **Execution**: `SimulationNode` runs simulations, extracts data, computes metrics, and persists to DB.
4.  **Analysis**: `ResultAnalyzer` retrieves metrics and plots, uses LLM to generate a report explaining observed dynamics.

### 4.2 Phase 2: The "Strategist" (Iterative Refinement)
**Goal:** Autonomous strategy generation and improvement loops.

1.  **Strategy Generation**: LLM generates Python code implementing `StrategicAgent`.
2.  **Validation**: Code is validated via AST (no dangerous imports).
3.  **Scenario Testing**: System runs simulation with new code injected.
4.  **Evaluation**: Compare against baseline or previous iteration.
5.  **Refinement**: Feedback loop to generate next iteration if needed.

## 5. Implementation Status & Roadmap

### ✅ CHECKPOINT: Phase 1 & 1.5 Foundations

Phase 1 (Data/Execution) and Phase 1.5 (Vertical Prototype) establish the core capabilities.

#### Phase 1.1: Database, Schemas, and Models ✅
**Status:** Complete and tested.
*   Pydantic schemas in `src/rohan/simulation/models/`.
*   **Pandera DataFrame schemas** in `src/rohan/simulation/models/schemas.py` (`OrderBookL1Schema`, `OrderBookL2Schema`, `AgentLogsSchema`).
*   SQLAlchemy models in `src/rohan/framework/database/models.py`.
*   DB Connection in `src/rohan/framework/database/database_connector.py`.
*   Repository Layer in `src/rohan/framework/repository.py`.
*   Initialization scripts in `src/rohan/framework/database/init_db.py`.

#### Phase 1.2: Execution Engine ✅
**Status:** Complete and tested.
*   `SimulationEngine` in `src/rohan/framework/simulation_engine.py` orchestrates local execution and persistence.

#### Phase 1.3: Analysis Service ✅
**Status:** Complete and tested.
*   `AnalysisService` in `src/rohan/framework/analysis_service.py` computes metrics and generates Matplotlib plots.

#### Phase 1.4: Framework Hardening ✅
**Status:** Complete and tested.
Technical debt verification and cleanup.

**Tasks:**
- [x] **Session Management**: Use `scoped_session` and ensure proper cleanup. — `DatabaseConnector` now uses `scoped_session` with `remove_session()` for thread-safe, leak-free session management.
- [x] **Schema Fixes**:
    - [x] `SimulationRun` status enum. — `RunStatus` enum (`PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`).
    - [x] Add `error_message`, `error_traceback` to `SimulationRun`. — Both `Text` columns, populated on failure.
    - [x] Add timestamps (`created_at`, `updated_at`). — `server_default=func.now()`, `onupdate=func.now()` on all relevant tables.
    - [x] Add cascade delete to relationships. — `cascade="all, delete-orphan"` + `ondelete="CASCADE"` on all FK relationships.
- [x] **Missing Indexes**: Add indexes for frequently queried fields (`agent_type`, `event_type`, `status`). — `ix_agent_logs_agent_type`, `ix_agent_logs_event_type`, `ix_simulation_runs_status`.
- [x] **Artifact Storage**: Refactor to support file-system or S3 backed storage (currently DB-only). — `ArtifactStore` accepts `artifact_root` for filesystem backend; `Artifact` model has `storage_backend` and `ArtifactType` enum.
- [x] **Logging**: Replace `print()` with `logging` module. — All `print()` calls replaced with `logging.getLogger(__name__)` across `init_db.py`, `database_connector.py`, `simulation_engine.py`, `repository.py`.
- [x] **Metrics**: Improve handling of missing metrics (None vs 0.0). — All metric fields are now `float | None`; `None` = "not computed".
- [x] **Plot Pipeline**: Implement `figure_to_bytes` and ensure plots are saved as artifacts during runs. — `AnalysisService.figure_to_bytes()` returns raw PNG bytes; `SimulationEngine.run_local()` auto-saves price/volume/spread plots as artifacts.

#### Phase 1.5: Minimal Vertical Prototype
**Goal:** Validate key interfaces and interpreter loop.

*   **1.5.1 StrategicAgent API Redesign** ✅
    *   Defined in [src/rohan/simulation/models/strategy_api.py](../src/rohan/simulation/models/strategy_api.py).
    *   Mapped to ABIDES internals.

*   **1.5.2 ABIDES Adapter & Injection** ✅
    *   Implemented in [src/rohan/simulation/abides_impl/strategic_agent_adapter.py](../src/rohan/simulation/abides_impl/strategic_agent_adapter.py).
    *   Allows dynamic injection of strategies.

*   **1.5.3 Sandboxed Execution** ✅
    *   Implemented in [src/rohan/simulation/strategy_validator.py](../src/rohan/simulation/strategy_validator.py).
    *   AST validation and restricted environment execution.

*   **1.5.4 Agent-Specific KPIs** ✅
    *   Implemented in `src/rohan/simulation/models/simulation_metrics.py`.
    *   **`SimulationMetrics`:** Market-wide metrics — `volatility` (annualised), `mean_spread`, `effective_spread`, `avg_bid_liquidity`, `avg_ask_liquidity`, `traded_volume`. Monetary averages are `float | None` in cents (None = not yet computed).
    *   **`AgentMetrics`:** Per-agent — `initial_cash` (`int`, cents), `ending_cash` (`int`, cents), `total_pnl` (`float | None`, cents — fractional from mid-price), `sharpe_ratio`, `max_drawdown`, `inventory_std`, `trade_count`, `fill_rate`, `order_to_trade_ratio`, inventory fields.
    *   **`MarketMetrics`:** Same shape as `SimulationMetrics` for baseline comparison.
    *   **`MarketImpact`:** Typed Pydantic model (not `dict`) with `spread_delta_pct`, `volatility_delta_pct`, `bid_liquidity_delta_pct`, `ask_liquidity_delta_pct`.
    *   **`ComparisonResult`:** Uses `MarketImpact` (typed) for `market_impact` field.
    *   Volatility annualisation derives observation frequency from actual median timestamp intervals, not a hard-coded assumption.
    *   **Agent auto-detection:** `compute_agent_metrics()` accepts an optional `agent_id`. When omitted (or `None`), the strategic agent is located automatically via `AbidesOutput.get_strategic_agent_id()` which matches on `agent.type == "StrategicAgent"`. When auto-detected, `initial_cash` is read from the agent's `starting_cash` attribute so PnL is computed correctly without callers needing to know the agent ID or starting capital.

*   **1.5.5 Structured Summary for LLM** ✅
    *   `RunSummary` model and `generate_summary` in `analysis_service.py`.
    *   Prompt templates in `src/rohan/framework/prompts.py`.

*   **1.5.6 Single Iteration Pipeline** ✅
    *   Implemented in [src/rohan/framework/iteration_pipeline.py](../src/rohan/framework/iteration_pipeline.py).
    *   `IterationPipeline.run()` wires validate → execute (strategy + baseline) → analyse → persist → interpret.
    *   `PipelineConfig` for knobs (settings, baseline overrides, persistence toggle, goal description).
    *   `IterationResult` Pydantic model with `validation`, `comparison`, `summary`, `interpreter_prompt`, and `.success` property.
    *   Optional DB persistence of iterations, runs, and chart artifacts.
    *   Entry point for the "Game Loop" (Phase 2).

### ✅ CHECKPOINT: Phase 2 - LangGraph Orchestration (Steps 2.1–2.4)

Phase 2 implements the autonomous agent loop using LangGraph. Steps 2.1–2.4 (MVP scope) are complete and tested.

#### Step 2.1: LLM Integration MVP ✅
**Status:** Complete and tested.
**Goal:** Run a strategy, gather metrics, and have an LLM interpret the results.

##### 2.1.1 LangChain Model Setup
Uses **LangChain** for model abstraction with **OpenRouter** as default provider.

*   **Dependencies** in `pyproject.toml`:
    *   Core: `langchain>=0.3`, `langchain-openai>=0.2`, `langgraph>=1.0.8`
    *   Optional: `[llm]` extra includes `langchain-google-genai>=2.0`
*   **[src/rohan/llm/\_\_init\_\_.py](../src/rohan/llm/__init__.py)** — LLM module init.
*   **[src/rohan/llm/factory.py](../src/rohan/llm/factory.py)** — Model factory with provider dispatch:
    *   `create_chat_model(model_name, settings)` — Creates LangChain `BaseChatModel` for any provider.
    *   `get_codegen_model()`, `get_analysis_model()`, `get_judge_model()` — Convenience getters with `_cached_settings()`.
    *   Provider factories: `_create_openrouter_model` (OpenAI-compatible), `_create_openai_model`, `_create_google_model`.
*   **[src/rohan/config/llm_settings.py](../src/rohan/config/llm_settings.py)** — Pydantic settings:
    *   `LLMProvider` (StrEnum): `openrouter`, `openai`, `google`.
    *   `LLMSettings` (BaseSettings): `env_prefix="LLM_"`, `.env` file support, `validation_alias` for API keys (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`), `populate_by_name=True`.
    *   Default models: `anthropic/claude-sonnet-4` (codegen), `google/gemini-2.0-flash-001` (analysis/judge).
*   **[src/rohan/llm/models.py](../src/rohan/llm/models.py)** — Pydantic structured output models:
    *   `GeneratedStrategy` (class_name, code, reasoning), `ScenarioExplanation`, `JudgeVerdict` (score 1-10, comparison, recommendation), `AggregatedFeedback`, `ScenarioMetrics`, `IterationSummary`.
*   **[src/rohan/llm/prompts.py](../src/rohan/llm/prompts.py)** — Prompt templates:
    *   `WRITER_SYSTEM` (StrategicAgent protocol spec), `WRITER_HUMAN`, `WRITER_FEEDBACK_TEMPLATE`, `EXPLAINER_SYSTEM/HUMAN`, `AGGREGATOR_SYSTEM/HUMAN`, `HISTORY_TABLE_HEADER/ROW_TEMPLATE`.

**File Structure:**
```
src/rohan/llm/
├── __init__.py
├── cli.py              # CLI entry point
├── factory.py          # LangChain model factory
├── graph.py            # LangGraph state machine
├── models.py           # Pydantic response models
├── nodes.py            # LangGraph node functions
├── prompts.py          # Prompt templates
├── state.py            # RefinementState TypedDict
└── tools.py            # Explainer agent tools

src/rohan/config/
└── llm_settings.py     # LLMProvider, LLMSettings

tests/
├── test_llm_factory.py
├── test_llm_graph.py
├── test_llm_models.py
├── test_llm_nodes.py
├── test_llm_prompts.py
├── test_llm_state.py
└── test_llm_tools.py
```

---

#### Step 2.2: Multi-Agent Architecture ✅
**Status:** Complete and tested.

**Design Principle:** Each agent is a separate LangGraph node with a single responsibility. Agents communicate through state, not direct calls.

##### Agent Topology

```mermaid
flowchart TD
    START([🎯 Goal]) --> W

    subgraph generation ["📝 Strategy Generation Loop"]
        W[Writer Agent<br/>Generates strategy code] --> V[Validator Agent<br/>AST + sandbox check]
        V -->|❌ Invalid<br/>retries ≤ 3| W
    end

    V -->|✅ Valid| SE
    V -->|❌ Max retries| FAIL([💥 Failed])

    subgraph execution ["⚡ Scenario Execution"]
        SE[Scenario Executor] --> S1[Scenario 1<br/>Volatile Market]
        SE --> S2[Scenario 2<br/>Stable Market]
        SE -.-> SN[Scenario N<br/>Custom]
    end

    subgraph analysis ["🔍 Scenario Analysis"]
        S1 --> E1[Explainer Agent<br/>+ Tools 🛠️]
        S2 --> E2[Explainer Agent<br/>+ Tools 🛠️]
        SN -.-> EN[Explainer Agent<br/>+ Tools 🛠️]
    end

    E1 --> AGG[Aggregator<br/>Cross-scenario synthesis]
    E2 --> AGG
    EN -.-> AGG

    AGG -->|iteration < max| W
    AGG -->|converged OR<br/>iteration = max| DONE([✅ Done])

    style W fill:#4CAF50,color:white
    style V fill:#FF9800,color:white
    style SE fill:#2196F3,color:white
    style E1 fill:#9C27B0,color:white
    style E2 fill:#9C27B0,color:white
    style EN fill:#9C27B0,color:white
    style AGG fill:#607D8B,color:white
```

##### 2.2.1 Writer Agent ✅
**Role:** Generate strategy code from goal + feedback.

*   **Input:** Goal description, previous feedback (if any).
*   **Output:** Python strategy code (`GeneratedStrategy` via `with_structured_output`).
*   **LangGraph node:** `writer_node` in [src/rohan/llm/nodes.py](../src/rohan/llm/nodes.py).
*   **Implementation:** Uses `get_codegen_model()` with `WRITER_SYSTEM` / `WRITER_HUMAN` prompt templates. Incorporates iteration history and aggregated feedback on subsequent iterations.

##### 2.2.2 Validator Agent ✅
**Role:** Validate strategy code (AST + sandbox execution).

*   **Input:** Strategy code from state.
*   **Output:** Valid flag, error message (if any).
*   **Loop:** If invalid, return error to Writer. **Failsafe:** Max 3 retries (`MAX_VALIDATION_RETRIES`).
*   **Implementation:** Uses existing `StrategyValidator` from Phase 1.5 + instantiation check via `exec()` in restricted environment.
*   **LangGraph node:** `validator_node` in [src/rohan/llm/nodes.py](../src/rohan/llm/nodes.py).

##### 2.2.3 Scenario Executor ✅
**Role:** Run validated strategy across multiple scenarios.

*   **Input:** Validated strategy code, list of scenario configs.
*   **Output:** List of `ScenarioResult` (metrics + output per scenario).
*   **Implementation:** Uses `execute_strategy_safely()` from `simulation.utils` for both strategy and baseline runs per scenario. Computes comparison metrics via `AnalysisService`.
*   **LangGraph node:** `scenario_executor_node` in [src/rohan/llm/nodes.py](../src/rohan/llm/nodes.py).

##### 2.2.4 Explainer Agent (per scenario) ✅
**Role:** Analyze simulation results with tool access.

*   **Input:** `SimulationOutput`, `SimulationMetrics`, `RunSummary`.
*   **Tools available** (7 tools via `make_explainer_tools(output)` in [src/rohan/llm/tools.py](../src/rohan/llm/tools.py)):
    - `get_order_book_snapshot(timestamp)` — Query L1 data at a point in time.
    - `get_agent_trades(agent_id)` — Get trade history for an agent.
    - `compute_pnl_curve()` — Generate PnL series over time.
    - `get_price_stats()` — Price statistics (mean, std, min, max, return).
    - `get_spread_stats()` — Spread statistics at configurable percentiles.
    - `query_logs(event_type)` — Filter agent logs by event type.
    - `get_volume_profile(n_bins)` — Volume distribution across price bins.
*   **Output:** `ScenarioExplanation` with strengths, weaknesses, recommendations, key observations.
*   **LangGraph node:** `explainer_node` in [src/rohan/llm/nodes.py](../src/rohan/llm/nodes.py). Uses `get_analysis_model()` with `with_structured_output(ScenarioExplanation)`.

##### 2.2.5 Aggregator ✅
**Role:** Combine all scenario explanations into unified feedback + convergence assessment.

*   **Input:** List of `ScenarioExplanation`, iteration history.
*   **Output:** `JudgeVerdict` (score 1-10, comparison vs. previous, recommendation) + `IterationSummary` for history.
*   **Implementation:** Uses `get_judge_model()` with `with_structured_output(JudgeVerdict)`. Builds formatted iteration history table via `_build_history_table()` and `_format_explanations()`.
*   **LangGraph node:** `aggregator_node` in [src/rohan/llm/nodes.py](../src/rohan/llm/nodes.py).

---

#### Step 2.3: LangGraph State & Graph ✅
**Status:** Complete and tested.

##### State Schema
Implemented in [src/rohan/llm/state.py](../src/rohan/llm/state.py):
```python
class IterationSummary(BaseModel):
    """Summary of a single iteration for history tracking."""
    iteration_number: int
    strategy_code: str
    scenario_metrics: dict[str, ScenarioMetrics]  # scenario_name -> metrics
    aggregated_explanation: str
    judge_score: float | None  # 1-10 from LLM judge
    judge_reasoning: str | None
    timestamp: datetime

class RefinementState(TypedDict):
    goal: str
    current_code: str | None
    validation_errors: list[str]
    validation_attempts: int
    scenarios: list[ScenarioConfig]
    scenario_results: list[ScenarioResult]
    explanations: list[ScenarioExplanation]
    iterations: list[IterationSummary]  # ← Full iteration history
    aggregated_feedback: str | None
    iteration_number: int
    max_iterations: int
    status: Literal["writing", "validating", "executing", "explaining", "aggregating", "done", "failed"]
```

##### Convergence Assessment (LLM-as-Judge)

The **Aggregator** evaluates each iteration against the goal and previous iterations.

**Judge Input (Prompt):**
```
## Goal
{goal}

## Previous Iterations
| Iter | PnL | Volatility Δ | Spread Δ | Score | Summary |
|------|-----|--------------|----------|-------|---------|
| 1    | -$50 | +12%        | +5%      | 3/10  | High market impact... |
| 2    | +$20 | +8%         | +2%      | 5/10  | Improved but still... |

## Current Iteration (3)
[Current metrics and scenario explanations]

## Task
1. Score this iteration (1-10) relative to the GOAL
2. Compare to previous iterations — is it improving?
3. Recommend: continue, stop (converged), or stop (no progress)
```

**Judge Output (Structured):**
```python
class JudgeVerdict(BaseModel):
    score: float  # 1-10
    comparison: Literal["better", "worse", "similar"]
    reasoning: str
    recommendation: Literal["continue", "stop_converged", "stop_plateau"]
```

**Convergence Criteria:**
- `stop_converged`: Score ≥ 8 and improvement plateaued
- `stop_plateau`: Score similar for 3+ iterations
- `continue`: Otherwise, keep refining

##### Graph Definition
Implemented in [src/rohan/llm/graph.py](../src/rohan/llm/graph.py) with `build_refinement_graph()` and `run_refinement()` convenience runner.
CLI entry point in [src/rohan/llm/cli.py](../src/rohan/llm/cli.py) (`python -m rohan.llm.cli --goal "..." --max-iterations 3`).
```python
from langgraph.graph import StateGraph, END

def build_refinement_graph():
    graph = StateGraph(RefinementState)

    # Nodes
    graph.add_node("writer", writer_node)
    graph.add_node("validator", validator_node)
    graph.add_node("executor", scenario_executor_node)
    graph.add_node("explainer", explainer_node)  # Runs for each scenario
    graph.add_node("aggregator", aggregator_node)

    # Edges
    graph.add_edge("writer", "validator")
    graph.add_conditional_edges("validator", validation_router, {
        "retry": "writer",      # Invalid + retries left
        "execute": "executor",  # Valid
        "fail": END,            # Max retries exceeded
    })
    graph.add_edge("executor", "explainer")
    graph.add_edge("explainer", "aggregator")
    graph.add_conditional_edges("aggregator", should_continue, {
        "continue": "writer",   # More iterations
        "done": END,            # Converged or max iterations
    })

    graph.set_entry_point("writer")
    return graph.compile()
```

---

#### Step 2.4: Tool-Equipped Explainer ✅
**Status:** Complete and tested.

The Explainer agent uses **tool calling** to deeply analyze simulation results.

##### Tool Definitions
*   **[src/rohan/llm/tools.py](../src/rohan/llm/tools.py)** — `make_explainer_tools(output)` factory function that creates 7 LangChain tools closed over a `SimulationOutput` instance:
    ```python
    def make_explainer_tools(output: SimulationOutput) -> list[BaseTool]:
        """Create LangChain tools bound to a specific simulation output."""

        @tool
        def get_order_book_snapshot(timestamp: int) -> dict: ...

        @tool
        def get_agent_trades(agent_id: int) -> list[dict]: ...

        @tool
        def compute_pnl_curve() -> list[dict]: ...

        @tool
        def get_price_stats() -> dict: ...

        @tool
        def get_spread_stats() -> dict: ...

        @tool
        def query_logs(event_type: str) -> list[dict]: ...

        @tool
        def get_volume_profile(n_bins: int = 20) -> list[dict]: ...

        return [get_order_book_snapshot, get_agent_trades, ...]
    ```

##### Explainer as Structured Output Agent
The Explainer uses `with_structured_output(ScenarioExplanation)` rather than a full ReAct agent for deterministic, structured analysis output. Tool results are injected into the prompt context directly.

```python
# In explainer_node (src/rohan/llm/nodes.py)
model = get_analysis_model().with_structured_output(ScenarioExplanation)
```

---

### 🚧 TODO: Phase 2 Remaining (Steps 2.6–2.7)

#### Step 2.5: UI & Notebook for Local Testing ✅
**Status:** Complete and tested.

*   **Created `notebooks/quickstart.ipynb`** — Interactive demo.
*   **Added "Strategy" tab** — Code editor for strategy input.
*   **Added "Interpretation" panel** — Display agent feedback.
*   **Added "Scenario Results" view** — Per-scenario metrics and explanations.

**UV Scripts:**
```toml
[project.scripts]
ui = "rohan.ui.__main__:main"
refine = "rohan.llm.cli:main"
```

---

#### Step 2.6: LLM Feedback Benchmarking
**Status:** TODO.

*   **Create `src/rohan/llm/eval/`** — Evaluation module.
*   **Create `notebooks/llm_benchmark.ipynb`** — Compare providers.
*   **Document findings** in `docs/llm_evaluation.md`.

---

#### Step 2.7: Code Quality & Hardening
**Status:** IN PROGRESS (2.7.1–2.7.6 ✅, 2.7.9 ✅; remaining: 2.7.7–2.7.8, 2.7.10–2.7.11).

A cross-cutting pass to address technical debt, strengthen safety guarantees, improve testability, and bring the codebase closer to production-grade standards. Items are prioritized from most critical (safety/correctness) to least (thesis polish).

##### 2.7.1 Enforce Strategy Execution Timeout (Safety — Critical) ✅

The `_timeout_seconds` parameter in `execute_strategy_safely()` ([src/rohan/simulation/strategy_validator.py](../src/rohan/simulation/strategy_validator.py)) is currently **accepted but not enforced** (note the leading underscore). An LLM-generated strategy containing an infinite loop in `__init__`, `on_market_data`, or `on_order_update` would hang the entire process indefinitely.

**Implementation:**
- Replace the current in-process `exec()` invocation with a **`multiprocessing.Process`** wrapper.
- The parent process calls `.join(timeout)` on the child and `.terminate()` + `.kill()` if it exceeds the deadline.
- This approach is cross-platform (works on both Windows and Unix), unlike `signal.alarm`.
- Capture the child's return value via `multiprocessing.Queue` or a shared `Value`.
- On timeout, raise a new `SimulationTimeoutError` (see §2.7.4) with a descriptive message including the elapsed time and the timeout limit.
- Update `execute_strategy_safely()` to remove the underscore prefix from `_timeout_seconds`, marking it as an enforced parameter.

**Implemented:** `execute_strategy_safely()` now uses `concurrent.futures.ThreadPoolExecutor` with `future.result(timeout=timeout_seconds)`. Thread-based timeout was chosen over `multiprocessing` because `AbidesOutput.end_state` contains unpicklable C-extension thread locks that cannot cross process boundaries. The `timeout_seconds` parameter (previously `_timeout_seconds`) is now enforced. Raises `SimulationTimeoutError` on timeout. Security rationale documented inline with `# SECURITY` comment block.


##### 2.7.2 Replace `hasattr`-Based Caching with `@cached_property` (Correctness) ✅

`AbidesOutput` ([src/rohan/simulation/abides_impl/abides_output.py](../src/rohan/simulation/abides_impl/abides_output.py)) uses two inconsistent caching patterns:
- **L1 data:** `hasattr(self, "_order_book_l1")` check followed by manual attribute assignment.
- **L2 data:** `dict`-based cache keyed by depth level.

Both patterns are fragile — `hasattr` cannot distinguish "not computed" from "computed and returned `None`", and the `dict` cache scatters state management across multiple methods.

**Implementation:**
- Convert all cacheable computed properties to **`@functools.cached_property`** where possible (single-argument, no parameters).
- For the L2 depth-parameterized cache, initialize a `dict[int, DataFrame]` sentinel in `__init__` and check against it explicitly, or use a helper method with `@lru_cache` scoped to the instance (via `__hash__` + `__eq__` if needed, or a standalone inner function).
- Ensure all cache entries are populated **after** Pandera validation, preserving the current "validate at production boundary" contract.

**Implemented:** All `hasattr(self, "_attr")` patterns in `AbidesOutput` replaced with `@functools.cached_property`. L2 dict cache initialized in `__init__`. Public `get_*()` methods delegate to cached properties.


##### 2.7.3 Review Pandera Schema Strictness (Correctness) ✅

The order book schemas (`OrderBookL1Schema`, `OrderBookL2Schema` in [src/rohan/simulation/models/schemas.py](../src/rohan/simulation/models/schemas.py)) are validated via `SchemaModel.validate()`. By default, Pandera may silently **coerce** column types (e.g., `float64` → `int64`), masking upstream data quality issues from ABIDES.

**Implementation:**
- Audit each schema and explicitly set `strict=True` or `coerce=False` in the `Config` class, depending on intent:
  - `OrderBookL1Schema`: Currently `strict=False` to allow downstream-computed columns (`mid_price`). This is intentional — **document it** with a comment explaining why.
  - `OrderBookL2Schema`: Should be `strict=True` since no downstream columns are expected.
  - `AgentLogsSchema`: Currently `strict=False` because `parse_logs_df` may add extra columns. This is intentional — **document it**.
**Implemented:** `OrderBookL2Schema.Config.strict` changed from `False` → `True`. L1 and AgentLogs intentionally remain `strict=False` with inline comments explaining rationale. Module-level docstring added to `schemas.py` explaining the validation philosophy.


##### 2.7.4 Introduce Domain-Specific Exception Hierarchy (Architecture) ✅

The codebase currently relies on generic `Exception` and `ValueError` for all error paths. This makes it difficult to implement targeted error recovery in the LangGraph nodes (e.g., retry on validation failure but abort on timeout).

**Implementation:**
- Create `src/rohan/exceptions.py` with:
  ```python
  class RohanError(Exception):
      """Base exception for all ROHAN errors."""

  class StrategyValidationError(RohanError):
      """Raised when strategy code fails AST or sandbox validation."""

  class SimulationTimeoutError(RohanError):
      """Raised when a simulation exceeds the configured timeout."""

  class BaselineComparisonError(RohanError):
      """Raised when baseline simulation fails, preventing comparison."""

  class StrategyExecutionError(RohanError):
      """Raised when strategy code fails during simulation execution."""
  ```
- Update `strategy_validator.py` to raise `StrategyValidationError` instead of `ValueError`.
- Update `execute_strategy_safely()` to raise `SimulationTimeoutError` (from §2.7.1) and `StrategyExecutionError`.
- Update `iteration_pipeline.py` to raise `BaselineComparisonError` when the baseline run fails.
**Implemented:** `src/rohan/exceptions.py` created with `RohanError`, `StrategyValidationError`, `SimulationTimeoutError`, `BaselineComparisonError`, `StrategyExecutionError`. `strategy_validator.py` updated to raise domain exceptions. Tests updated to expect `StrategyValidationError` / `StrategyExecutionError` instead of generic `ValueError` / `RuntimeError`.


##### 2.7.5 Replace Database Singleton with Factory Function (Architecture) ✅

`DatabaseConnector` ([src/rohan/framework/database/database_connector.py](../src/rohan/framework/database/database_connector.py)) uses a `__new__`-based singleton pattern that is:
- **Not thread-safe:** Concurrent calls to `__new__` can race on `_instance is None`.
- **Hard to test:** Singletons are global state — test isolation requires manual `_instance = None` resets.
- **Unconventional:** `_initialize()` is called inside `__new__`, which is surprising.

**Implementation:**
- Remove the `__new__` override from `DatabaseConnector`.
- Move initialization logic into a standard `__init__`.
- Add a module-level factory function:
  ```python
  @lru_cache(maxsize=1)
  def get_database_connector() -> DatabaseConnector:
      return DatabaseConnector()
  ```
  `@lru_cache` is safe here because the connector is a lightweight, intentionally long-lived global (no large data payloads — unlike `AbidesOutput`).
- Update all call sites to use `get_database_connector()` instead of `DatabaseConnector()`.
**Implemented:** `__new__` singleton removed from `DatabaseConnector`. Standard `__init__` added. `get_database_connector()` factory with `@functools.lru_cache(maxsize=1)` added. All call sites (`init_db.py`, `repository.py`, `simulation_engine.py`, `scenario_repository.py`, `refinement_repository.py`) updated. `conftest.py` uses `get_database_connector.cache_clear()` for test isolation.


##### 2.7.6 Consolidate Dollar Formatting Utilities (Code Organization) ✅

Monetary formatting logic is currently scattered across three locations:
- `_fmt_dollar()` in [src/rohan/framework/prompts.py](../src/rohan/framework/prompts.py) (for LLM prompt rendering).
- `_m_dollar()` and `_mv_dollar()` in [src/rohan/ui/app.py](../src/rohan/ui/app.py) (for Streamlit metrics).
- `dollarize()` from ABIDES itself (third-party, different conventions).

**Implementation:**
- Create `src/rohan/utils/formatting.py` with canonical formatting functions:
  ```python
  def fmt_dollar(cents: int | float, precision: int = 2) -> str:
      """Format a value in cents as a dollar string. E.g. 18550 → '$185.50'."""

  def fmt_dollar_metric(cents: int | float) -> str:
      """Format for Streamlit st.metric() — no currency symbol, signed."""

  def fmt_dollar_delta(cents: int | float) -> str:
      """Format a delta value for st.metric(delta=...) display."""
  ```
- Replace all inline formatting functions with imports from the shared module.
**Implemented:** `src/rohan/utils/formatting.py` created with `fmt_dollar()`, `fmt_dollar_metric()`, `fmt_dollar_delta()`. `prompts.py` imports `fmt_dollar_metric` and drops inline `_fmt_dollar`. `0_Terminal.py` imports `fmt_dollar` and drops inline `_m_dollar` / `_mv_dollar`. `src/rohan/utils/__init__.py` initializes the package.


##### 2.7.7 Split UI Monolith (Code Organization) ✅
**Status:** Complete and tested.

[src/rohan/ui/app.py](../src/rohan/ui/app.py) was a ~1300+ line file handling all tabs, charts, configuration, and callbacks. This made it difficult to navigate, test, and maintain.

**Implementation:**
- Extracted into a Streamlit multipage app structure:
  ```
  src/rohan/ui/
  ├── 0_Terminal.py               # Main entry: Terminal dashboard (Execute, Metrics, Logs)
  ├── pages/
  │   └── 1_Refinement_Lab.py     # Refinement Lab: Autonomous strategy optimization
  └── utils/
      ├── presets.py              # (existing)
      └── theme.py                # Shared UI theme and colors
  ```
- The Refinement Lab is now a dedicated page with its own state management and layout.

##### 2.7.8 Test Hardening (Testing)

Current test coverage is solid for the happy path but has gaps in failure scenarios, edge cases, and integration.

**2.7.8a Failure-Path Tests for `IterationPipeline`:**
- Baseline simulation failure → assert `BaselineComparisonError` is raised (or graceful degradation).
- Strategy timeout → assert `SimulationTimeoutError` propagates correctly.
- Invalid code surviving AST validation but failing at runtime → assert proper error capture in `IterationResult`.

**2.7.8b Edge-Case Tests for `AbidesOutput`:**
- Empty order book (no snapshots) → assert empty DataFrame with correct schema, not an exception.
- Single snapshot → assert L1/L2 computation handles edge cases (no spread change, no volume delta).
- Missing columns in raw ABIDES output → assert `SchemaError` is raised at the Pandera validation boundary.

**2.7.8c Property-Based Testing with `hypothesis`:**
- Add `hypothesis` to `[dev]` dependencies in `pyproject.toml`.
- Apply `@given(...)` strategies to agent settings validators:
  ```python
  from hypothesis import given, strategies as st

  @given(max_size=st.integers(min_value=-1000, max_value=10000))
  def test_momentum_agent_max_size_bounds(max_size: int):
      if max_size > 1000 or max_size < 1:
          with pytest.raises(ValidationError):
              MomentumAgentSettings(max_size=max_size)
      else:
          settings = MomentumAgentSettings(max_size=max_size)
          assert settings.max_size == max_size
  ```
- Fuzz `OrderAction` model validators (LIMIT must have price, MARKET must not) with random combinations.
- Fuzz `SimulationSettings` validators (minimum duration, seed bounds, agent count limits).

**2.7.8d Minimal Integration Test:**
- Generate a trivial strategy string (e.g., a no-op that returns `[]` from `on_market_data`).
- Pass through `StrategyValidator` → `execute_strategy_safely()` with a short simulation (100 timesteps).
- Assert: `SimulationResult` is populated, metrics are non-None, no exceptions raised.
- This validates the full vertical slice without LLM involvement.

##### 2.7.9 Seed Logging for Reproducibility (Thesis Polish) ✅

The random seed in `SimulationSettings` is generated from `datetime.now().timestamp()`, which is appropriate for exploration but makes reproduction difficult unless the seed is captured.

**Implementation:**
- Log the seed at `INFO` level at the start of every simulation run in `IterationPipeline.run()` and `SimulationEngine`.
- Include the seed in `IterationResult` and `SimulationRun` DB records (already stored via settings, but not prominently surfaced).
- Display the seed in the Streamlit UI sidebar when a run completes.
- In the LangGraph refinement loop, log the seed for each scenario execution in the CLI output.

**Implemented:** `IterationResult.seed` field added. `IterationPipeline.run()` logs seed at INFO level at start. `SimulationEngine.run_local()` logs seed inside the try block. `nodes.py` executor logs seed per scenario. Streamlit sidebar shows "Last Run Seed" code block after a run completes, with a copy hint for reproducibility.

##### 2.7.10 Document Model Role Rationale (Thesis Polish)

`LLMSettings` ([src/rohan/config/llm_settings.py](../src/rohan/config/llm_settings.py)) assigns different models to different roles (codegen, analysis, judge) but does not explain why.

**Implementation:**
- Add docstring comments to each model field explaining the selection criteria:
  ```python
  codegen_model: str = Field(
      default="anthropic/claude-sonnet-4",
      description="Model for strategy code generation. Claude Sonnet is chosen for its "
      "strong code generation capabilities, structured output adherence, and ability to "
      "follow complex protocol specifications (StrategicAgent API)."
  )
  ```
- Optionally, add a section in `docs/` documenting model comparison results and trade-offs (cost, latency, quality) to strengthen the thesis evaluation chapter.

##### 2.7.11 Add Security Boundary Documentation (Thesis Polish)

The `exec()` call in `strategy_validator.py` is the primary security boundary for LLM-generated code. While Docker isolation (Phase 3) is deferred, the current MVP approach should be clearly documented for thesis reviewers.

**Implementation:**
- Add a `# SECURITY` comment block above `execute_strategy()` explaining:
  - The threat model: LLM-generated code may contain infinite loops, excessive memory allocation, or attempts to access the filesystem/network.
  - Current mitigations: AST-level import whitelist, builtin restriction, restricted `exec()` namespace.
  - Known gaps: No CPU/memory limits, no filesystem isolation, no network isolation.
  - Planned mitigation: Docker sandbox (Phase 3).
- Reference this in the thesis writeup as a conscious trade-off between development velocity and security hardening.

---

**MVP Scope (Step 2.1-2.5):** ✅ Complete
1. ✅ Writer + Validator loop (3 retries)
2. ✅ Single pre-defined scenario (architecture supports multiple)
3. ✅ Single Explainer agent with 7 analysis tools
4. ✅ Aggregator with LLM-as-Judge convergence assessment
5. ✅ Configurable 1–N refinement iterations
6. ✅ CLI entry point (`python -m rohan.llm.cli`)
7. ✅ 92 tests covering all LLM modules (253 total across project)
8. ✅ Streamlit UI with Terminal and Refinement Lab pages

**Future Extensions:**
- Multiple scenarios in parallel
- Agent-driven scenario selection
- Advanced tool suite for Explainer
- Persistent checkpointing
- **Advanced Order Management API:**
  - `modify_order()` — Change price/quantity of existing orders without canceling
  - `partial_cancel_order()` — Reduce order quantity while keeping remainder active
  - Note: Not exposing these from ABIDES in current implementation to avoid complexity. Can be added later without breaking changes to the StrategicAgent protocol.

**Dependencies & Order:**
```mermaid
graph LR
    A[Phase 1.5 IterationPipeline] --> B[2.1 LLM Setup]
    B --> C[2.2 Agent Architecture]
    C --> D[2.3 LangGraph State]
    D --> E[2.4 Tool-Equipped Explainer]
    E --> F[2.5 UI/Notebook]
    F --> G[2.6 Benchmarking]
```

**Estimated Timeline:**
| Step | Effort | Notes |
|------|--------|-------|
| 2.1 | 1 day | LangChain setup (Done) |
| 2.2 | 2-3 days | Agent nodes implementation (Done) |
| 2.3 | 1-2 days | LangGraph wiring (Done) |
| 2.4 | 2-3 days | Tools + ReAct explainer (Done) |
| 2.5 | 1-2 days | UI updates (Done) |
| 2.6 | 2-3 days | Benchmarking |
| 2.7 | 3-5 days | Code Quality & Hardening |

---

### 🚧 TODO: Phase 3 - Docker Sandbox
**Status:** Deferred.
*   Docker container for untrusted code execution.

### 🚧 TODO: Phase 4 - Production Features
**Status:** Deferred.
*   Web UI, Dashboards, Leaderboards.
