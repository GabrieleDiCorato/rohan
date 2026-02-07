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


```mermaid
graph TD
    User[User / CLI] -->|Start Session| Manager[LangGraph Manager]

    subgraph "LangGraph Process (In-Memory)"
        Manager -->|1. Generate Code| StratGen[Strategy Generator]
        StratGen -->|Validated Code| SimNode[Simulation Node]
        SimNode -->|Direct call| SimService[SimulationService]
        SimService -->|SimulationResult| Interpreter[Interpreter Agent]
        Interpreter -->|Scenario Summary| Comparator[Comparator Agent]
        Comparator -->|Insights| Synthesizer[Synthesizer Agent]
        Synthesizer -->|Final Report| Manager
    end

    subgraph "Async Persistence (Fire & Forget)"
        SimService -.->|persist_async| DB[(PostgreSQL)]
        StratGen -.->|Save Iteration| DB
    end

    Manager -->|LangGraph Checkpointer| DB
```

**Communication Protocol:**
*   **LangGraph Nodes:** Direct in-memory communication via **LangGraph State**.
    *   `SimulationNode` calls `SimulationService.run_simulation()` directly, returning `SimulationResult` in-memory.
    *   Results are passed to `InterpreterAgent`, `ComparatorAgent`, `SynthesizerAgent` via state.
    *   No database polling — DB is used for **async persistence** only (fire-and-forget).
*   **Database:** Historical storage and checkpointing, not inter-agent communication.

## 3. Data Models

### 3.1 Strategic Agent Protocol (Framework Agnostic)
Defined in `src/rohan/simulation/models/strategy_api.py`. This is the **ONLY** interface the LLM interacts with.

```python
from enum import Enum
from typing import Protocol
from pydantic import BaseModel

class Side(str, Enum):
    BID = "BID"
    ASK = "ASK"

class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"

class MarketState(BaseModel):
    timestamp_ns: int           # Nanoseconds from midnight
    best_bid: int | None        # Price in cents
    best_ask: int | None        # Price in cents
    last_trade: int | None      # Last trade price in cents
    inventory: int              # Signed position
    cash: int                   # Cash in cents
    open_orders: list["Order"]  # Active orders

class OrderAction(BaseModel):
    side: Side
    quantity: int
    price: int | None           # Required for LIMIT orders
    order_type: OrderType
    cancel_order_id: int | None = None  # If set, cancel this order

class StrategicAgent(Protocol):
    """Protocol for LLM-generated trading strategies."""

    def initialize(self, config: AgentConfig) -> None:
        """Called once at the start of the simulation."""
        ...

    def on_market_data(self, state: MarketState) -> list[OrderAction]:
        """Called when new market data (L1/L2) is received."""
        ...

    def on_order_update(self, update: Order) -> list[OrderAction]:
        """Called when an order status changes."""
        ...
```

### 3.2 Database Schema (PostgreSQL)

**Hierarchy:**
`StrategySession` (User goal) -> `StrategyIteration` (One code version) -> `SimulationScenario` (Conditions) -> [SimulationRun](../src/rohan/simulation/simulation_runner.py) (Execution).

**Table: `strategy_sessions`**
| Column | Type | Description |
| :--- | :--- | :--- |
| `session_id` | UUID (PK) | Unique ID for the entire optimization task |
| `goal_description` | TEXT | User's prompt (e.g. "Create a mean reverting strategy") |
| `created_at` | TIMESTAMPTZ | |

**Table: `simulation_scenarios`** (Test Cases)
| Column | Type | Description |
| :--- | :--- | :--- |
| `scenario_id` | UUID (PK) | Unique ID |
| `session_id` | UUID (FK) | Link to session (or NULL if global default) |
| `name` | TEXT | "High Volatility", "Low Liquidity" |
| `config_override` | JSONB | keys to override in SimulationSettings (e.g. `{noise_agents: 500}`) |

**Table: `strategy_iterations`** (Code Versions)
| Column | Type | Description |
| :--- | :--- | :--- |
| `iteration_id` | UUID (PK) | Unique ID |
| `session_id` | UUID (FK) | Link to session |
| `generation_number` | INT | 0, 1, 2... |
| `code` | TEXT | The Python source code of the agent |
| `reasoning` | TEXT | LLM reasoning for this change |

**Table: `simulation_runs`** (Execution Results)
| Column | Type | Description |
| :--- | :--- | :--- |
| `run_id` | UUID (PK) | Unique ID |
| `iteration_id` | UUID (FK) | The strategy code used |
| `scenario_id` | UUID (FK) | The scenario setup used |
| `full_config` | JSONB | Complete JSON config used for reproduction |
| `metrics_summary` | JSONB | e.g. `{sharpe: 1.2, pnl: 5000}` |
| `status` | TEXT | 'PENDING', 'RUNNING', 'COMPLETED', 'FAILED' |

**Table: `market_data_l1`** (Matches [AbidesOutput](../src/rohan/simulation/abides_impl/abides_output.py) L1 DataFrame)
| Column | Type | Description |
| :--- | :--- | :--- |
| `run_id` | UUID (FK) | Link to run |
| `time` | BIGINT | Nanoseconds from midnight |
| `bid_price` | DOUBLE | Event Price |
| `bid_qty` | INT | Event Size |
| `ask_price` | DOUBLE | Event Price |
| `ask_qty` | INT | Event Size |
| `timestamp` | TIMESTAMP | Human readable time |

**Table: `agent_logs`** (Matches [AbidesOutput](../src/rohan/simulation/abides_impl/abides_output.py) Logs DataFrame)
| Column | Type | Description |
| :--- | :--- | :--- |
| `run_id` | UUID (FK) | Link to run |
| `agent_id` | INT | Agent identifier |
| `agent_type` | TEXT | "ValueAgent", "NoiseAgent" |
| `event_type` | TEXT | "ORDER_SUBMITTED", "ORDER_EXECUTED" |
| `time_placed` | TIMESTAMP | When the event occurred |
| `log_json` | JSONB | The full log dict (contains internal state) |

**Table: `artifacts`**
| Column | Type | Description |
| :--- | :--- | :--- |
| `run_id` | UUID (FK) | Link to run |
| `type` | TEXT | 'IMAGE', 'LOG_FILE' |
| `path` | TEXT | Virtual path |
| `content` | BYTEA | Binary content |

## 4. Workflows

### 4.1 Phase 1: The "Observer" (Running Baselines on Existing Agents)
**Goal:** Verify the system can run simulations, capture data, and provide intelligent analysis without injecting new strategy code yet.

1.  **Session Initialization**
    *   **Action:** User (via CLI) requests "Analyze the impact of Noise Agent volume on Volatility".
    *   **System:**
        *   Creates a `StrategySession`.
        *   Creates multiple `SimulationScenarios` (e.g., Scenario A: 500 Noise Agents, Scenario B: 5000 Noise Agents).
2.  **Orchestration (LangGraph Manager)**
    *   **Action:** Manager reads the pending scenarios from the Database.
    *   **System:**
        *   Creates [SimulationRun](../src/rohan/simulation/simulation_runner.py) records for each scenario (Status='PENDING').
        *   Invokes the `SimulationNode` with the list of generic Run IDs.
3.  **Execution (SimulationNode)**
    *   **Action:** Node spins up Docker containers (or local processes) for each Run ID.
    *   **System (Worker):**
        *   Queries DB for `run_id` -> retrieves `SimulationSettings`.
        *   Runs `SimulationService.run_simulation()`.
        *   **ETL:**
            *   Explodes `results.get_order_book_l1()` -> Batch Insert into `market_data_l1`.
            *   Explodes `results.get_logs_df()` -> Batch Insert into `agent_logs`.
            *   Computes Summary Metrics (Sharpe, Volatility) -> Update `simulation_runs`.
            *   Generates Plots (PnL, Price Series) -> Insert into `artifacts` as BYTEA.
    *   **Completion:** Worker updates Status='COMPLETED' and exits.
4.  **Analysis (ResultAnalyzer Agent)**
    *   **Action:** Manager detects completion and triggers Analysis.
    *   **System:**
        *   Retrieves metrics and plot images for *all* runs in the session.
        *   Constructs a Multimodal Prompt: "Here are the results for 2 scenarios. Scenario A (Low Noise) vs Scenario B (High Noise). Explain the difference in Volatility."
        *   Sends to **Gemini 1.5 Pro**.
    *   **Output:** The LLM generates a textual report explaining the market dynamics observed.

### 4.2 Phase 2: The "Strategist" (Iterative Refinement)
**Goal:** Autonomous strategy generation and improvement loops.

1.  **Strategy Generation (StrategyGenerator Agent)**
    *   **Input:** User Goal ("Market Make with low drawdown") + Previous Analysis (if any).
    *   **Action:** LLM (DeepSeek/Claude) generates Python code conforming to `StrategicAgent` protocol.
    *   **System:** Saves code to `strategy_iterations` table (Generation N).
2.  **Scenario testing**
    *   **System:** Creates `SimulationRuns` testing this new Strategy Code against the defined Scenarios (e.g., specific market conditions).
3.  **Execution (SimulationNode)**
    *   **Action:** Similar to Phase 1, but injects the new Python code into the container.
    *   **System:** `ABIDESAdapter` loads the `StrategicAgent` class and wraps it.
4.  **Evaluation & Feedback**
    *   **Action:** ResultAnalyzer compares current metrics vs previous generation.
    *   **Result:**
        *   If Improved: "Adopt as new baseline."
        *   If Worse: "Analyze why (e.g., 'Overtraded during high volatility')."
5.  **Refinement Loop**
    *   **Action:** Manager passes the "Why" analysis back to the StrategyGenerator.
    *   **Next Step:** Goto Step 1 (Generation N+1).

## 5. Implementation Status & Roadmap

### ✅ COMPLETED: Phase 1 - Data Persistence & Execution Layer

Phase 1 establishes the complete data persistence layer, execution engine, and analysis capabilities.

#### Phase 1.1: Database, Schemas, and Models ✅
**Status:** Complete and tested (92.4% coverage)

**Implemented Components:**

1. **Pydantic Schemas** (`src/rohan/simulation/models/`)
   - `MarketState`: Market conditions snapshot (timestamp, prices, inventory, cash, open orders)
   - `OrderAction`: Trading actions (Side, OrderType, price, quantity)
   - `StrategicAgent`: Protocol for LLM-generated strategies
   - `SimulationMetrics`: Summary statistics (PnL, Sharpe, max drawdown, win rate, volatility)

2. **SQLAlchemy Models** (`src/rohan/framework/database/models.py`)
   - **Strategy Management**: `StrategySession`, `StrategyIteration`, `SimulationScenario`, `SimulationRun`
   - **Data Tables**: `MarketDataL1` (L1 order book), `AgentLog` (event logs), `Artifact` (plots, files)
   - All models support JSONB fields for flexible data storage

3. **Database Connection** (`src/rohan/framework/database/database_connector.py`)
   - `DatabaseConnector` class with connection pooling
   - Support for SQLite (default/dev) and PostgreSQL (production)
   - Configured via `.env`: `DB_CONNECTION_STRING`, `DB_POOL_SIZE`, `DB_MAX_OVERFLOW`

4. **Repository Layer** (`src/rohan/framework/repository.py`)
   - `ArtifactStore` class providing high-level data access
   - **Write ops**: `save_run()`, `save_market_data()`, `save_logs()`, `save_artifact()`
   - **Read ops**: `get_run()`, `get_market_data()`, `get_logs()`, `get_artifacts()`
   - **Management**: `create_session()`, `create_scenario()`, `create_iteration()`, `create_run()`
   - **Updates**: `update_run_status()`
   - Handles NaN values, bulk inserts, JSON sanitization

5. **Database Initialization** (`src/rohan/framework/database/init_db.py`)
   - `initialize_database()`: Create all tables (idempotent)
   - `drop_all_tables()`: Drop all tables
   - `reset_database()`: Drop and recreate
   - CLI: `python -m rohan.framework.init_db [init|drop|reset]`

#### Phase 1.2: Execution Engine ✅
**Status:** Complete and tested

**Implemented Components:**

1. **SimulationEngine** (`src/rohan/framework/simulation_engine.py`)
   - `run_local()`: Execute simulation and persist all results
   - Orchestrates: SimulationService → AbidesOutput → Database
   - Updates run status: PENDING → RUNNING → COMPLETED/FAILED
   - Error handling with status tracking
   - ETL pipeline from `AbidesOutput` to database tables

**Features:**
- Automatic status tracking throughout execution
- Transactional saves (rollback on error)
- Integration with existing `SimulationService`
- Metrics computation via `AnalysisService`

#### Phase 1.3: Analysis Service ✅
**Status:** Complete and tested

**Implemented Components:**

1. **AnalysisService** (`src/rohan/framework/analysis_service.py`)
   - `compute_metrics()`: Calculate PnL, Sharpe, volatility, spread, liquidity
   - `plot_price_series()`: Bid/ask/mid price visualization
   - `plot_volume()`: Volume at best bid/ask
   - `plot_spread()`: Bid-ask spread over time
   - All plots use matplotlib (Agg backend for server environments)

**Features:**
- Handles empty data gracefully
- Returns matplotlib Figure objects for flexibility
- Computes market-wide statistics (volatility, spread, liquidity)
- Custom metrics support via dictionary

#### Phase 1.4: Framework Hardening 🚧
**Status:** TODO

Technical debt and improvements identified during architecture review.

> [!NOTE]
> ~~**Priority 1 (Breaking Bugs):** Fix `reasoning` type annotation in `models.py:55`~~ — **FIXED** (2026-02-07)

**1. Session Management**
- [ ] Use `scoped_session` for thread-local session management
- [ ] Ensure SQLAlchemy internals do not leak outside `ArtifactStore`
- [ ] Add `remove_session()` for cleanup at worker/thread lifecycle end

**2. Schema Fixes**

| Table | Change | File | Line |
|-------|--------|------|------|
| ~~`StrategyIteration`~~ | ~~Fix `reasoning` type~~ | ~~`database/models.py`~~ | ~~55~~ | ✅ DONE |
| `SimulationRun` | Add `RunStatus` enum (PENDING, RUNNING, COMPLETED, FAILED) | `database/models.py` | 68 |
| `SimulationRun` | Add `error_message: str \| None` for simple error text | `database/models.py` | - |
| `SimulationRun` | Add `error_traceback: str \| None` for full traceback (AI debugging) | `database/models.py` | - |
| `SimulationRun` | Add `created_at`, `updated_at` timestamps | `database/models.py` | - |
| All FK relationships | Add `cascade="all, delete-orphan"` | `database/models.py` | all |

**3. Missing Indexes**
```python
Index("ix_agent_logs_agent_type", AgentLog.agent_type)
Index("ix_agent_logs_event_type", AgentLog.event_type)
Index("ix_market_data_time", MarketDataL1.time)
Index("ix_simulation_runs_status", SimulationRun.status)
```

**4. Artifact Storage Abstraction**
- [ ] Create `ArtifactStorage` protocol with `save()` and `load()` methods
- [ ] Implement `LocalStorage` (file system)
- [ ] Replace `Artifact.content: LargeBinary` with `storage_path: String`
- [ ] Prepare abstraction for future S3/GCS backends (no implementation needed now)

**5. Logging**

| Issue | File | Lines |
|-------|------|-------|
| Replace `print()` with logging | `simulation_engine.py` | 29, 67, 70 |
| Use `rohan.*` namespace | All framework modules | - |
| Store error tracebacks in DB | `simulation_engine.py` | 74 (commented out) |

**6. Metrics**

| Issue | File | Lines |
|-------|------|-------|
| Use `None` for unavailable metrics (not `0.0`) | `analysis_service.py` | 47-51 |
| Raise explicit errors for mandatory metrics | `analysis_service.py` | - |

**7. Plot Pipeline (Incomplete)**

| Issue | Current | Expected |
|-------|---------|----------|
| `generate_plots()` returns empty list | `analysis_service.py:154-161` | Should return bytes for DB storage |
| `run_local()` doesn't save plots | `simulation_engine.py` | Should call `save_artifact()` with plot bytes |
| Plot figure to bytes conversion | Not implemented | Add `figure_to_bytes()` utility |

### Testing

**Test Organization** (13 tests, 92.4% coverage):

- **`test_repository.py`** - Database and repository layer (7 tests)
  - Schema validation, model persistence, CRUD operations

- **`test_simulation_engine.py`** - Execution orchestration (2 tests)
  - End-to-end simulation, error handling

- **`test_analysis.py`** - Metrics and visualization (3 tests)
  - Metrics computation, plot generation, artifact storage

- **`test_reproducibility.py`** - Determinism (1 test)
  - Reproducible results with same random seed

Run tests:
```bash
# All tests
uv run pytest tests/ -v

# Specific test file
uv run pytest tests/test_repository.py -v
```

### Database Setup

**SQLite (Default - Development):**
```bash
# No setup required, auto-created at ./rohan.db
python -m rohan.framework.init_db init
```

**PostgreSQL (Production):**
```bash
createdb rohan

# Set in .env:
# DB_CONNECTION_STRING=postgresql://user:password@localhost/rohan
# DB_POOL_SIZE=10
# DB_MAX_OVERFLOW=20

python -m rohan.framework.init_db init
```

### Usage Example

```python
from rohan.framework.repository import ArtifactStore
from rohan.framework.engine import SimulationEngine
from rohan.config import SimulationSettings

# Initialize
repo = ArtifactStore()
engine = SimulationEngine()

# Create session and scenarios
session = repo.create_session("Develop mean-reverting market maker")
scenario = repo.create_scenario(
    name="High Volatility",
    config_override={"noise_agents": 1000},
    session_id=session.session_id
)

# Create iteration
iteration = repo.create_iteration(
    session_id=session.session_id,
    generation_number=1,
    code="class MyAgent: pass",
    reasoning="Initial baseline"
)

# Create and execute run
settings = SimulationSettings()
settings.start_time = "09:30:00"
settings.end_time = "10:00:00"
settings.agents.noise.num_agents = 1000

run = repo.create_run(
    iteration_id=iteration.iteration_id,
    scenario_id=scenario.scenario_id,
    full_config=settings.model_dump(),
    status="PENDING"
)

# Execute
engine.run_local(run.run_id, settings)

# Retrieve results
updated_run = repo.get_run(run.run_id)
print(f"Status: {updated_run.status}")
print(f"Metrics: {updated_run.metrics_summary}")
```

---

### 🚧 TODO: Phase 1.5 - Minimal Vertical Prototype

**Goal:** Validate the key interfaces end-to-end before building the full agentic framework:
1. **Strategy API**: The contract between LLM-generated code and the simulation
2. **Result Interpretation**: What the LLM analyst receives for feedback

```mermaid
graph LR
    Prompt[User Prompt] --> LLM[LLM Generator]
    LLM --> Code[Strategy Code]
    Code --> Validate[AST Validator]
    Validate --> Docker[Docker Execution]
    Docker --> Result[SimulationResult]
    Result --> Analyze[Compute KPIs + Plots]
    Analyze --> Interpret[LLM Interpreter]
    Interpret --> Feedback[Evaluation Report]
```

---

#### Phase 1.5.1: StrategicAgent API Redesign ✅

**Status:** Complete — API implemented in `src/rohan/simulation/models/strategy_api.py`

> [!NOTE]
> **This is a critical integration point.** The API bridges the LLM's "strategy logic" view and ABIDES' "event-driven" reality.

**ABIDES Mapping (Verified):**

| `MarketState` Field | ABIDES Source | Notes |
|---------------------|---------------|-------|
| `timestamp_ns` | `current_time` param in callbacks | Nanoseconds |
| `best_bid` / `best_ask` | `TradingAgent.get_known_bid_ask(symbol)` | Cached L1 data |
| `last_trade` | `order_book.last_trade` | From market data subscription |
| `inventory` | `TradingAgent.holdings[symbol]` | Signed position |
| `cash` | `TradingAgent.holdings["CASH"]` | In cents |
| `open_orders` | `TradingAgent.orders.values()` | Active limit orders |

**Lifecycle Mapping:**
- `kernel_starting()` → `initialize(config: AgentConfig)`
- `receive_message(MarketDataMsg)` → `on_market_data(state: MarketState)`
- `receive_message(OrderExecutedMsg)` → `on_order_update(update: Order)`

**Implemented API:**
```python
class StrategicAgent(Protocol):
    """Framework-agnostic trading strategy interface."""

    def initialize(self, config: AgentConfig) -> None:
        """Called once at start. Set up initial state."""
        ...

    def on_market_data(self, state: MarketState) -> list[OrderAction]:
        """Called when new market data is available (e.g., L1 update)."""
        ...

    def on_order_update(self, update: Order) -> list[OrderAction]:
        """Called when an order status changes (filled, cancelled)."""
        ...
```

**Implemented Models:**
- `Side`, `OrderType`, `OrderStatus` enums
- `Order`: Full order state including `status`, `filled_quantity`
- `MarketState`: Agent's view of market + portfolio
- `OrderAction`: Strategy's trading decisions
- `AgentConfig`: Initialization parameters

**Deliverables:**
- [x] Define `StrategicAgent` protocol and data models
- [x] Ensure `MarketState` maps correctly to `TradingAgent.holdings` and `TradingAgent.orders`

---

#### Phase 1.5.2: ABIDES Adapter & Injection ✅

**Status:** Complete — Implemented in `src/rohan/simulation/abides_impl/strategic_agent_adapter.py`

**Architecture:**

| Component | Changes | Purpose |
|-----------|---------|---------|
| `StrategicAgentAdapter` | New class | Wraps `StrategicAgent` protocol, extends ABIDES `TradingAgent` |
| `SimulationService` | Added `strategy` param | Entry point for injection |
| `SimulationRunnerAbides` | Added `strategy` param | Passes strategy to config mapper |
| `AbidesConfigMapper` | Added `strategy` param | Injects adapter into agent list |

**Lifecycle Mapping (Implemented):**
- `kernel_starting()` → `strategy.initialize(config: AgentConfig)`
- `wakeup()` → `get_current_spread()` → triggers `QuerySpreadResponseMsg`
- `receive_message(QuerySpreadResponseMsg)` → `strategy.on_market_data(state: MarketState)`
- Strategy returns `list[OrderAction]` → `place_limit_order()` / `place_market_order()` / `cancel_order()`

**Key Implementation Details:**
```python
class StrategicAgentAdapter(TradingAgent):
    def get_wake_frequency(self) -> int:  # Required by TradingAgent
        return self.wake_up_freq_ns

    def _build_market_state(self, current_time: int) -> MarketState:
        bid, _, ask, _ = self.get_known_bid_ask(self.symbol)  # 4-tuple return
        return MarketState(
            timestamp_ns=current_time,
            best_bid=bid, best_ask=ask,
            inventory=self.holdings.get(self.symbol, 0),
            cash=self.holdings.get("CASH", 0),
            open_orders=[...],
        )
```

**Tests:** `tests/test_strategic_agent.py`
- `test_strategy_receives_initialize_callback` ✅
- `test_strategy_receives_market_data_callbacks` ✅
- `test_strategy_market_state_has_valid_data` ✅
- `test_strategy_can_place_orders` ✅
- `test_simulation_without_strategy_still_works` ✅
- `test_adapter_can_be_instantiated` ✅

**Deliverables:**
- [x] `StrategicAgentAdapter` implementation
- [x] `SimulationService` refactoring for dynamic strategy injection
- [x] `AbidesConfigMapper` refactoring
- [x] Integration test verifying the strategy receives callbacks

---

#### Phase 1.5.3: Sandboxed Execution via Limited Interpreter 🔴

**Goal:** Execute LLM-generated strategy code safely without full Docker complexity.

> [!NOTE]
> **Why not Docker for MVP?** The ABIDES `end_state` contains Python objects (agents, order book, numpy arrays) that aren't JSON-serializable. A limited in-process interpreter avoids serialization while providing "good enough" isolation for a thesis PoC. Docker can be added in Phase 3 for production.

**Security Model:**

1. **AST Validation** — Block dangerous imports and builtins before execution
2. **Restricted Namespace** — Only expose safe builtins + framework types
3. **Timeout** — Kill execution after configurable limit

**Implementation:**

```python
import ast
import signal
from typing import Any

ALLOWED_IMPORTS = {"math", "numpy", "collections"}
BLOCKED_BUILTINS = {"exec", "eval", "compile", "open", "input", "__import__",
                    "getattr", "setattr", "delattr", "globals", "locals"}

class StrategyValidator:
    """Validates LLM-generated strategy code via AST analysis."""

    def validate(self, code: str) -> ValidationResult:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(is_valid=False, errors=[f"Syntax error: {e}"])

        errors = []
        errors.extend(self._check_imports(tree))
        errors.extend(self._check_blocked_names(tree))
        errors.extend(self._check_protocol_compliance(tree))

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def _check_imports(self, tree: ast.AST) -> list[str]:
        errors = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] not in ALLOWED_IMPORTS:
                        errors.append(f"Blocked import: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] not in ALLOWED_IMPORTS:
                    errors.append(f"Blocked import: {node.module}")
        return errors

    def _check_blocked_names(self, tree: ast.AST) -> list[str]:
        errors = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in BLOCKED_BUILTINS:
                errors.append(f"Blocked builtin: {node.id}")
        return errors

    def _check_protocol_compliance(self, tree: ast.AST) -> list[str]:
        """Check that code defines a class with required methods."""
        errors = []
        has_class = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                has_class = True
                methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
                if "on_market_data" not in methods:
                    errors.append("Missing required method: on_market_data")
        if not has_class:
            errors.append("No class definition found")
        return errors


def execute_strategy_safely(
    strategy_code: str,
    settings: SimulationSettings,
    timeout_seconds: int = 300
) -> SimulationResult:
    """Execute strategy code in a restricted namespace with timeout."""

    # Validate first
    validator = StrategyValidator()
    validation = validator.validate(strategy_code)
    if not validation.is_valid:
        return SimulationResult(error=f"Validation failed: {validation.errors}")

    # Create restricted namespace with safe builtins only
    safe_builtins = {k: v for k, v in __builtins__.items()
                     if k not in BLOCKED_BUILTINS}

    namespace: dict[str, Any] = {
        "__builtins__": safe_builtins,
        # Framework types available to LLM code
        "StrategicAgent": StrategicAgent,
        "MarketState": MarketState,
        "OrderAction": OrderAction,
        "OrderUpdate": OrderUpdate,
        "Side": Side,
    }

    # Compile and execute
    try:
        exec(strategy_code, namespace)
    except Exception as e:
        return SimulationResult(error=f"Execution error: {e}")

    # Find the strategy class
    strategy_class = None
    for name, obj in namespace.items():
        if isinstance(obj, type) and name != "StrategicAgent":
            if hasattr(obj, "on_market_data"):
                strategy_class = obj
                break

    if not strategy_class:
        return SimulationResult(error="No valid StrategicAgent implementation found")

    # Run simulation with timeout
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Simulation exceeded {timeout_seconds}s timeout")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        result = SimulationService().run_simulation(
            settings,
            strategy=strategy_class()
        )
    finally:
        signal.alarm(0)  # Cancel alarm

    return result
```

**Deliverables:**
- [ ] `StrategyValidator` class with AST checks
- [ ] `execute_strategy_safely()` function with timeout
- [ ] `ValidationResult` model
- [ ] Unit tests for blocked imports/builtins

> [!TIP]
> **Phase 3 Enhancement:** For production or cloud deployment, wrap this in Docker for true process isolation and resource limits.

---

#### Phase 1.5.4: Agent-Specific KPIs 🔴

**Goal:** Compute performance metrics for the strategic agent, not just market-wide stats.

**Agent KPIs:**

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Total PnL** | `realized_pnl + (inventory × mid_price)` | Overall performance |
| **Sharpe Ratio** | `mean(returns) / std(returns) × √N` | Risk-adjusted returns |
| **Max Drawdown** | `max(peak - trough) / peak` | Worst loss from peak |
| **Trade Count** | Count of `ORDER_EXECUTED` events | Activity level |
| **Fill Rate** | `filled_orders / placed_orders` | Execution quality |
| **End Inventory** | Final position size | Risk exposure |
| **Avg Hold Time** | Mean time between open and close | Trading style |

**Baseline Comparison (Market Impact):**

For each evaluation, run **two simulations**:
1. **With Strategy**: Scenario + LLM-generated agent
2. **Baseline**: Same scenario, agent disabled or replaced with passive agent

Compute delta metrics:
```python
class ComparisonResult(BaseModel):
    strategy_metrics: AgentMetrics
    baseline_metrics: MarketMetrics
    market_impact: dict  # e.g., {"volatility_delta": 0.02, "spread_delta": 0.001}
```

**Implementation:**
```python
def compute_agent_metrics(output: SimulationOutput, agent_id: int) -> AgentMetrics:
    logs = output.get_logs_by_agent()[agent_id]
    # Parse order events, compute PnL, etc.
    ...
```

**Deliverables:**
- [ ] `AgentMetrics` model in `src/rohan/simulation/models/`
- [ ] `compute_agent_metrics()` function
- [ ] `run_with_baseline()` utility for comparison runs
- [ ] `ComparisonResult` model

---

#### Phase 1.5.5: Structured Summary for LLM 🔴

**Goal:** Create a rich but compact summary for the Interpreter Agent.

**RunSummary Model:**
```python
class RunSummary(BaseModel):
    """Structured summary for LLM interpretation."""

    # Numeric KPIs
    agent_metrics: AgentMetrics
    market_metrics: MarketMetrics
    comparison: ComparisonResult | None  # If baseline was run

    # Visualizations (base64 PNG)
    price_chart: str
    spread_chart: str
    pnl_chart: str

    # Trade summary
    trade_count: int
    notable_trades: list[TradeSummary]  # Largest wins/losses

    # Execution metadata
    duration_seconds: float
    error: str | None
```

**Prompt Template for Interpreter:**
```
You are analyzing the results of a trading strategy simulation.

## Strategy Goal
{goal_description}

## Performance Metrics
- Total PnL: ${agent_metrics.total_pnl:,.2f}
- Sharpe Ratio: {agent_metrics.sharpe_ratio:.2f}
- Max Drawdown: {agent_metrics.max_drawdown:.1%}
- Trade Count: {agent_metrics.trade_count}
- Fill Rate: {agent_metrics.fill_rate:.1%}

## Market Impact (vs Baseline)
- Volatility Change: {comparison.volatility_delta:+.2%}
- Spread Change: {comparison.spread_delta:+.4f}

## Charts
[Price Chart]
[PnL Chart]

## Analysis Task
1. Summarize the strategy's performance
2. Identify strengths and weaknesses
3. Suggest specific improvements for the next iteration
```

**Deliverables:**
- [ ] `RunSummary` model
- [ ] `figure_to_bytes()` utility (already in plan)
- [ ] `generate_summary()` function
- [ ] Prompt template for Interpreter

---

#### Phase 1.5.6: Single Iteration Pipeline 🔴

**Goal:** Wire everything together for a single generate → validate → execute → interpret cycle.

```python
def single_iteration(
    goal: str,
    scenario: SimulationSettings,
    previous_feedback: str | None = None
) -> IterationResult:
    """Execute one iteration of strategy development."""

    # 1. Generate strategy code
    code = llm_generate_strategy(goal, previous_feedback)

    # 2. Validate
    validation = validate_strategy_code(code)
    if not validation.is_valid:
        return IterationResult(error=validation.errors, code=code)

    # 3. Execute in Docker (strategy + baseline)
    strategy_result = run_in_docker(code, scenario)
    baseline_result = run_in_docker(PASSIVE_AGENT_CODE, scenario)

    # 4. Compute KPIs
    summary = generate_summary(strategy_result, baseline_result)

    # 5. LLM Interpretation
    feedback = llm_interpret_results(summary)

    return IterationResult(
        code=code,
        summary=summary,
        feedback=feedback,
    )
```

**Deliverables:**
- [ ] `single_iteration()` orchestration function
- [ ] `validate_strategy_code()` with AST checks
- [ ] LLM prompts for generation and interpretation
- [ ] End-to-end test with a simple strategy goal

---

### 🚧 TODO: Phase 2 - LangGraph Orchestration

Phase 2 implements the autonomous agent loop using LangGraph with **in-memory execution** and **async persistence**.

#### Core Architecture

```mermaid
graph TD
    subgraph "LangGraph Process"
        A[SimulationNode] -->|Direct call| B[SimulationService]
        B -->|Returns| C[SimulationResult]
        C -->|In-memory| D[Interpreter Agent]
        D --> E[Comparator Agent]
        E --> F[Synthesizer Agent]
        F --> G[Final Report]
    end

    C -.->|Fire & forget| H[(Database)]
```

**Key Principles:**
- **In-memory first**: LangGraph nodes call `SimulationService.run_simulation()` directly, returning [`SimulationResult`](../src/rohan/simulation/models/simulation_result.py) in-memory
- **No DB polling**: All agent communication happens through LangGraph state
- **Async persistence**: Database writes are fire-and-forget via background thread
- **Result pattern**: Simulations never throw — errors are returned in `SimulationResult.error`

**Persistence Strategy (Hybrid):**
- **LangGraph PostgreSQL checkpointer**: Stores `AgentState` after each node (enables resume, UI observability)
- **Async data persistence**: Simulation data (`SimulationResult`, metrics, plots) written via `persist_async()` to custom tables
- **UI access**: Query `strategy_sessions`, `strategy_iterations`, `simulation_runs` for KPI data; query LangGraph checkpoint table for current workflow state

#### Phase 2.1: Execution Model

**Direct Execution (No Docker)**
```python
def simulate(settings: SimulationSettings) -> SimulationResult:
    """Execute simulation directly. Never throws."""
    return SimulationService().run_simulation(settings)
```

**Parallel Execution**
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_simulations_parallel(settings_list: list[SimulationSettings]) -> list[SimulationResult]:
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(simulate, s): s for s in settings_list}
        for future in as_completed(futures):
            result = future.result()  # SimulationResult (never throws)
            results.append(result)
            persist_async(result)  # Fire-and-forget
    return results
```

**Async Persistence**
```python
_persist_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="persist")

def persist_async(result: SimulationResult) -> None:
    """Fire-and-forget database write. Errors logged, not raised."""
    def _persist():
        try:
            engine = SimulationEngine()
            engine.save_result(result)
        except Exception as e:
            logger.error(f"Persistence failed: {e}")

    _persist_executor.submit(_persist)
```

**Deliverables:**
- [ ] In-memory simulation execution via `SimulationService`
- [ ] `ThreadPoolExecutor` for parallel scenario execution
- [ ] Async persistence with `persist_async()`

**Image Handling (Multimodal Analysis):**

Plots are stored as binary in the existing `Artifact` table and passed to LLMs via LangChain multimodal messages:

```python
# Storage: figure_to_bytes utility
import io
def figure_to_bytes(fig: Figure, format: str = "png") -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=format)
    buf.seek(0)
    return buf.read()

# Persist: save to Artifact table
repo.save_artifact(run_id, "IMAGE", "price_series.png", figure_to_bytes(fig))

# LLM: pass as base64 in LangChain message
import base64
from langchain.schema import HumanMessage

message = HumanMessage(content=[
    {"type": "text", "text": "Analyze this price chart:"},
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(artifact.content).decode()}"}}
])

# UI: Streamlit displays bytes directly
st.image(artifact.content)  # No conversion needed
```

---

#### Phase 2.2: Analysis Agent Topology

**Three Agent Types:**

| Agent | Purpose | Input | Output |
|-------|---------|-------|--------|
| **Interpreter** | "What happened here?" | 1 SimulationResult | Scenario summary |
| **Comparator** | "How do A and B differ?" | 2 Summaries | Comparative insight |
| **Synthesizer** | "What's the conclusion?" | All summaries + insights | Final report |

**When to Use:**

| Scenario | Agents Used |
|----------|-------------|
| Single run | Interpreter only |
| 2 runs (A vs B) | 2 Interpreters → 1 Comparator |
| N runs (sweep) | N Interpreters → (N-1) Comparators → Synthesizer |
| Strategy iteration | 2 Interpreters (gen N, N-1) → 1 Comparator → Feedback |

**LangGraph Flow:**
```python
def analysis_flow(state: AgentState) -> AgentState:
    results = state["simulation_results"]

    # 1. Interpret each (parallel)
    summaries = [interpret(r) for r in results]

    # 2. Compare pairs (if multiple)
    insights = []
    if len(summaries) > 1:
        for i in range(len(summaries) - 1):
            insights.append(compare(summaries[i], summaries[i+1]))

    # 3. Synthesize
    report = synthesize(summaries, insights)

    return {**state, "final_report": report}
```

**Deliverables:**
- [ ] `InterpreterAgent` for single-scenario analysis
- [ ] `ComparatorAgent` for pairwise comparison
- [ ] `SynthesizerAgent` for final report generation
- [ ] Multimodal analysis with Gemini (text + plots)

---

#### Phase 2.3: LangGraph State & Nodes

**State Schema:**
```python
from typing import TypedDict
from uuid import UUID

class AgentState(TypedDict):
    # Session context
    session_id: UUID
    goal_description: str

    # Execution
    pending_settings: list[SimulationSettings]
    simulation_results: list[SimulationResult]  # In-memory results

    # Analysis
    scenario_summaries: list[str]
    comparison_insights: list[str]
    final_report: str | None

    # Control flow
    iteration_number: int
    max_iterations: int
    should_continue: bool
```

**Deliverables:**
- [ ] Typed `AgentState` schema
- [ ] State persistence via LangGraph PostgreSQL checkpointer
- [ ] Nodes: `SimulationNode`, `InterpreterNode`, `ComparatorNode`, `SynthesizerNode`

---

#### Phase 2.4: The "Observer" (Running Baselines)
**Goal:** Run simulations and provide intelligent analysis without generating new strategies.

**Workflow:**
1. User provides analysis goal (e.g., "Analyze impact of Noise Agent volume on Volatility")
2. System creates multiple `SimulationSettings` with different parameters
3. `SimulationNode` executes runs in parallel, returns `SimulationResult` list
4. `InterpreterAgent` summarizes each scenario
5. `ComparatorAgent` analyzes differences between scenarios
6. `SynthesizerAgent` generates final report

**Deliverables:**
- [ ] LangGraph workflow for baseline analysis
- [ ] Session-level result aggregation

---

#### Phase 2.5: The "Strategist" (Iterative Refinement)
**Goal:** Autonomous strategy generation and improvement loops.

> [!IMPORTANT]
> **Phase 2.5 generates and validates code only — execution requires Phase 3 (Docker sandbox).**
> Invalid code is returned to the LLM with error details for regeneration.

**Workflow:**
1. `StrategyGenerator` (DeepSeek/Claude) creates code conforming to `StrategicAgent` protocol
2. `StrategyValidator` validates code (syntax, imports, protocol compliance)
3. **If validation fails**: Return code + errors to LLM, regenerate (max 3 attempts)
4. **If validation passes**: Save to `strategy_iterations`, proceed to Phase 3 execution
5. `ComparatorAgent` compares current vs previous generation
6. Feedback loop: If improved → adopt; If worse → analyze why and iterate

**Code Validation Pipeline (`StrategyValidator`):**

1. **File Type Validation**: Single Python file only
2. **Syntax Validation**: Parse AST, check for errors
3. **Import Whitelist**: Only allow safe imports (`math`, `numpy`, `pandas`, `rohan.framework.schema`)
4. **Protocol Compliance**: Must implement `StrategicAgent.on_market_data()`
5. **Security Checks**: No file I/O, network, or code execution
6. **Code Quality**: Warnings for long methods, missing docstrings

**Error Recovery:**
```python
class AgentState(TypedDict):
    # ... existing fields ...
    consecutive_validation_failures: int  # Reset on successful validation
    consecutive_regression_count: int     # Generations that performed worse

def should_stop(state: AgentState) -> bool:
    return (
        state["iteration_number"] >= state["max_iterations"] or
        state["consecutive_validation_failures"] >= 3 or
        state["consecutive_regression_count"] >= 3
    )
```

**Deliverables:**
- [ ] `StrategyGenerator` node with LLM integration
- [ ] `StrategyValidator` with validation pipeline
- [ ] Validation error formatting for LLM feedback (re-generation loop)
- [ ] Multi-generation comparison and feedback loop
- [ ] Error recovery with consecutive failure counters
- [ ] Convergence criteria and stopping conditions

### Phase 3: Docker Sandbox for LLM-Generated Code
**Goal:** Secure execution of LLM-generated strategy code in isolated containers.

> [!NOTE]
> Docker is only needed when executing untrusted LLM-generated code. Local simulations with built-in agents run directly without containers.

- [ ] Docker container with minimal Python environment
- [ ] Strategy code injection via volume mount
- [ ] Container captures `SimulationResult` and returns via stdout/file
- [ ] Resource limits (CPU, memory, timeout)
- [ ] Container orchestration for parallel execution

### Phase 4: Production Features
- [ ] Web UI for session monitoring
- [ ] Real-time progress tracking
- [ ] Strategy leaderboard and comparison
- [ ] Export/import of strategies
- [ ] Advanced visualization dashboard
