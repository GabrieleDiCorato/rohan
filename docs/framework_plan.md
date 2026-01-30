# Agentic Simulation Framework - Technical Specification (SPEC_DEV.md)

## 1. Overview
This framework provides an autonomous loop for generating, testing, and refining specific trading strategies using `abides-rohan`. It leverages **LangGraph** for state management, **PostgreSQL** for relational data storage, and **Docker** for isolated execution.

## 2. Core Architecture

### 2.1 Technology Stack
*   **Orchestration:** LangGraph (Python)
*   **Simulation Engine:** abides-rohan (Existing)
*   **Database:** PostgreSQL
    *   *Features:* Relational Tables for Metrics/Logs/Series, BYTEA for Images/Code.
*   **LLM Integration:**
    *   *Analysis:* Google Gemini 1.5 Pro (via `langchain-google-genai`)
    *   *Code Generation:* Claude 3.5 (via `langchain-openai` / OpenRouter)
*   **Execution:** Docker containers (via `docker-py`).

### 2.2 System Components & Communication


```mermaid
graph TD
    User[User / CLI] -->|Start Session| Manager[LangGraph Manager]
    Manager -->|Status Update| DB[(PostgreSQL)]

    subgraph "Strategy Generation"
        Manager -->|1. Generate Code| StratGen[Strategy Generator]
        StratGen -->|Save Code| DB
    end

    subgraph "Simulation Orchestration"
        Manager -->|2. Queue Scenarios| SimNode[Simulation Node]
        SimNode -->|Read Scenarios| DB
        SimNode -->|3. Docker Run| Worker[Simulation Worker Container]
        Worker -->|Writes Data| DB
    end

    subgraph "Analysis"
        Manager -->|4. Trigger| Analyzer[Result Analyzer Agent]
        Analyzer -->|Reads Metrics/Logs| DB
        Analyzer -->|5. Feedback| Manager
    end
```
(install Mermaid if the graph is not visualized correctly)

**Communication Protocol:**
*   **Manager <-> Worker:** Asynchronous via **Database**.
    *   The Manager creates [SimulationRun](file:///c:/Users/gabri/workspace/rohan/src/rohan/simulation/simulation_runner.py#9-23) rows (Status='PENDING').
    *   The Docker Worker starts, grabs the config from DB, runs, writes results to DB tables, and updates Status='COMPLETED'/'FAILED'.
    *   The Manager polls or waits on the container exit code.
*   **Manager <-> Agents (SimNode/Analyzer):** Shared **LangGraph State** (Python Dictionary in memory) containing `session_id`, `current_strategy_id`, etc.

## 3. Data Models

### 3.1 Strategic Agent Protocol (Framework Agnostic)
Defined in `src/rohan/framework/schema.py`. This is the **ONLY** interface the LLM interacts with.

```python
from typing import Protocol, List, Literal
from pydantic import BaseModel

class MarketState(BaseModel):
    timestamp: int          # Nanoseconds from midnight
    best_bid: float
    best_ask: float
    last_trade_price: float
    inventory: int          # Current position
    cash: float             # Available cash

class OrderAction(BaseModel):
    side: Literal["BUY", "SELL"]
    type: Literal["LIMIT", "MARKET"]
    price: float            # 0.0 for MARKET
    quantity: int

class StrategicAgent(Protocol):
    """
    Protocol for LLM-generated strategies.
    Must be stateless between calls (state stored in instance attributes).
    """
    def on_market_update(self, market_state: MarketState) -> List[OrderAction]:
        ...
```

### 3.2 Database Schema (PostgreSQL)

**Hierarchy:**
`StrategySession` (User goal) -> `StrategyIteration` (One code version) -> `SimulationScenario` (Conditions) -> [SimulationRun](file:///c:/Users/gabri/workspace/rohan/src/rohan/simulation/simulation_runner.py#9-23) (Execution).

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

**Table: `market_data_l1`** (Matches [AbidesOutput](file:///c:/Users/gabri/workspace/rohan/src/rohan/simulation/abides_impl/abides_output.py#26-193) L1 DataFrame)
| Column | Type | Description |
| :--- | :--- | :--- |
| `run_id` | UUID (FK) | Link to run |
| `time` | BIGINT | Nanoseconds from midnight |
| `bid_price` | DOUBLE | Event Price |
| `bid_qty` | INT | Event Size |
| `ask_price` | DOUBLE | Event Price |
| `ask_qty` | INT | Event Size |
| `timestamp` | TIMESTAMP | Human readable time |

**Table: `agent_logs`** (Matches [AbidesOutput](file:///c:/Users/gabri/workspace/rohan/src/rohan/simulation/abides_impl/abides_output.py#26-193) Logs DataFrame)
| Column | Type | Description |
| :--- | :--- | :--- |
| `run_id` | UUID (FK) | Link to run |
| `agent_id` | INT | |
| `agent_type` | TEXT | "ValueAgent", "NoiseAgent" |
| `EventType` | TEXT | "ORDER_SUBMITTED", "ORDER_EXECUTED" |
| `time_placed` | TIMESTAMP | |
| `log_json` | JSONB | The full log dict (contains internal state) |

**Table: `artifacts`**
| Column | Type | Description |
| :--- | :--- | :--- |
| `run_id` | UUID (FK) | Link to run |
| `type` | TEXT | 'IMAGE', 'LOG_FILE' |
| `path` | TEXT | Virtual path |
| `content` | BYTEA | Binary content |

## 4. Workflows

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
        *   Creates [SimulationRun](file:///c:/Users/gabri/workspace/rohan/src/rohan/simulation/simulation_runner.py#9-23) records for each scenario (Status='PENDING').
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

1. **Pydantic Schemas** (`src/rohan/framework/schema.py`)
   - `MarketState`: Market conditions snapshot (timestamp, prices, inventory, cash)
   - `OrderAction`: Trading actions (BUY/SELL, LIMIT/MARKET, price, quantity)
   - `StrategicAgent`: Protocol for LLM-generated strategies
   - `SimulationMetrics`: Summary statistics (PnL, Sharpe, max drawdown, win rate, volatility)

2. **SQLAlchemy Models** (`src/rohan/framework/models.py`)
   - **Strategy Management**: `StrategySession`, `StrategyIteration`, `SimulationScenario`, `SimulationRun`
   - **Data Tables**: `MarketDataL1` (L1 order book), `AgentLog` (event logs), `Artifact` (plots, files)
   - All models support JSONB fields for flexible data storage

3. **Database Connection** (`src/rohan/framework/database.py`)
   - Singleton `Database` class with connection pooling
   - Support for SQLite (default/dev) and PostgreSQL (production)
   - Configured via `.env`: `DB_CONNECTION_STRING`, `DB_POOL_SIZE`, `DB_MAX_OVERFLOW`

4. **Repository Layer** (`src/rohan/framework/repository.py`)
   - `ArtifactStore` class providing high-level data access
   - **Write ops**: `save_run()`, `save_market_data()`, `save_logs()`, `save_artifact()`
   - **Read ops**: `get_run()`, `get_market_data()`, `get_logs()`, `get_artifacts()`
   - **Management**: `create_session()`, `create_scenario()`, `create_iteration()`, `create_run()`
   - **Updates**: `update_run_status()`
   - Handles NaN values, bulk inserts, JSON sanitization

5. **Database Initialization** (`src/rohan/framework/init_db.py`)
   - `initialize_database()`: Create all tables (idempotent)
   - `drop_all_tables()`: Drop all tables
   - `reset_database()`: Drop and recreate
   - CLI: `python -m rohan.framework.init_db [init|drop|reset]`

#### Phase 1.2: Execution Engine ✅
**Status:** Complete and tested

**Implemented Components:**

1. **SimulationEngine** (`src/rohan/framework/engine.py`)
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

1. **AnalysisService** (`src/rohan/framework/analysis.py`)
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

### 🚧 TODO: Phase 2 - LangGraph Orchestration

Phase 2 will implement the autonomous agent loop using LangGraph.

#### Phase 2.1: LangGraph State & Nodes
- [ ] Define LangGraph state schema
- [ ] Implement nodes: StrategyGenerator, SimulationNode, ResultAnalyzer
- [ ] State persistence and checkpointing

#### Phase 2.2: The "Observer" (Running Baselines)
**Goal:** Verify system can run simulations and provide intelligent analysis without generating new strategies.

Workflow:
1. User provides analysis goal (e.g., "Analyze impact of Noise Agent volume on Volatility")
2. Manager creates multiple scenarios with different parameters
3. SimulationNode executes runs for each scenario
4. ResultAnalyzer (Gemini 1.5 Pro) compares results and generates insights

**Deliverables:**
- [ ] LangGraph workflow for baseline analysis
- [ ] Multimodal analysis with Gemini (text + plots)
- [ ] Session-level result aggregation

#### Phase 2.3: The "Strategist" (Iterative Refinement)
**Goal:** Autonomous strategy generation and improvement loops.

Workflow:
1. StrategyGenerator (DeepSeek/Claude) creates code conforming to `StrategicAgent` protocol
2. **Code Validation**: Multi-stage validation before execution
3. Code injected into simulation runs across test scenarios
4. ResultAnalyzer compares vs previous generation
5. Feedback loop: If improved → adopt; If worse → analyze why
6. Manager passes analysis back to StrategyGenerator for next iteration

**Code Validation Pipeline:**

Before any generated strategy is executed, it passes through a validation pipeline that ensures safety and correctness:

**`StrategyValidator`** (`src/rohan/framework/strategy_validator.py`)

1. **File Type Validation**
   - Verify only a single Python script is produced
   - Check file extension and content type
   - Reject multiple files or non-Python content
   - **Error message**: "Expected a single Python file, but received {file_count} files of types {types}"

2. **Syntax & Compilation Validation**
   - Parse Python AST (Abstract Syntax Tree)
   - Check for syntax errors
   - Verify code compiles without errors
   - **Error message**: "Syntax error at line {line_no}, column {col}: {error_detail}. Please fix the syntax before resubmitting."

3. **Import Whitelist Validation**
   - Extract all import statements from AST
   - Check against allowed imports whitelist:
     ```python
     ALLOWED_IMPORTS = {
         # Standard library (safe subset)
         "math", "statistics", "collections", "itertools", "functools",
         "typing", "dataclasses", "enum", "abc",
         # Scientific computing
         "numpy", "pandas",
         # Framework-specific
         "rohan.framework.schema",  # MarketState, OrderAction
     }
     ```
   - Reject dangerous modules: `os`, `sys`, `subprocess`, `eval`, `exec`, `__import__`
   - **Error message**: "Forbidden import detected: '{module_name}'. Allowed imports are: {allowed_list}. Please remove or replace this import."

4. **Protocol Compliance Validation**
   - Verify class implements `StrategicAgent` protocol
   - Check `on_market_update(self, market_state: MarketState) -> List[OrderAction]` method exists
   - Validate method signature matches exactly
   - Check return type annotations
   - **Error message**: "Class '{class_name}' does not properly implement StrategicAgent protocol. Missing or incorrect method: {details}. Required: `def on_market_update(self, market_state: MarketState) -> List[OrderAction]`"

5. **Security & Safety Checks**
   - Scan for dangerous operations:
     - File I/O (`open`, `read`, `write`)
     - Network operations (`socket`, `urllib`, `requests`)
     - Code execution (`eval`, `exec`, `compile` with user input)
     - System commands (`os.system`, `subprocess`)
   - Check for infinite loops (basic heuristics: `while True` without clear break)
   - Verify no global state modifications outside class scope
   - **Error message**: "Dangerous operation detected: {operation} at line {line_no}. Strategy code should only process market data and return orders. Forbidden operations: {forbidden_list}"

6. **Code Quality Checks** (Warnings, non-blocking)
   - Check for extremely long methods (>100 lines)
   - Verify docstrings exist
   - Check for unused imports
   - **Warning message**: "Code quality concern: {issue}. While not blocking execution, consider: {suggestion}"

**Validation Response Format:**

```python
@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationWarning]

@dataclass
class ValidationError:
    stage: str  # "syntax", "imports", "protocol", "security"
    line_number: Optional[int]
    message: str
    suggestion: str  # Actionable fix suggestion

@dataclass
class ValidationWarning:
    category: str
    message: str
```

**Error Feedback to LLM:**

When validation fails, the error message is formatted for LLM understanding:

```
VALIDATION FAILED - Please fix the following issues:

[SYNTAX ERROR] Line 15, Column 8
  → Missing colon after if statement
  Fix: Add ':' at the end of line 15

[FORBIDDEN IMPORT] Line 3
  → Import 'os' is not allowed
  Allowed imports: math, numpy, pandas, rohan.framework.schema
  Fix: Remove 'import os' or use allowed alternatives

[PROTOCOL VIOLATION] Class 'MyStrategy'
  → Method signature mismatch
  Expected: def on_market_update(self, market_state: MarketState) -> List[OrderAction]
  Found: def on_market_update(self, state)
  Fix: Update method signature to match the protocol exactly

Please generate a corrected version of the code addressing all issues above.
```

**Deliverables:**
- [ ] StrategyGenerator node with LLM integration
- [ ] **StrategyValidator with comprehensive validation pipeline**
- [ ] **Validation result formatting for LLM feedback**
- [ ] Code injection into simulation worker
- [ ] Multi-generation comparison and feedback loop
- [ ] Convergence criteria and stopping conditions

### Phase 3: Docker & Distributed Execution
- [ ] Docker container for simulation workers
- [ ] Worker pulls config from DB, writes results back
- [ ] Container orchestration (local/cloud)
- [ ] Parallel scenario execution

### Phase 4: Production Features
- [ ] Web UI for session monitoring
- [ ] Real-time progress tracking
- [ ] Strategy leaderboard and comparison
- [ ] Export/import of strategies
- [ ] Advanced visualization dashboard
