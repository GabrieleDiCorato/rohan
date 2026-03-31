# ABIDES — Declarative Configuration System

This document covers the pluggable, AI-friendly configuration system for ABIDES
market simulations. It replaces (or complements) the procedural `build_config()`
functions with declarative Pydantic models, YAML/JSON serialization, and
composable templates.

> **Interactive tutorial:** See [`notebooks/demo_Config_System.ipynb`](../notebooks/demo_Config_System.ipynb)
> for a hands-on walkthrough of every feature below.

---

## Quick Start

```python
from abides_markets.config_system import SimulationBuilder
from abides_markets.simulation import run_simulation

# Build an immutable config from a template
config = (SimulationBuilder()
    .from_template("rmsc04")
    .market(ticker="AAPL")
    .seed(42)
    .build())

# run_simulation() compiles a fresh runtime dict internally, runs the
# simulation, and returns a typed, immutable SimulationResult.
result = run_simulation(config)
```

The same `SimulationConfig` can be passed to `run_simulation()` any number
of times — each call compiles fresh agents and oracle, so results are always
reproducible.

<details>
<summary>Low-level path (direct Kernel access)</summary>

```python
from abides_markets.config_system import SimulationBuilder, compile
from abides_core import abides

config = (SimulationBuilder()
    .from_template("rmsc04")
    .market(ticker="AAPL")
    .seed(42)
    .build())

runtime = compile(config)       # fresh runtime dict — consumed once
end_state = abides.run(runtime)
```

</details>

---

## Architecture

The config system has four layers:

| Layer | Module | Purpose |
|-------|--------|---------|
| **Models** | `models.py` | Pydantic models for `SimulationConfig` and its sections |
| **Registry** | `registry.py` | Agent type registration with `@register_agent` decorator |
| **Builder** | `builder.py` | Fluent API: `SimulationBuilder().from_template(...).build()` |
| **Compiler** | `compiler.py` | Converts `SimulationConfig` → Kernel runtime dict |

Supporting modules: `templates.py` (composable presets), `serialization.py`
(YAML/JSON I/O), `agent_configs.py` (per-agent Pydantic configs with factories),
`builtin_registrations.py` (auto-registers 5 built-in agents).

---

## Configuration Structure

`SimulationConfig` has four clearly separated sections:

```yaml
market:
  ticker: ABM
  date: "20210205"
  start_time: "09:30:00"
  end_time: "10:00:00"
  oracle:                        # REQUIRED — set explicitly or use null
    type: sparse_mean_reverting
    r_bar: 100000          # $1000.00 in cents
    mean_reversion_half_life: "48d"
    fund_vol: 5.0e-05
  exchange:
    book_logging: true
    book_log_depth: 10
    computation_delay: 0

agents:
  noise:
    enabled: true
    count: 1000
    params: {}
  value:
    enabled: true
    count: 102
    params:
      r_bar: 100000
      mean_reversion_half_life: "4.8d"
      mean_wakeup_gap: "175s"
      computation_delay: 200   # per-agent-type override (ns)
  momentum:
    enabled: true
    count: 12
    params:
      wake_up_freq: "37s"

infrastructure:
  latency:
    type: deterministic
  default_computation_delay: 50   # global default (ns)

simulation:
  seed: 42
  log_level: INFO
  log_orders: true
```

---

## Templates

Templates are composable presets. Base templates provide a full config;
overlay templates add agent groups without replacing existing ones.

| Template | Type | Description |
|----------|------|-------------|
| `rmsc04` | base | Reference config: 1000 Noise, 102 Value, 12 Momentum, 2 MM |
| `liquid_market` | base | High liquidity: 5000 Noise, 200 Value, 25 Momentum, 4 MM |
| `thin_market` | base | Low liquidity: 100 Noise, 20 Value, no MM |
| `with_momentum` | overlay | Adds 12 Momentum agents |
| `with_execution` | overlay | Adds 1 POV Execution agent |

Stack templates: later ones override earlier ones.

```python
config = (SimulationBuilder()
    .from_template("rmsc04")
    .from_template("with_execution")   # adds execution agent
    .seed(42)
    .build())
```

---

## Builder API

The `SimulationBuilder` provides a fluent interface:

```python
config = (SimulationBuilder()
    .from_template("rmsc04")
    .market(ticker="AAPL", date="20220315")
    .oracle(r_bar=150_000)
    .exchange(book_log_depth=20)
    .enable_agent("noise", count=500)
    .enable_agent("value", count=50, r_bar=200_000, computation_delay=100)
    .disable_agent("momentum")
    .agent_computation_delay("noise", 200)  # set per-type delay
    .latency(type="deterministic")
    .computation_delay(75)                  # global default
    .seed(42)
    .log_level("DEBUG")
    .log_orders(True)
    .build())
```

---

## Oracle Configuration

The oracle field in `market` is **required** — there is no implicit default.
Set it to an oracle config dict for simulations with a fundamental-value oracle,
or to `null`/`None` for oracle-less simulations.

### Oracle-present simulation (default pattern)

```python
config = (SimulationBuilder()
    .from_template("rmsc04")       # includes oracle config
    .oracle(r_bar=150_000)         # override oracle params
    .seed(42)
    .build())
```

### Oracle-less simulation (LOB-only agents)

```python
config = (SimulationBuilder()
    .oracle(type=None)             # explicitly no oracle
    .market(opening_price=100_000) # required when oracle is None ($1000.00)
    .enable_agent("noise", count=500)
    .enable_agent("momentum", count=10)
    .seed(42)
    .build())
```

When oracle is `None`:
- `opening_price` is required — it seeds the ExchangeAgent's last-trade price.
- `ValueAgent` cannot be used (compile-time error).
- Only LOB-based agents work: Noise, Momentum, AMM, POV, custom agents.

### ValueAgent parameter auto-inheritance

`ValueAgent`'s Bayesian model parameters (`r_bar`, `kappa`, `sigma_s`) auto-inherit
from the oracle config when not explicitly set in the agent params:

```python
config = (SimulationBuilder()
    .market(ticker="ABM")
    .oracle(type="sparse_mean_reverting", r_bar=200_000, kappa=5e-16, sigma_s=100)
    .enable_agent("value", count=50)  # r_bar/kappa/sigma_s inherited from oracle
    .seed(42)
    .build())
```

Override per-agent when needed:

```python
.enable_agent("value", count=50, r_bar=300_000)  # only r_bar overridden; kappa, sigma_s still inherited
```

`sigma_n` (observation noise) is always agent-specific and defaults to `r_bar / 100`.

### External data oracle injection

For `ExternalDataOracle`, build the oracle externally and inject it:

```python
from abides_markets.oracles import ExternalDataOracle

oracle = ExternalDataOracle(my_data_provider)
config = (SimulationBuilder()
    .oracle_instance(oracle)           # injects pre-built oracle
    .enable_agent("noise", count=500)
    .seed(42)
    .build())

runtime = builder.build_and_compile()  # passes oracle through to runtime
```

---

## Per-Agent Computation Delays

Every agent group can specify a `computation_delay` (in nanoseconds) that
overrides the simulation-level `default_computation_delay`. This controls
how long an agent "thinks" after each wakeup or message receipt.

**Use cases:**
- Market makers with fast computation (low delay)
- Background noise agents with high delay
- Execution agents with realistic processing times

```python
# Via builder
config = (SimulationBuilder()
    .from_template("rmsc04")
    .enable_agent("adaptive_market_maker", count=2, computation_delay=10)
    .enable_agent("noise", count=1000, computation_delay=500)
    .computation_delay(50)   # default for agents without override
    .seed(42)
    .build())

# Via YAML
# agents:
#   adaptive_market_maker:
#     enabled: true
#     count: 2
#     params:
#       computation_delay: 10
```

The compiler produces a `per_agent_computation_delays` dict in the runtime
config, which the Kernel applies on initialization.

---

## Agent Registry

Agent types self-register via the `@register_agent` decorator or
`registry.register()`. Built-in agents are registered at import time.

When an `agent_class` is provided at registration, `BaseAgentConfig`
auto-generates the `create_agents()` factory by inspecting the agent
constructor and mapping config fields to constructor args by name.

### Registry metadata

Each `AgentRegistryEntry` carries:

| Field | Type | Purpose |
|-------|------|---------|
| `name` | `str` | Unique identifier (e.g. `"noise"`) |
| `config_model` | `type` | Pydantic model for agent params |
| `category` | `str` | One of the keys in `CATEGORIES` |
| `description` | `str` | Human-readable description |
| `agent_class` | `type \| None` | The ABIDES agent class to instantiate |
| `requires_oracle` | `bool` | Whether the agent needs an oracle |
| `typical_count_range` | `tuple[int,int] \| None` | Suggested (min, max) agent count |
| `recommended_with` | `tuple[str,...]` | Agent types that work well alongside this one |

### Registered built-in agents

| Name | Category | Requires Oracle | Typical Count | Recommended With |
|------|----------|----------------|---------------|-----------------|
| `noise` | background | No | 50–5,000 | value, adaptive_market_maker |
| `value` | background | **Yes** | 10–500 | noise |
| `momentum` | strategy | No | 1–50 | noise, value |
| `adaptive_market_maker` | market_maker | No | 1–5 | noise, value |
| `pov_execution` | execution | No | 1–1 | noise, value, adaptive_market_maker |

### Registering a custom agent

```python
from pydantic import Field
from abides_markets.config_system import BaseAgentConfig, register_agent

@register_agent("my_strategy", category="strategy",
                agent_class=MyAgent, description="My custom strategy")
class MyStrategyConfig(BaseAgentConfig):
    threshold: float = Field(default=0.05)
    wake_up_freq: str = Field(default="30s")

    def _prepare_constructor_kwargs(self, kwargs, agent_id, agent_rng, context):
        from abides_core.utils import str_to_ns
        kwargs["wake_up_freq"] = str_to_ns(self.wake_up_freq)
        return kwargs
```

> **Notebook re-execution:** `@register_agent` uses `allow_overwrite=True` by default,
> so re-running a cell that defines a custom agent replaces the previous registration
> silently instead of raising `ValueError`.
>
> When calling `registry.register()` directly, pass `allow_overwrite=True` explicitly
> if you need the same overwrite behaviour.

Parameters follow standard Pydantic conventions:
- **Required**: Fields without defaults → must be provided in config
- **Optional**: Fields with defaults → inherited from base or overridden
- **Inherited**: Subclass fields extend base class fields (`starting_cash`, `log_orders`, `computation_delay`)
- **Validated**: Unknown fields are rejected (`extra="forbid"`)

### Auto-generated factories

When `agent_class` is provided at registration, the base `create_agents()`
implementation:

1. Inspects the agent constructor via `inspect.signature()`
2. Maps config field names → constructor parameter names
3. Injects context arguments: `id`, `name`, `type`, `symbol`, `random_state`
4. Calls `_prepare_constructor_kwargs()` for computed args (e.g., duration string → nanoseconds)
5. Instantiates `count` agents with sequential IDs

Override `_prepare_constructor_kwargs()` for non-trivial mappings.
Override `create_agents()` entirely for agents that don't follow the pattern.

### Eager parameter validation

`build()` validates agent parameters at build-time (not just at compile-time):

```python
# This raises ValueError immediately — no need to wait until compile()
config = (SimulationBuilder()
    .from_template("rmsc04")
    .enable_agent("noise", count=10, unknown_param=42)
    .build())  # ← raises ValueError: Invalid parameters for agent type 'noise'
```

---

## Serialization (YAML / JSON)

```python
from abides_markets.config_system import save_config, load_config

# Save
save_config(config, "my_sim.yaml")
save_config(config, "my_sim.json")

# Load
config = load_config("my_sim.yaml")
```

---

## AI Discoverability API

```python
from abides_markets.config_system import (
    list_agent_types,
    list_templates,
    get_config_schema,
    get_full_manifest,
    validate_config,
    CATEGORIES,
)

# What agent types are available?
list_agent_types()
# → [{"name": "noise", "category": "background", "requires_oracle": False,
#     "typical_count_range": [50, 5000], "recommended_with": ["value", ...],
#     "parameters": {...}}, ...]

# What templates are available?
list_templates()
# → [{"name": "rmsc04", "description": "...", "agent_types": [...]}, ...]

# Full JSON Schema
get_config_schema()

# Complete manifest for auto-generating a configuration dashboard
manifest = get_full_manifest()
# → {"agent_types": [...], "market_config_schema": {...},
#    "oracle_options": [...], "templates": [...], "categories": {...}}
```

### Structured validation

`validate_config()` returns a `ValidationResult` with structured issues
(not just flat error strings). It runs three validation phases:

1. Pydantic structural validation
2. Per-agent param validation against registry config models
3. Cross-agent consistency checks (soft warnings)

```python
result = validate_config(config_dict)

# New API
result.valid          # True if no errors (warnings don't count)
result.errors         # list[ValidationIssue] — severity="error" only
result.warnings       # list[ValidationIssue] — severity="warning" only
for issue in result.issues:
    print(issue.severity, issue.field_path, issue.agent_name, issue.message)

# Backward-compatible dict access still works
result["valid"]       # True/False
result["errors"]      # list[str] — only present when invalid
```

### Category taxonomy

The `CATEGORIES` constant provides display metadata for agent categories:

```python
CATEGORIES = {
    "background":    {"label": "Liquidity & Background", "sort_order": 1, ...},
    "market_maker":  {"label": "Market Makers",          "sort_order": 2, ...},
    "strategy":      {"label": "Trading Strategies",     "sort_order": 3, ...},
    "execution":     {"label": "Execution Algorithms",   "sort_order": 4, ...},
}
```

### Cross-agent validation warnings

`build()` now runs `_cross_validate()` after Pydantic validation, emitting
`UserWarning`s for semantically suspect configurations:

- Market maker with no noise/value agents (empty order book)
- POV execution with <10 background agents
- `start_time >= end_time` (empty trading window)
- POV execution offsets consuming the entire market window
- Total agent count >10,000 (performance)
- No enabled agents at all

---

## Backward Compatibility

The old `build_config()` functions (e.g., `rmsc04.build_config(seed=42)`)
continue to work unchanged. The new system produces the same runtime dict
format, so `abides.run()`, gymnasium environments, and `config_add_agents()`
all work with either approach.

---

## Seed Derivation (v2.4.0+)

The compiler derives per-component random seeds via **identity-based
SHA-256 hashing**, not a sequential draw from a shared RNG.  Each
component's seed depends only on `(master_seed, component_name, index)`:

```
sha256(f"{seed}:{component}:{index}") → first 4 bytes → uint32 seed
```

This provides two guarantees:

- **Order independence** — agent groups declared in any order produce
  identical seeds.  (`SimulationConfig` sorts groups alphabetically.)
- **Composition invariance** — adding, removing, or resizing an agent
  group does not shift any other component's seed.

See [PARALLEL_SIMULATION_GUIDE.md](PARALLEL_SIMULATION_GUIDE.md) for the
full derivation tree.

> **Breaking change from v2.3.0:** the same master seed produces different
> agent `random_state` objects than before.  This is intentional — the old
> sequential scheme had order-dependent and composition-dependent bugs.

---

## Runtime Lifecycle

`SimulationConfig` is an immutable Pydantic model — it can be reused, serialized,
and shared freely.  Each call to `compile()` (or `run_simulation()`, which calls
`compile()` internally) instantiates a fresh set of agents and oracle objects.

- **`run_simulation(config)`** (recommended): compiles, runs, and returns an
  immutable `SimulationResult`.  Safe to call repeatedly on the same config.
- **`compile()` + `abides.run()`** (low-level): the runtime dict is **consumed
  once** — agents accumulate state during the run, so the dict must not be
  reused.  Call `compile()` again for a fresh dict.

```python
from abides_markets.simulation import run_simulation

# Recommended: config is reusable, each call compiles fresh agents
result1 = run_simulation(config)   # first run
result2 = run_simulation(config)   # second run — identical results
```

For the low-level path, call `compile()` once per run:

```python
from abides_markets.config_system import compile
from abides_core import abides

runtime = compile(config)          # fresh runtime dict
end_state = abides.run(runtime)    # consumes the dict
# Do NOT reuse `runtime` — call compile() again for another run.
```
