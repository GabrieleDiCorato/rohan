# ABIDES — Parallel Simulation Guide

> **Audience:** LLM agents or developers who need to launch multiple ABIDES simulations concurrently.
> **Last updated:** 2026-02-21
> **Codebase version:** branch `rohan` of `jpmorganchase/abides-jpmc-public`

---

## 1. Architecture Overview

ABIDES is a **discrete-event simulation** framework. Each simulation is driven by a single `Kernel` instance that processes a `queue.PriorityQueue` of timestamped messages, dispatching `wakeup()` and `receive_message()` calls to `Agent` objects sequentially.

**Key facts:**
- One `Kernel` = one simulation. The Kernel is single-threaded internally.
- Each `Kernel`, each `Agent`, the `LatencyModel`, and each Oracle hold their own `np.random.RandomState` — all derived deterministically from a master seed in the config builder.
- There is **no built-in parallelism** within a simulation. All existing multi-run infrastructure (`version_testing/`) uses **process-based** parallelism (`p_tqdm.p_map`).

### Known Thread-Safety Issues (as of this writing)

The codebase has several **class-level and global mutable variables** that make **in-process threading unsafe**. These are:

| # | Issue | Location | Severity |
|---|---|---|---|
| 1 | `Message.__message_id_counter` — shared `ClassVar[int]`, non-atomic increment | `abides-core/abides_core/message.py:23-28` | **CRITICAL** |
| 2 | `Order._order_id_counter` — shared class `int`, non-atomic increment | `abides-markets/abides_markets/orders.py:31,72-73` | **CRITICAL** |
| 3 | `MeanRevertingOracle` uses `np.random.normal()` (global PRNG) | `abides-markets/abides_markets/oracles/mean_reverting_oracle.py:91` | **CRITICAL** |
| 4 | `coloredlogs.install()` reconfigures global root logger per run | `abides-core/abides_core/abides.py:33-36` | **MEDIUM** |
| 5 | Default `log_dir` is wall-clock seconds — can collide | `abides-core/abides_core/kernel.py:131-133` | **MEDIUM** |
| 6 | No simulation ID in log format — concurrent logs are interleaved | All modules | **LOW** |
| 7 | `Kernel.__init__` fallback path uses global `np.random.randint()` | `abides-core/abides_core/kernel.py:66-71` | **LOW** |

**Because of issues 1–3, running simulations with `threading` or `concurrent.futures.ThreadPoolExecutor` WILL produce bugs.** Use `multiprocessing` instead.

---

## 2. Safe Parallel Usage with `multiprocessing` (Recommended)

Using separate processes avoids all shared-state issues because each process gets its own copy of class variables, global PRNG, and logging configuration.

### 2.1 Minimal Example

```python
import multiprocessing as mp
from abides_markets.configs import rmsc04

from abides_core.abides import run


def run_one_simulation(args: dict) -> dict:
    """
    Entry point for a single simulation in a worker process.
    Each process gets its own memory space — no shared state concerns.
    """
    seed = args["seed"]
    log_dir = args["log_dir"]

    # Build a config with a unique seed
    config = rmsc04.build_config(
        seed=seed,
        # Override any parameters as needed:
        # end_time="10:00:00",
    )

    # Run the simulation
    end_state = run(
        config=config,
        log_dir=log_dir,           # MUST be unique per simulation
        kernel_seed=seed,
    )

    # Extract any results you need (end_state["agents"], etc.)
    # NOTE: return values must be picklable for multiprocessing
    return {
        "seed": seed,
        "elapsed": str(end_state.get("kernel_event_queue_elapsed_wallclock", "")),
    }


def main():
    num_simulations = 8
    seeds = list(range(1, num_simulations + 1))

    # Prepare arguments with unique log_dir per simulation
    sim_args = [
        {"seed": s, "log_dir": f"parallel_run/sim_{s}"}
        for s in seeds
    ]

    # Launch in parallel using a process pool
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(run_one_simulation, sim_args)

    for r in results:
        print(f"Seed {r['seed']}: elapsed {r['elapsed']}")


if __name__ == "__main__":
    main()
```

### 2.2 Using `concurrent.futures.ProcessPoolExecutor`

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

def main():
    num_simulations = 8
    seeds = list(range(1, num_simulations + 1))

    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = {
            executor.submit(
                run_one_simulation,
                {"seed": s, "log_dir": f"parallel_run/sim_{s}"}
            ): s
            for s in seeds
        }

        for future in as_completed(futures):
            seed = futures[future]
            result = future.result()
            print(f"Seed {seed}: done")
```

### 2.3 Using `p_tqdm` (already used in codebase for testing)

```python
from p_tqdm import p_map

results = p_map(
    run_one_simulation,
    [{"seed": s, "log_dir": f"parallel_run/sim_{s}"} for s in range(1, 9)],
    num_cpus=8,
)
```

---

## 3. Critical Rules for Safe Parallel Runs

### 3.1 Always pass a unique `log_dir`

The Kernel defaults `log_dir` to `str(int(datetime.now().timestamp()))`. Two simulations started within the same second will **overwrite each other's log files**. Always provide an explicit, unique `log_dir`:

```python
run(config=config, log_dir=f"experiment_42/seed_{seed}")
```

### 3.2 Always pass an explicit seed or `random_state`

The config builders (`rmsc03.build_config()`, `rmsc04.build_config()`) accept a `seed` parameter and derive all component-level `RandomState` objects from it deterministically. Always provide it:

```python
config = rmsc04.build_config(seed=42)
end_state = run(config=config, kernel_seed=42)
```

If you omit the seed/random_state, fallback paths hit the **global numpy PRNG** (`np.random.randint()`), which is not reproducible and not safe even across sequential runs.

### 3.3 Use `SparseMeanRevertingOracle`, not `MeanRevertingOracle`

`MeanRevertingOracle.generate_fundamental_value_series()` calls `np.random.normal()` on the **global PRNG**. Even with `multiprocessing`, this means the series is not reproducible unless you call `np.random.seed()` in the worker — which is fragile.

`SparseMeanRevertingOracle` (used by `rmsc03` and `rmsc04`) correctly uses per-symbol `RandomState` objects and is fully deterministic given the seed. **Prefer it.**

### 3.4 Configure logging once, at the top level

`abides_core.abides.run()` calls `coloredlogs.install()` internally, which reconfigures the root logger. In multiprocessing this is harmless (each process has its own logger state), but be aware that log output from multiple processes will be interleaved on stdout.

For clean per-simulation log files:

```python
import logging

def run_one_simulation(args):
    seed = args["seed"]
    log_dir = args["log_dir"]

    # Set up a per-process file handler
    handler = logging.FileHandler(f"./log/{log_dir}/simulation.log")
    handler.setFormatter(logging.Formatter(
        f"[sim_{seed}] %(levelname)s %(name)s %(message)s"
    ))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)

    config = rmsc04.build_config(seed=seed)
    return run(config=config, log_dir=log_dir, kernel_seed=seed)
```

### 3.5 Return values must be picklable

`multiprocessing` serializes return values via `pickle`. The full `end_state` dictionary contains `Agent` objects, pandas DataFrames, and numpy arrays — these are generally picklable, but large. Extract only what you need:

```python
def run_one_simulation(args):
    end_state = run(config=config, log_dir=log_dir, kernel_seed=seed)

    # Extract only needed results
    return {
        "seed": seed,
        "elapsed": str(end_state["kernel_event_queue_elapsed_wallclock"]),
        # Agent states, if needed:
        # "agent_state": end_state.get("agent_state", {}),
    }
```

### 3.6 Gym environments

Each `AbidesGymCoreEnv.reset()` creates a fresh `Kernel` and full agent set. Gym environments are designed for sequential use (one env = one simulation at a time). To run multiple gym environments in parallel, use multiple processes — each with its own env instance. **Do not share gym env instances across threads.**

```python
# Each worker creates its own env — safe
def run_gym_episode(seed):
    env = SubGymEnv(...)  # your specific gym env subclass
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        action = your_policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()
    return info
```

---

## 4. RNG Hierarchy (How Seeds Flow)

Understanding the RNG design helps ensure reproducibility:

```
build_config(seed=42)
│
├── master_rng = np.random.RandomState(seed=42)     ← local, not global
│
├── oracle_seed = master_rng.randint(...)
│   └── SparseMeanRevertingOracle(random_state=np.random.RandomState(oracle_seed))
│       └── per-symbol RandomState derived from oracle's random_state
│
├── agent_seeds = [master_rng.randint(...) for each agent]
│   └── Agent(random_state=np.random.RandomState(agent_seed))
│
├── kernel_seed = master_rng.randint(...)
│   └── Kernel(random_state=np.random.RandomState(kernel_seed))
│
└── latency_seed = master_rng.randint(...)
    └── LatencyModel(random_state=np.random.RandomState(latency_seed))
```

**Every component gets its own `RandomState`, derived deterministically from the master seed.** Given the same seed, the same simulation produces identical results — provided no global PRNG is used.

---

## 5. File Layout for Logs

When `skip_log=False`, each simulation writes:

```
./log/{log_dir}/
├── summary_log.bz2                    # Kernel summary (agent types, final values)
├── ExchangeAgent0.bz2                 # Per-agent event logs (pandas DataFrames)
├── NoiseAgent1.bz2
├── ValueAgent2.bz2
├── ...
```

For parallel runs, ensure each simulation has a unique `log_dir` to avoid file collisions.

---

## 6. Common Pitfalls

| Pitfall | Consequence | Prevention |
|---|---|---|
| Not passing `log_dir` | Multiple sims overwrite same directory | Always pass unique `log_dir` |
| Not passing `seed` | Hits global PRNG, non-reproducible | Always pass explicit seed |
| Using `MeanRevertingOracle` | Global PRNG usage, non-deterministic | Use `SparseMeanRevertingOracle` |
| Sharing gym env across threads | Undefined behavior | One env per process |
| Returning full `end_state` from workers | Large pickle overhead | Extract only needed fields |
| Using `ThreadPoolExecutor` | Race conditions on `Message`/`Order` counters | Use `ProcessPoolExecutor` |

---

---

# Appendix: Changes Required to Fix All Concurrency Issues

> These changes would make ABIDES safe for **in-process threading** (`ThreadPoolExecutor`).
> They are **NOT required** if you use `multiprocessing` (Option A above).

## Fix 1: `Message.__message_id_counter` — Use `itertools.count()`

**File:** `abides-core/abides_core/message.py`
**Effort:** ~10 minutes
**Risk:** Very low — `itertools.count()` is a drop-in replacement, thread-safe in CPython.

**Current:**
```python
from dataclasses import dataclass, field
from typing import ClassVar, List

@dataclass
class Message:
    __message_id_counter: ClassVar[int] = 1
    message_id: int = field(init=False)

    def __post_init__(self):
        self.message_id: int = Message.__message_id_counter
        Message.__message_id_counter += 1
```

**Proposed:**
```python
import itertools
from dataclasses import dataclass, field
from typing import ClassVar, List

@dataclass
class Message:
    _message_id_counter: ClassVar[itertools.count] = itertools.count(1)
    message_id: int = field(init=False)

    def __post_init__(self):
        self.message_id: int = next(Message._message_id_counter)
```

**Testing:** Existing tests pass unchanged. Add a test that creates Messages from two threads and verifies uniqueness.

---

## Fix 2: `Order._order_id_counter` — Use `itertools.count()`

**File:** `abides-markets/abides_markets/orders.py`
**Effort:** ~10 minutes
**Risk:** Low — same pattern as Fix 1. The only concern is tests that manually reset the counter (`Order._order_id_counter = 0`); those would need to be updated to `Order._order_id_counter = itertools.count(0)`.

**Current:**
```python
class Order:
    _order_id_counter: int = 0

    def __init__(self, ..., order_id=None):
        if order_id is None:
            order_id = Order._order_id_counter
            Order._order_id_counter += 1
```

**Proposed:**
```python
import itertools

class Order:
    _order_id_counter = itertools.count(0)

    def __init__(self, ..., order_id=None):
        if order_id is None:
            order_id = next(Order._order_id_counter)
```

**Also update:**
- `abides-markets/tests/__init__.py:5` — `Order._order_id_counter = itertools.count(0)`
- `abides-markets/tests/test_orders.py:11,22,33,42` — same reset pattern

---

## Fix 3: `MeanRevertingOracle` — Replace global PRNG with per-instance `RandomState`

**File:** `abides-markets/abides_markets/oracles/mean_reverting_oracle.py`
**Effort:** ~30 minutes (includes updating all callers to pass `random_state`)
**Risk:** Medium — changes the constructor signature. Any config that builds a `MeanRevertingOracle` must now pass `random_state`. The `SparseMeanRevertingOracle` (used by rmsc03/rmsc04) already does this correctly and is not affected.

**Current:**
```python
class MeanRevertingOracle(Oracle):
    def __init__(self, mkt_open, mkt_close, symbols):
        ...
        self.r[symbol] = self.generate_fundamental_value_series(symbol=symbol, **s)

    def generate_fundamental_value_series(self, symbol, r_bar, kappa, sigma_s):
        shock = np.random.normal(scale=sigma_s, size=(r.shape[0]))  # GLOBAL PRNG
```

**Proposed:**
```python
class MeanRevertingOracle(Oracle):
    def __init__(self, mkt_open, mkt_close, symbols, random_state: np.random.RandomState):
        self.random_state = random_state
        ...
        self.r[symbol] = self.generate_fundamental_value_series(symbol=symbol, **s)

    def generate_fundamental_value_series(self, symbol, r_bar, kappa, sigma_s):
        shock = self.random_state.normal(scale=sigma_s, size=(r.shape[0]))
```

**Also update:**
- Any config file or test that instantiates `MeanRevertingOracle(...)` must add `random_state=np.random.RandomState(seed)`.
- `abides-markets/tests/test_mean_reverting_oracle.py` — remove `np.random.seed(seed)`, pass explicit `RandomState` instead.

---

## Fix 4: Move `coloredlogs.install()` out of `run()`

**File:** `abides-core/abides_core/abides.py`
**Effort:** ~10 minutes
**Risk:** Low — behavioral change is that callers must configure logging themselves. Add a convenience function.

**Current:**
```python
def run(config, log_dir="", kernel_seed=0, kernel_random_state=None):
    coloredlogs.install(
        level=config["stdout_log_level"],
        fmt="[%(process)d] %(levelname)s %(name)s %(message)s",
    )
    kernel = Kernel(...)
    ...
```

**Proposed:**
```python
def setup_logging(level="INFO"):
    """Call once at program startup to configure logging."""
    coloredlogs.install(
        level=level,
        fmt="[%(process)d] %(levelname)s %(name)s %(message)s",
    )

def run(config, log_dir="", kernel_seed=0, kernel_random_state=None):
    # No longer reconfigures global logging
    kernel = Kernel(...)
    ...
```

---

## Fix 5: Make `log_dir` collision-proof by default

**File:** `abides-core/abides_core/kernel.py`
**Effort:** ~5 minutes
**Risk:** Very low — only affects the default; explicit `log_dir` overrides it.

**Current:**
```python
self.log_dir = log_dir or str(int(self.kernel_wall_clock_start.timestamp()))
```

**Proposed:**
```python
import uuid
self.log_dir = log_dir or f"{int(self.kernel_wall_clock_start.timestamp())}_{uuid.uuid4().hex[:8]}"
```

---

## Fix 6 (Optional): Add simulation ID to log format

**Files:** `abides-core/abides_core/kernel.py`, `abides-core/abides_core/abides.py`
**Effort:** ~1 hour
**Risk:** Low — additive change.

Use `logging.LoggerAdapter` in the Kernel to prefix all log messages with a simulation ID:

```python
class Kernel:
    def __init__(self, ..., sim_id=None):
        self.sim_id = sim_id or uuid.uuid4().hex[:8]
        self.logger = logging.LoggerAdapter(
            logging.getLogger(__name__),
            {"sim_id": self.sim_id}
        )
        # Use self.logger instead of module-level logger throughout
```

This requires replacing `logger.debug(...)` with `self.logger.debug(...)` throughout `kernel.py` (~40 call sites).

---

## Effort Summary

| Fix | Files Changed | Effort | Required for Threading? |
|---|---|---|---|
| 1. `Message` counter | 1 file | 10 min | **Yes** |
| 2. `Order` counter | 1 file + 2 test files | 10 min | **Yes** |
| 3. `MeanRevertingOracle` PRNG | 1 file + callers/tests | 30 min | **Yes** (if using this oracle) |
| 4. Move `coloredlogs.install()` | 1 file | 10 min | **Yes** |
| 5. Collision-proof `log_dir` | 1 file | 5 min | Recommended |
| 6. Simulation ID in logs | 2+ files | 1 hour | Optional |
| **Total** | **~5-8 files** | **~2 hours** | |

All fixes are backward-compatible. Fixes 1–4 are required for thread safety. Fix 5 is strongly recommended. Fix 6 is a quality-of-life improvement for debugging parallel runs.

**If using `multiprocessing` (Option A), none of these fixes are required** — but Fix 3 and Fix 5 are still recommended for correctness and reproducibility.
