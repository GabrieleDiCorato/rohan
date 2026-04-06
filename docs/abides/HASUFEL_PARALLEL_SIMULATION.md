# ABIDES — Parallel Simulation Guide

> **Snapshotted from `abides-hasufel` v2.5.8.** If you've upgraded hasufel, verify this doc is still current.

> **Audience:** LLM agents or developers who need to launch multiple ABIDES simulations concurrently.

---

## 1. Architecture Overview

ABIDES is a **discrete-event simulation** framework. Each simulation is driven by a single `Kernel` instance that processes a `heapq`-managed priority list of timestamped messages, dispatching `wakeup()` and `receive_message()` calls to `Agent` objects sequentially.

**Key facts:**
- One `Kernel` = one simulation. The Kernel is single-threaded internally.
- Each `Kernel`, each `Agent`, the `LatencyModel`, and each Oracle hold their own `np.random.RandomState` — all derived deterministically from a master seed via identity-based hashing (`SHA-256`). Each component's seed depends only on the master seed and the component's name — adding or removing agent groups never shifts other components' seeds.
- There is **no built-in parallelism** within a simulation. All existing multi-run infrastructure (`version_testing/`) uses **process-based** parallelism (`p_tqdm.p_map`).

### RNG Hierarchy (Identity-Based Derivation)

Each component derives its seed independently from the master seed via `hashlib.sha256(f"{seed}:{component}:{index}")`:

```
compile(config, seed=42)
├── oracle     → sha256("42:oracle:0")
├── exchange   → sha256("42:exchange:0")
├── agent group "noise"
│   ├── agent 0 → group_rng from sha256("42:agent:noise:0"), then sequential randint()
│   ├── agent 1 → ...
│   └── agent N → ...
├── agent group "value"
│   ├── agent 0 → group_rng from sha256("42:agent:value:0"), independent of noise
│   └── ...
├── kernel     → sha256("42:kernel:0")
└── latency    → sha256("42:latency:0")
```

**Composition invariance:** adding a new agent group (e.g. a custom strategy) does not change any existing agent's seed. This enables fair baseline-vs-strategy A/B comparison.

### Remaining Thread-Safety Concerns

The `Message` and `Order` ID generators use `itertools.count()`, which is GIL-safe in CPython. Both `MeanRevertingOracle` and `SparseMeanRevertingOracle` use injected `RandomState` objects (no global PRNG). However, in-process threading is still **not recommended** due to:

| # | Issue | Location | Severity |
|---|---|---|---|
| 1 | Default `log_dir` is wall-clock seconds — can collide | `kernel.py` | **MEDIUM** |
| 2 | No simulation ID in log format — concurrent logs are interleaved | All modules | **LOW** |
| 3 | `Kernel.__init__` fallback uses global `np.random.randint()` when no `random_state` is provided | `kernel.py` | **LOW** |

**Use `multiprocessing` for parallel runs.** Each process gets its own memory space, eliminating all shared-state concerns.

---

## 2. Using `run_batch()` (Recommended)

The simplest way to run simulations in parallel is `run_batch()` from
`abides_markets.simulation`.  It accepts a list of `SimulationConfig` objects,
spawns worker processes, compiles each config independently, and returns a list
of immutable `SimulationResult` objects in input order.

```python
from abides_markets.config_system import SimulationBuilder
from abides_markets.simulation import run_batch

configs = [
    SimulationBuilder().from_template("rmsc04").seed(s).build()
    for s in range(1, 9)
]

results = run_batch(configs)      # uses all available CPUs by default

for r in results:
    print(f"Seed {r.metadata.seed}: {r.markets['ABM'].l1_close}")
```

Each worker compiles its own runtime dict from the provided config — no shared
state, unique log directories (UUID-based by default), and fully deterministic
given the seed.  The returned `SimulationResult` objects are frozen Pydantic
models safe to share across threads.

For finer control over extraction, pass a `profile` or `extractors`:

```python
from abides_markets.simulation import run_batch, ResultProfile

results = run_batch(configs, profile=ResultProfile.QUANT, n_workers=4)
```

---

## 3. Manual Parallelism with `multiprocessing`

For cases where you need direct control over the worker processes (e.g.,
custom logging, non-standard return values), you can use `multiprocessing`
directly with the low-level `compile()` → `abides.run()` path.

### 3.1 Minimal Example

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

### 3.2 Using `concurrent.futures.ProcessPoolExecutor`

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

### 3.3 Using `p_tqdm` (already used in codebase for testing)

```python
from p_tqdm import p_map

results = p_map(
    run_one_simulation,
    [{"seed": s, "log_dir": f"parallel_run/sim_{s}"} for s in range(1, 9)],
    num_cpus=8,
)
```

---

## 4. Critical Rules for Safe Parallel Runs

### 4.1 Always pass a unique `log_dir`

The Kernel defaults `log_dir` to `str(int(datetime.now().timestamp()))`. Two simulations started within the same second will **overwrite each other's log files**. Always provide an explicit, unique `log_dir`:

```python
run(config=config, log_dir=f"experiment_42/seed_{seed}")
```

### 4.2 Always pass an explicit seed or `random_state`

The config builders (`rmsc03.build_config()`, `rmsc04.build_config()`) accept a `seed` parameter and derive all component-level `RandomState` objects from it deterministically. Always provide it:

```python
config = rmsc04.build_config(seed=42)
end_state = run(config=config, kernel_seed=42)
```

If you omit the seed/random_state, fallback paths hit the **global numpy PRNG** (`np.random.randint()`), which is not reproducible and not safe even across sequential runs.

### 4.3 Both oracles use injected `RandomState`

Both `MeanRevertingOracle` and `SparseMeanRevertingOracle` accept and use an injected `random_state` parameter — neither calls the global `np.random` PRNG. Results are fully deterministic given the seed.

`SparseMeanRevertingOracle` (used by `rmsc03` and `rmsc04`) additionally derives per-symbol `RandomState` objects. **Either oracle is safe for parallel runs.**

### 4.4 Configure logging at the top level

The `abides_core.abides.run()` function does **not** configure logging. If you want colored log output, call `coloredlogs.install()` yourself at program startup (the CLI script `abides-core/scripts/abides` does this). In multiprocessing, each process has its own logger state, so log output from multiple processes will be interleaved on stdout unless you add per-process file handlers.

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

### 4.5 Return values must be picklable

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

### 4.6 Gym environments

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

## 5. RNG Hierarchy (How Seeds Flow)

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

## 6. File Layout for Logs

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

## 7. Common Pitfalls

| Pitfall | Consequence | Prevention |
|---|---|---|
| Not passing `log_dir` | Multiple sims overwrite same directory | Always pass unique `log_dir` |
| Not passing `seed` | Hits global PRNG, non-reproducible | Always pass explicit seed |
| Sharing gym env across threads | Undefined behavior | One env per process |
| Returning full `end_state` from workers | Large pickle overhead | Extract only needed fields |
| Using `ThreadPoolExecutor` | Shared class-level counters across threads | Use `ProcessPoolExecutor` |
