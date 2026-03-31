# ABIDES-Hasufel — Export Documentation

These documents are designed to be **copied into consumer projects** that use
`abides-hasufel` as a dependency. The primary audience is **AI coding agents**
that need to configure simulations, build custom agents, and analyze results.

## Reading Order

1. **ABIDES_REFERENCE.md** — Start here. Condensed quick-reference covering
   the mental model, safe data-access patterns, order placement, and config
   system basics. Enough to run a simulation and write a basic agent.

2. **ABIDES_LLM_INTEGRATION_GOTCHAS.md** — Every `None`/`NaN`/`KeyError`
   trap with safe-access patterns. Read before writing any agent code.

3. **ABIDES_CUSTOM_AGENT_IMPLEMENTATION_GUIDE.md** — Full adapter pattern,
   registration, config model, copy-paste scaffold, testing guide. Read when
   building a custom trading agent.

4. **ABIDES_CONFIG_SYSTEM.md** — Builder API, templates, oracle configuration,
   serialization, AI discoverability API. Read when configuring simulations
   programmatically or via YAML/JSON.

5. **ABIDES_DATA_EXTRACTION.md** — `SimulationResult`, `parse_logs_df`,
   L1/L2/L3 order book history. Read when analyzing simulation output.

6. **PARALLEL_SIMULATION_GUIDE.md** — `run_batch()`, multiprocessing,
   RNG hierarchy, log directory management. Read when running batch
   experiments or parameter sweeps.

## Maintenance

These files are **copies** of the originals in `docs/`. When updating
documentation, edit the originals in `docs/` first, then re-copy here.
