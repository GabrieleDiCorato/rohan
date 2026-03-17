# ROHAN: Risk Optimization with Heuristic Agent Network

[![CI](https://github.com/GabrieleDiCorato/rohan/actions/workflows/ci.yml/badge.svg)](https://github.com/GabrieleDiCorato/rohan/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/GabrieleDiCorato/rohan/branch/main/graph/badge.svg)](https://codecov.io/gh/GabrieleDiCorato/rohan)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

An agentic framework for autonomously evolving and stress-testing robust trading strategies using Large Language Models (LLMs) and high-fidelity market simulation.

## Functional Overview

ROHAN provides an autonomous loop for generating, testing, and refining specific trading strategies using the `abides-rohan` market simulator. It leverages **LangGraph** for state management, **PostgreSQL** for relational data storage, and **Streamlit** for real-time monitoring and analysis.

The core refinement loop operates as follows:

1. **Define:** The user inputs a natural language trading goal (e.g., "Market making with low inventory risk").
2. **Generate & Validate:** The LLM (Writer Agent) generates Python code implementing the strategy. The code is instantly validated for safety and syntax via an AST sandbox.
3. **Simulate:** The validated strategy is injected into the ABIDES high-fidelity market simulator and stress-tested against various market scenarios.
4. **Analyze:** An Explainer Agent uses data tools to analyze the order book, trade logs, and PnL to understand why the strategy performed the way it did.
5. **Refine:** A Judge Agent scores the iteration. If the strategy has not converged on the goal, the feedback is routed back to the Writer to improve the code in the next loop.

## Features

- **Deterministic 6-axis scoring** — Profitability, Risk, Volatility Impact, Spread Impact, Liquidity Impact, Execution Quality. Formulaic (no LLM noise), auto-normalized to scenario config.
- **Tool-equipped Explainer Agent** — ReAct agent with 8 investigation tools queries fills, PnL curves, inventory trajectories, adverse-selection windows, and L2 order book snapshots.
- **Rich analysis pipeline** — `RichAnalysisBundle` captures per-fill execution data, PnL trajectory, order lifecycle, counterparty breakdown, and sampled L2 snapshots. Serialized as JSON for checkpoint safety.
- **6 simulation charts** — Price, Spread, Volume (market), PnL Curve, Inventory Trajectory, Fill Scatter (strategy). All persisted to DB and displayed in a 2×3 grid in the UI.
- **Session persistence** — Full save/load round-trip of refinement sessions (iterations, scores, charts, analysis data) via SQLAlchemy ORM.
- **Strategy sandbox** — AST-based validation rejects unsafe imports/constructs. Execution runs in a timeout-bounded thread pool.
- **Seed consistency** — Deterministic per-scenario seeds (SHA-256) ensure identical random state across iterations for fair comparisons.

## Documentation

ROHAN organizes its documentation into three distinct categories to serve different audiences:

- **[Functional Docs (`docs/functional/`)](docs/functional/)**: User-facing documentation explaining the "why" and "how" of the system. Includes functional definitions of all quantitative market metrics.
- **[Technical Docs (`docs/technical/`)](docs/technical/)**: Developer-facing documentation containing the core system architecture, adversarial scenario generation design, and parallel simulation execution guides.
- **[LLM Knowledge Base (`docs/llm/`)](docs/llm/)**: Explicitly designed references, API contracts, and gotchas built for AI coding assistants integrating with the `abides-rohan` simulator.

## Getting Started

### Prerequisites
- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- PostgreSQL (optional, defaults to SQLite for local development)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/GabrieleDiCorato/rohan.git
cd rohan
```

2. Install dependencies:
```bash
uv sync --all-groups
```

3. Set up environment variables:
Create a `.env` file in the root directory and add your LLM provider keys:
```env
OPENROUTER_API_KEY=your_api_key_here
# Or OPENAI_API_KEY, ANTHROPIC_API_KEY, etc. depending on your configuration
```
Your API key is not required to run and explore market simulations in the Terminal page of the UI.

Optional writer reliability tuning for the Refinement Lab:
```env
LLM_WRITER_MAX_RETRIES=3
LLM_WRITER_RETRY_PROMPT_TRIM=true
# Optional: use a dedicated model for the final retry attempt
LLM_WRITER_FALLBACK_MODEL=

# Explainer resilience controls
LLM_EXPLAINER_REACT_RECURSION_LIMIT=25
LLM_EXPLAINER_MAX_TOOL_CALLS=12

# Baseline simulation caching (SimulationSettings via env)
SIM_BASELINE_CACHE_ENABLED=true
SIM_BASELINE_CACHE_MAX_ENTRIES=64
```

Refinement telemetry is emitted as structured JSON log lines under the `rohan.telemetry` logger.

Rollout feature flags (for staged enablement):
```env
FEATURE_LLM_EXPLAINER_TIERS_V1=true
FEATURE_EXPLICIT_TERMINAL_REASONS_V1=true
FEATURE_BASELINE_CACHE_V1=true
FEATURE_LLM_TELEMETRY_V1=true
```

### Usage

Launch the Streamlit UI (Terminal and Refinement Lab):
```bash
uv run ui
```

Run the LLM refinement loop via CLI:
```bash
uv run refine --goal "Create a momentum strategy" --max-iterations 3
```

## Development Setup

### Pre-commit Hooks
Install pre-commit hooks (mandatory for all developers):
```bash
uv run pre-commit install
uv run nbstripout --install
```

This ensures that:
- Jupyter notebook outputs and metadata are automatically stripped from commits (keeping your local files intact).
- Code is automatically formatted and linted before commits.
- Type checking is performed on committed code.

### Running Tests
Execute the test suite using pytest:
```bash
uv run pytest
```

### Code Quality
The project uses pre-commit hooks to maintain code quality:
- **ruff**: Linting and formatting
- **pyright**: Type checking
- **nbstripout**: Strips Jupyter notebook outputs/metadata from commits

To run pre-commit hooks manually:
```bash
uv run pre-commit run --all-files
```
