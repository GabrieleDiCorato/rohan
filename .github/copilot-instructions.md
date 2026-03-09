# Copilot Instructions for ROHAN

## Project Overview

ROHAN (Risk Optimization with Heuristic Agent Network) is a Python framework for autonomously evolving and stress-testing trading strategies using LLMs and high-fidelity market simulation.

The core loop:
1. A **Writer Agent** generates Python trading strategy code from a natural-language goal.
2. The strategy is validated via an **AST sandbox** (no unsafe imports or constructs).
3. The validated strategy is run inside the **ABIDES** market simulator across multiple scenarios.
4. An **Explainer Agent** (ReAct, tool-equipped) analyzes order book, PnL, and trade data.
5. A **Judge Agent** scores the iteration on 6 axes and decides whether to loop back or converge.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12+ |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| LLM orchestration | LangGraph + LangChain (OpenAI / Google GenAI) |
| Market simulation | `abides-rohan` (custom fork of ABIDES-JPMC) |
| Database | SQLAlchemy ORM — SQLite (dev) / PostgreSQL (prod) |
| UI | Streamlit |
| Linting / formatting | Ruff |
| Type checking | Pyright (basic mode) |
| Testing | pytest + hypothesis (property-based) + pytest-xdist (parallel) |

## Repository Layout

```
src/rohan/
  config/           # Pydantic-settings environment config
  framework/        # Core refinement loop: pipeline, repository, analysis, DB models
  llm/              # LangGraph graph, nodes, agents, tools, scoring, CLI
  simulation/       # ABIDES adapter, strategy validator, simulation runner/service
  ui/               # Streamlit pages (Terminal, Refinement Lab)
  utils/            # Shared utilities
tests/              # pytest test suite (mirrors src layout by topic)
notebooks/          # Exploratory Jupyter notebooks (outputs stripped from git)
docs/               # Additional documentation
```

## Development Workflow

### Setup
```bash
uv sync --all-groups          # Install all dependency groups
uv run pre-commit install     # Install git hooks (mandatory)
uv run nbstripout --install   # Strip notebook outputs on commit
```

### Common Commands
```bash
uv run pytest                        # Run tests (excludes slow/integration by default)
uv run pytest -m slow                # Run slow tests
uv run pytest -m integration        # Run integration tests
uv run ruff check .                  # Lint
uv run ruff format .                 # Format
uv run pyright                       # Type check
uv run pre-commit run --all-files    # Run all hooks
uv run ui                            # Launch Streamlit UI
uv run refine --goal "..." --max-iterations 3  # Run LLM refinement CLI
```

### Environment Variables
Copy `.env.template` to `.env` and fill in your LLM provider API keys (e.g. `OPENROUTER_API_KEY`, `OPENAI_API_KEY`). No API key is required to run market simulations alone.

## Coding Conventions

- **Formatting & linting**: Ruff enforces PEP 8, isort, pyupgrade, bugbear, and more. Line length is set to 200 characters (intentionally wide to accommodate long LangGraph/Pydantic expressions). Run `uv run ruff check --fix .` before committing.
- **Type annotations**: All public functions and methods must have full type annotations. Pyright runs in `basic` mode; avoid `Any` unless truly necessary.
- **Pydantic models**: Use Pydantic v2 for all data models (`model_config`, `model_validator`, field validators). Prefer `BaseModel` for data transfer and `BaseSettings` (via `pydantic-settings`) for configuration.
- **Path handling**: Use `pathlib.Path` (PTH rules enforced by Ruff) — never raw string paths.
- **Imports**: Absolute imports only; `rohan.*` is the first-party namespace. Group order: stdlib → third-party → `rohan.*`.
- **Tests**: Place tests in `tests/test_<topic>.py`. Use `pytest` fixtures (`conftest.py`). Mark slow tests with `@pytest.mark.slow` and integration tests with `@pytest.mark.integration`. Property-based tests use `hypothesis`.
- **Notebooks**: Never commit notebook outputs or metadata. `nbstripout` handles this automatically via the pre-commit hook.
- **UI modules**: Streamlit page files live under `src/rohan/ui/pages/` and follow Streamlit's multipage naming convention (N999 is ignored by Ruff for these files).

## Key Domain Concepts

- **Scenario**: A parameterised market simulation run (duration, agents, volatility, etc.).
- **Iteration**: One full loop pass — code generation → simulation → analysis → scoring.
- **RichAnalysisBundle**: A serialisable JSON snapshot of per-fill data, PnL trajectory, inventory, L2 order book, and counterparty breakdown captured after each simulation.
- **Scoring**: Six deterministic axes — Profitability, Risk, Volatility Impact, Spread Impact, Liquidity Impact, Execution Quality. Computed formulaically (no LLM noise), normalised to scenario config.
- **Strategy sandbox**: AST-based validation in `simulation/strategy_validator.py` that rejects unsafe imports and constructs before execution.
- **Seed consistency**: Per-scenario seeds are derived via SHA-256 to ensure identical random state across iterations for fair comparisons.

## CI / Quality Gates

The CI workflow (`.github/workflows/ci.yml`) runs on every push and PR to `main`:
1. `ruff check` + `ruff format --check`
2. `pyright`
3. `pytest` with branch coverage uploaded to Codecov

All three gates must pass before merging.
