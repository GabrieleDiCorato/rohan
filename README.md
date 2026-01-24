# rohan
An agentic framework for autonomously evolving and stress-testing robust trading strategies using LLMs and market simulation.

## Development Setup

### Prerequisites
- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager

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

3. Install pre-commit hooks (mandatory for all developers):
```bash
uv run pre-commit install
uv run nbstripout --install
```

This ensures that:
- Jupyter notebook outputs and metadata are automatically stripped from commits (keeping your local files intact)
- Code is automatically formatted and linted before commits
- Type checking is performed on committed code

### Running Tests
```bash
uv run pytest
```

### Code Quality
The project uses pre-commit hooks to maintain code quality:
- **ruff**: Linting and formatting
- **mypy**: Type checking
- **nbstripout**: Strips Jupyter notebook outputs/metadata from commits

To run pre-commit hooks manually:
```bash
uv run pre-commit run --all-files
```
