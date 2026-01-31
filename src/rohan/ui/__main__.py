"""Main entry point for launching the ABIDES Simulation Terminal UI.

This module provides the main() function that launches the Streamlit app.
It can be invoked via `uv run ui` command.
"""

import sys
from pathlib import Path


def main():
    """Launch the Streamlit UI application."""
    try:
        import streamlit.web.cli as stcli
    except ImportError:
        print("Error: Streamlit is not installed.")
        print("Please install the UI dependencies with: uv sync --group ui")
        sys.exit(1)

    # Get the path to the app.py file
    ui_dir = Path(__file__).parent
    app_path = ui_dir / "app.py"

    if not app_path.exists():
        print(f"Error: Could not find app.py at {app_path}")
        sys.exit(1)

    # Launch Streamlit with the app
    sys.argv = ["streamlit", "run", str(app_path)]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
