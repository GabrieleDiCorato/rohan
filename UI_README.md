# ABIDES Simulation Terminal

A Bloomberg Terminal-inspired Streamlit application for configuring, running, and analyzing ABIDES market simulations.

## Features

### ⚙️ Configure
- **Comprehensive Parameter Settings**: Configure all simulation parameters including date, time range, seed, and logging options
- **Agent Configuration**: Set up Exchange, Noise, Value, Adaptive Market Maker, and Momentum agents with detailed parameters
- **Oracle Settings**: Control fundamental value evolution and megashock parameters
- **Latency Models**: Choose from no latency, deterministic, or cubic latency models
- **Preset Configurations**: Quick-start templates for different market scenarios:
  - Default (Balanced Market)
  - High Volatility
  - Low Liquidity
  - Market Maker Stress Test
  - Momentum Dominated
- **Import/Export**: Save and load configurations as .env files

### ▶️ Execute
- **Real-time Progress Tracking**: Monitor simulation execution with status updates
- **Configuration Summary**: Review settings before running
- **Error Handling**: Clear error messages with troubleshooting tips
- **Quick Metrics**: View key metrics immediately after completion

### 📊 Analyze
- **Metrics Dashboard**: View volatility, PnL, Sharpe ratio, win rate, and custom metrics
- **Interactive Price Charts**: Plotly-powered visualizations of bid/ask/mid prices and returns
- **Volume Analysis**: Track bid/ask volume and volume imbalance over time
- **Spread Analysis**: Analyze bid-ask spread with multiple visualizations including distribution and basis points
- **Execution Logs**: Searchable and filterable log viewer with agent-specific filtering
- **Order Book Visualization**: View L1 and L2 order book data with export capabilities

## Installation

### Prerequisites
- Python 3.12+
- UV package manager

### Install Dependencies

```bash
# Install core dependencies (includes pandas 3.x)
uv sync

# Install UI dependencies separately (Note: Streamlit doesn't support pandas 3 yet)
# The UI will work with pandas 2.x that gets installed alongside streamlit
uv pip install streamlit plotly
```

**Note on Pandas Compatibility:** The main project uses pandas 3.x, but Streamlit currently only supports pandas <3. When you install Streamlit, it will install pandas 2.x in the same environment. The UI code is compatible with both pandas versions, so this works fine for the Streamlit app while your main code continues to use pandas 3.x features.

## Usage

### Launch the UI

```bash
# Using the launch script
uv run ui

# Or using Python module directly
uv run python -m rohan.ui
```

The app will open in your default web browser at `http://localhost:8501`.

## Workflow

1. **Configure**: Navigate to the Configure page and set up your simulation parameters
   - Choose a preset configuration or customize your own
   - Adjust agent counts and parameters
   - Configure oracle and latency settings
   - Save configuration or export to .env file

2. **Execute**: Go to the Execute page and run your simulation
   - Review configuration summary
   - Click "Run Simulation" and monitor progress
   - View quick metrics upon completion

3. **Analyze**: Explore results on the Analyze page
   - View comprehensive metrics dashboard
   - Interact with price, volume, and spread charts
   - Filter and search execution logs
   - Export data for further analysis

## Design

The application features a Bloomberg Terminal-inspired design with:
- Dark background (#0A0E27) for reduced eye strain
- Cyan (#00D9FF) and amber (#FFB800) accents for visual hierarchy
- Monospace fonts for a professional, terminal-like aesthetic
- Interactive Plotly charts with consistent theming
- Responsive layout optimized for wide screens

## Tips

- **Simulation Duration**: Simulations must run for at least 5 minutes to avoid edge cases
- **Agent Counts**: Higher agent counts increase simulation time but provide more realistic market dynamics
- **Preset Configurations**: Use presets as starting points and customize as needed
- **Export Configurations**: Save successful configurations as .env files for reproducibility
- **Data Export**: All charts and logs can be exported to CSV for external analysis
