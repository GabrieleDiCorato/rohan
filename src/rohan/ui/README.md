# ABIDES Simulation Terminal

A Streamlit-based web application for configuring, running, and analyzing ABIDES market simulations with an intuitive interface and comprehensive visualization tools.

## Features

### 📋 Configuration Panel

**Compact Key-Value Layout**
- All configuration parameters displayed in a space-efficient format
- Expandable sections for different agent types and settings
- Real-time input validation

**Preset Management**
- Load preset configurations for common market scenarios:
  - Default (Balanced Market)
  - High Volatility
  - Low Liquidity
  - Market Maker Stress Test
  - Momentum Dominated
- Customize any preset to fit your needs
- "Load Preset" updates sidebar inputs
- "Apply Configuration" transfers settings to Execute tab

**Comprehensive Parameter Control**
- **Simulation Settings**: Date, time range, random seed, logging options
- **Exchange Agent**: Book logging, order logging, pipeline delays
- **Market Participants**:
  - Noise Agents (random traders)
  - Value Agents (fundamental traders)
  - Adaptive Market Makers (liquidity providers)
  - Momentum Agents (trend followers)
- **Oracle Settings**: Fundamental value evolution, volatility, megashocks
- **Latency Models**: No latency, deterministic, or cubic latency with jitter

### ▶️ Execute Tab

**Applied Configuration Display**
- View the complete configuration that will be used for simulation runs
- Quick summary cards showing key parameters (date, time range, ticker, seed)
- Agent count overview with total agent calculation
- Expandable detailed configuration view organized by category

**Simulation Execution**
- Real-time progress tracking with status updates
- Clear error messages with troubleshooting tips
- Quick metrics displayed immediately after completion
- Execution history tracking

### 📊 Analyze Tab

**Metrics Dashboard**
- Volatility, Total PnL, Sharpe Ratio, Max Drawdown
- Win Rate, Average Win/Loss
- Custom metrics from simulation results

**Interactive Price Charts**
- Bid/Ask/Mid price evolution over time
- Returns distribution and time series
- Zoom, pan, and hover for detailed inspection
- Export charts as images

**Volume Analysis**
- Bid and ask volume tracking
- Volume imbalance visualization
- Statistical summaries (mean, std, percentiles)

**Spread Analysis**
- Bid-ask spread over time
- Spread distribution histogram
- Spread in basis points
- Statistical metrics

**Execution Logs**
- Searchable and filterable log viewer
- Filter by agent ID and event type
- Configurable row display limit
- Export logs to CSV
- Log statistics (total events, unique agents)

## Installation

### Prerequisites

- **Python**: 3.12 or higher
- **UV**: Modern Python package manager ([install UV](https://github.com/astral-sh/uv))

### Setup

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <repository-url>
   cd rohan
   ```

2. **Install dependencies**:
   ```bash
   # Install all project dependencies including UI
   uv sync
   ```

   The UI dependencies (Streamlit and Plotly) are included as an optional dependency group and will be installed automatically.

### Launch the Application

```bash
# Using the launch script (recommended)
uv run ui

# Or using Python module directly
uv run python -m rohan.ui
```

The application will automatically open in your default web browser at `http://localhost:8501`.

## Usage Workflow

### 1. Configure Your Simulation

1. **Select a preset** from the dropdown menu or start with "Custom"
2. **Click "Load Preset"** to populate the sidebar with preset values
3. **Customize parameters** as needed in the expandable sections
4. **Click "Apply Configuration"** to prepare the configuration for execution

### 2. Execute the Simulation

1. Navigate to the **"Execute" tab**
2. Review the **Applied Configuration** summary
3. Optionally expand **"View Full Configuration Details"** for complete settings
4. Click **"Run Simulation"** and monitor the progress
5. View **Quick Metrics** upon completion

### 3. Analyze Results

Navigate to the **"Analyze" tab** and explore:

- **Metrics**: Overall performance statistics
- **Price Charts**: Price evolution and returns
- **Volume**: Trading volume and imbalances
- **Spread**: Bid-ask spread analysis
- **Logs**: Detailed execution logs with filtering

All visualizations are interactive (zoom, pan, hover) and can be exported.

## Design Philosophy

The application features a modern, professional interface optimized for financial analysis:

- **Dark Theme**: Reduces eye strain during extended analysis sessions
- **Color Coding**: Cyan and amber accents for visual hierarchy and status indication
- **Compact Layout**: Maximizes information density without clutter
- **Responsive Design**: Optimized for wide screens and detailed data exploration
- **Performance**: Cached data processing for smooth interactions

## Tips & Best Practices

### Simulation Configuration

- **Minimum Duration**: Simulations should run for at least 5 minutes to avoid edge cases
- **Agent Counts**: Higher counts increase realism but also computation time
- **Start with Presets**: Use preset configurations as templates and customize from there
- **Seed Control**: Use consistent seeds for reproducible results

### Workflow Efficiency

- **Load → Modify → Apply**: Always click "Apply Configuration" after loading a preset or making changes
- **Check Execute Tab**: Verify the applied configuration before running simulations
- **Export Data**: Use CSV export features for external analysis in Excel, Python, R, etc.
- **Save Successful Configs**: Document configurations that produce interesting results

### Performance

- **Caching**: The app caches computed data for faster tab switching
- **Large Simulations**: For very long simulations or high agent counts, expect longer processing times
- **Browser Tab**: Keep the browser tab active during simulation runs for best performance

## Troubleshooting

### Common Issues

**"No configuration applied" in Execute tab**
- Make sure you clicked "Apply Configuration" in the sidebar
- Try reloading the page if the issue persists

**Simulation fails to run**
- Check that all agent counts are non-negative
- Ensure time range is at least 5 minutes
- Verify wake-up frequencies are in correct format (e.g., '60s')
- Check that numeric values are within reasonable ranges

**Analyze tab appears empty**
- Wait for the simulation to complete fully
- Check the Execute tab for any error messages
- Try switching to another tab and back

**Charts not displaying**
- Ensure the simulation completed successfully
- Check browser console for JavaScript errors
- Try refreshing the page

## Technical Details

### Architecture

- **Frontend**: Streamlit (reactive web framework)
- **Visualization**: Plotly (interactive charts)
- **Data Processing**: Pandas (cached for performance)
- **Simulation Engine**: ABIDES core library

### File Structure

```
src/rohan/ui/
├── __init__.py           # Package initialization
├── __main__.py           # Entry point for `python -m rohan.ui`
├── app.py                # Main Streamlit application
├── utils/
│   ├── presets.py        # Preset configurations
│   └── theme.py          # UI theme and styling
└── README.md             # This file
```

### Performance Optimizations

- `@st.cache_data` decorators for expensive computations
- Fragment-based rendering for sidebar (isolated updates)
- Lazy loading of analysis visualizations
- Efficient DataFrame operations

## Contributing

When modifying the UI:

1. **Test thoroughly**: Verify all tabs and features work after changes
2. **Maintain consistency**: Follow existing code style and patterns
3. **Update README**: Document new features or changed workflows
4. **Consider performance**: Use caching for expensive operations

## License

See the main project LICENSE file.
