# Performance Metrics & Evaluation Framework

This document outlines the quantitative metrics used to evaluate trading strategies within the Rohan simulation framework. The evaluation methodology focuses on three core dimensions: **profitability**, **risk**, and **market impact**.

## 1. Agent Performance (Profitability & Risk)

These metrics assess the standalone performance of the strategic agent.

### 1.1 Profit & Loss (PnL) Attribution

We decompose Total PnL into realized and unrealized components to understand the source of returns.

*   **Realized PnL**: The accumulated cash balance resulting from closed trades. In our simulation, this is tracked via the agent's cash holdings, assuming a starting cash balance of zero (or normalized to zero).
    *   *Formula:* $PnL_{realized} = Cash_{end} - Cash_{start}$
*   **Unrealized PnL**: The mark-to-market value of the remaining inventory at the end of the simulation.
    *   *Valuation Price:* Mid-Price at simulation end ($P_{mid} = \frac{P_{bid} + P_{ask}}{2}$).
    *   *Formula:* $PnL_{unrealized} = Inventory_{end} \times P_{mid, end}$
*   **Total PnL**: The sum of realized and unrealized components.
    *   *Formula:* $PnL_{total} = PnL_{realized} + PnL_{unrealized}$

### 1.2 Execution Quality

Metrics evaluating the agent's ability to access liquidity.

*   **Fill Rate**: The ratio of executed orders to submitted orders. A low fill rate may indicate passive limit orders are not crossing the spread or prices are moving away.
    *   *Formula:* $FillRate = \frac{N_{executed}}{N_{submitted}}$
*   **Trade Count**: Total number of executed trades. Indicates activity level and turnover.

### 1.3 Risk Metrics (Planned)

*   **Sharpe Ratio**: The risk-adjusted return, calculated as the mean of excess returns divided by the standard deviation of returns. (Currently a placeholder in MVP).
*   **Max Drawdown**: The maximum observe loss from a peak to a trough of the equity curve. (Currently a placeholder in MVP).

## 2. Market Impact Analysis

To assess the externality of the strategy, we compare market conditions between a **Strategy Run** (with the active agent) and a **Baseline Run** (agent disabled or replaced by noise).

### 2.1 Volatility Impact

Measures how much the strategy contributes to price instability.

*   **Volatility**: Annualized standard deviation of mid-price returns.
    *   *Normalization:* $\sigma_{annual} = \sigma_{sample} \times \sqrt{252 \times 390 \times 60}$ (assuming 1-second snapshots).
*   **Volatility Delta**: $\Delta \sigma = \sigma_{strategy} - \sigma_{baseline}$
    *   *Interpretation:* Positive delta implies the strategy increases market noise/instability.

### 2.2 Liquidity & Spread

*   **Spread Delta**: Change in the average bid-ask spread.
    *   *Formula:* $\Delta Spread = \mu(Spread_{strategy}) - \mu(Spread_{baseline})$
    *   *Interpretation:* Negative delta implies the strategy improves market quality (tighter spreads).
*   **Liquidity Consumption/Provision**: Change in order book depth at the best levels.
    *   *Formula:* $\Delta Liquidity_{bid} = \mu(Vol_{bid, strategy}) - \mu(Vol_{bid, baseline})$

## 3. Comparative Evaluation

The `ComparisonResult` object encapsulates these metrics to provide a holistic view:

| Metric Category | Key Indicators |
| :--- | :--- |
| **Alpha Generation** | Total PnL, Sharpe Ratio |
| **Execution** | Fill Rate, Inventory Management |
| **Market Quality** | Spread Delta, Volatility Delta |

This structured comparison allows specific feedback loops:
- *High PnL but High Volatility Delta* $\rightarrow$ Strategy is profitable but destabilizing.
- *Low Fill Rate & Low PnL* $\rightarrow$ Strategy is too passive.
