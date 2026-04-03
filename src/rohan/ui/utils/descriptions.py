"""Centralised metric descriptions for the Rohan financial cockpit.

Each key matches the ``label`` argument passed to
:func:`~rohan.ui.utils.components.glassmorphism_card`.  Descriptions are written
for financial-industry practitioners and include formulas where applicable.
"""

from __future__ import annotations

METRIC_DESCRIPTIONS: dict[str, str] = {
    # ── Summary KPI cards ─────────────────────────────────────────────────
    "Mid Price": (
        "Closing mid-price: the arithmetic mean of the best bid and "
        "best ask at market close. Mid = (Bid + Ask) / 2. "
        "It is the standard reference price used to mark positions and "
        "compute unrealised P&L."
    ),
    "Bid-Ask Spread": (
        "Quoted spread at market close, defined as Ask − Bid. "
        "The spread is a direct measure of the round-trip transaction cost "
        "faced by a liquidity taker and reflects market-maker compensation "
        "for adverse-selection and inventory risk."
    ),
    "VWAP": (
        "Volume-Weighted Average Price over the full simulation window: "
        "VWAP = Σ(Pᵢ × Qᵢ) / Σ Qᵢ, where Pᵢ and Qᵢ are the price and "
        "quantity of each fill. VWAP is the primary execution benchmark for "
        "institutional order-execution algorithms (POV, TWAP, VWAP)."
    ),
    "Volume": (
        "Total number of shares exchanged during the simulation. "
        "Volume is a key indicator of market activity and liquidity depth; "
        "higher volume generally implies tighter spreads and lower market-impact cost."
    ),
    "Realized Vol (σ)": (
        "Sample standard deviation of log mid-price returns: "
        "σ = std(ln(Midₜ / Midₜ₋₁)). "
        "Realized volatility quantifies the magnitude of price fluctuations "
        "and is a core input to option pricing (Black-Scholes) and risk models (VaR)."
    ),
    "Price Range": ("Intraday price range: Max(Mid) − Min(Mid). Captures the total price excursion during the session and is a simple proxy for intraday risk exposure."),
    "Wall-Clock": ("Elapsed real (wall-clock) time to execute the simulation, in seconds. Useful for profiling and capacity planning."),
    # ── Spread statistics ─────────────────────────────────────────────────
    "Mean Spread": (
        "Time-averaged quoted spread in dollar terms across all L1 snapshots where both bid and ask were present. A tighter mean spread signals higher liquidity and lower implicit trading costs."
    ),
    "Median Spread": ("Median of the quoted spread distribution (in dollars). The median is more robust than the mean to outlier spread blow-outs caused by fleeting one-sided book conditions."),
    "Max Spread": (
        "Maximum observed quoted spread during the session. A large max spread may indicate episodes of liquidity withdrawal or stress, and is relevant for worst-case transaction-cost analysis."
    ),
    "Spread Std": ("Standard deviation of the quoted spread (dollars). High spread volatility suggests unstable liquidity provision and greater uncertainty in execution costs."),
    "Mean Spread %": ("Mean quoted spread expressed as a percentage of the mid-price: Spread% = (Ask − Bid) / Mid × 100. Normalising by price allows cross-asset comparisons."),
    "Median Spread %": ("Median quoted spread as a percentage of the mid-price. Provides a robust, price-normalised view of typical trading costs."),
    # ── Market quality ────────────────────────────────────────────────────
    "% Time No Bid": ("Percentage of L1 snapshots with no resting bid. A high value indicates frequent loss of buy-side liquidity, exposing sellers to execution risk and potential price gaps."),
    "% Time No Ask": ("Percentage of L1 snapshots with no resting ask. Analogous to '% Time No Bid' for the sell side; frequent absence signals poor market quality for buyers."),
    "% Time Two-Sided": (
        "Percentage of time the order book was two-sided (both bid and ask present). A two-sided book is a prerequisite for orderly price discovery and is often a regulatory market-quality benchmark."
    ),
    "Last Trade": ("Price of the last executed trade in the simulation. Used as the settlement or closing price reference when the mid-price is unavailable."),
    # ── Returns distribution ──────────────────────────────────────────────
    "Mean Return": ("Arithmetic mean of log mid-price returns: μ = mean(ln(Midₜ / Midₜ₋₁)). A non-zero mean indicates directional price drift over the session."),
    "Std Dev": ("Standard deviation of log returns, measuring per-tick price risk. Equivalent to the realised volatility at the observation frequency."),
    "Skewness": (
        "Third standardised moment of the return distribution. Negative skew implies a longer left tail (crash risk); positive skew implies a longer right tail. Gaussian returns have skew = 0."
    ),
    "Excess Kurtosis": (
        "Fourth standardised moment minus 3 (the Gaussian baseline). "
        "Positive excess kurtosis ('leptokurtosis') signals heavier tails "
        "than a normal distribution — i.e., more frequent extreme returns. "
        "This is a key input for tail-risk models and stress testing."
    ),
    # ── Advanced microstructure ───────────────────────────────────────────
    "Mean Spread (¢)": ("Time-weighted average quoted spread in cents, computed from the full L1 snapshot series. Lower values indicate tighter markets and reduced implicit transaction costs."),
    "Ann. Volatility": (
        "Annualised mid-price return volatility: σ_ann = σ_tick × √(N_ticks_per_year). "
        "Scaling factor assumes 252 trading days. Volatility is the primary "
        "risk measure in portfolio construction and option pricing."
    ),
    "Sharpe Ratio": (
        "Annualised Sharpe ratio of the equity curve: SR = (mean(R) / std(R)) × √252. Measures risk-adjusted return; a ratio above 1.0 is generally considered attractive by institutional investors."
    ),
    "Avg Bid Depth": ("Average resting quantity at the best bid across all L1 snapshots. Larger depth means the market can absorb larger sell orders without significant price impact."),
    "Avg Ask Depth": ("Average resting quantity at the best ask across all L1 snapshots. Larger depth means the market can absorb larger buy orders without significant price impact."),
    "LOB Imbalance μ": (
        "Mean order-book imbalance, defined as "
        "I = (Q_bid − Q_ask) / (Q_bid + Q_ask). "
        "Ranges from −1 (all depth on ask) to +1 (all depth on bid). "
        "Persistent imbalance predicts short-term price direction "
        "(Cont, Kukanov & Stoikov, 2014)."
    ),
    "LOB Imbalance σ": ("Standard deviation of order-book imbalance. Higher variability indicates more dynamic shifts in supply/demand balance and greater microstructural uncertainty."),
    "VPIN": (
        "Volume-Synchronized Probability of Informed Trading "
        "(Easley, López de Prado & O'Hara, 2012). "
        "Estimates the fraction of volume attributable to informed traders. "
        "Values close to 1 signal high information asymmetry and elevated "
        "adverse-selection risk for market makers."
    ),
    "Resilience (ms)": (
        "Mean time (in milliseconds) for the quoted spread to revert to its "
        "average after a shock (Foucault, Kadan & Kandel, 2013). "
        "Faster resilience indicates a healthier, more self-correcting order book."
    ),
    # ── Execution summary ─────────────────────────────────────────────────
    "Exec Agents": ("Number of execution-algorithm agents (e.g., POV, TWAP, VWAP) active in the simulation."),
    "Total Filled": ("Aggregate shares filled versus the aggregate target quantity across all execution agents: Filled / Target."),
    "Avg Fill Rate": ("Average fill rate across execution agents: mean(Filledᵢ / Targetᵢ × 100). Measures how effectively execution algorithms completed their orders."),
    "Avg VWAP Slippage": (
        "Average slippage of execution agents relative to the market VWAP, in basis points: Slippage = (AvgFillPrice − VWAP) / VWAP × 10,000. Positive values indicate worse-than-benchmark execution."
    ),
    "Max Drawdown": (
        "Maximum peak-to-trough decline in the equity curve of execution agents, "
        "Max DD = max(Peak NAVₜ − NAVₜ). "
        "A critical risk metric that quantifies the worst cumulative loss "
        "experienced during the trading period."
    ),
    # ── Order flow ────────────────────────────────────────────────────────
    "Orders Submitted": ("Total number of order-submission events recorded in the simulation. Includes all order types (limit, market, cancel-replace)."),
    "Executions": ("Count of ORDER_EXECUTED events. Each event represents a fill (full or partial) matched by the exchange engine."),
    "Cancellations": ("Count of ORDER_CANCELLED and PARTIAL_CANCELLED events. High cancellation rates relative to submissions may indicate aggressive quoting strategies or quote stuffing."),
    "Fill Rate": ("Order-level fill rate: Executions / Submissions × 100. Reflects market liquidity conditions and agent aggressiveness."),
    "Cancel Rate": (
        "Cancellation rate: Cancellations / Submissions × 100. "
        "Under MiFID II (RTS 9), the order-to-trade ratio (inverse of fill rate) "
        "is subject to regulatory monitoring for high-frequency participants."
    ),
    # ── Trade attribution ─────────────────────────────────────────────────
    "Total Trades": ("Total number of matched trades in the simulation. Each trade pairs a passive (maker) and an aggressive (taker) side."),
    "Maker Types": ("Number of distinct agent types acting as passive (maker) counterparties. A diverse maker pool suggests broad liquidity provision."),
    "Taker Types": ("Number of distinct agent types acting as aggressive (taker) counterparties. A diverse taker pool reflects varied demand sources."),
    # ── Refinement lab ────────────────────────────────────────────────────
    "Iteration": "Current iteration number in the LLM refinement loop.",
    "Best Score": ("Highest composite score achieved so far across all refinement iterations. Higher is better; scale depends on the scoring rubric used."),
    "Current Score": "Composite score for the most recently completed iteration.",
    "Improvement": "Score delta between the current iteration and the previous one.",
}


def get_description(label: str) -> str:
    """Return the description for *label*, or an empty string if unknown."""
    return METRIC_DESCRIPTIONS.get(label, "")
