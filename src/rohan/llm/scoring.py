"""Multi-axis scoring for goal-conditional strategy evaluation.

Provides a deterministic, keyword-based system that classifies the user's
goal into a weighted scoring profile and generates a rubric for the judge.

Four axes are evaluated independently:
1. **Profitability** — absolute PnL relative to starting capital
2. **Risk-Adjusted Performance** — Sharpe ratio, drawdown, inventory risk
3. **Market Impact** — stabilising vs. destabilising effects
4. **Execution Quality** — fill rate, OTT ratio, order management

Weight profiles shift emphasis based on the goal (e.g. a "minimise VPIN"
goal up-weights Market Impact; a "maximise Sharpe" goal up-weights Risk).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringWeights:
    """Weights for each scoring axis.  Must sum to 1.0."""

    profitability: float
    risk_adjusted: float
    market_impact: float
    execution_quality: float

    def __post_init__(self) -> None:
        total = self.profitability + self.risk_adjusted + self.market_impact + self.execution_quality
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")


# ── Pre-defined weight profiles ──────────────────────────────────────────

WEIGHT_PROFILES: dict[str, ScoringWeights] = {
    "default": ScoringWeights(
        profitability=0.40,
        risk_adjusted=0.25,
        market_impact=0.15,
        execution_quality=0.20,
    ),
    "impact_focused": ScoringWeights(
        profitability=0.20,
        risk_adjusted=0.15,
        market_impact=0.45,
        execution_quality=0.20,
    ),
    "risk_focused": ScoringWeights(
        profitability=0.20,
        risk_adjusted=0.45,
        market_impact=0.15,
        execution_quality=0.20,
    ),
    "execution_focused": ScoringWeights(
        profitability=0.20,
        risk_adjusted=0.20,
        market_impact=0.20,
        execution_quality=0.40,
    ),
}

# ── Goal keyword sets ────────────────────────────────────────────────────

_IMPACT_KEYWORDS = frozenset(
    {
        "impact",
        "vpin",
        "spread",
        "stabiliz",
        "volatility",
        "destabiliz",
        "liquidity",
        "market quality",
    }
)
_RISK_KEYWORDS = frozenset(
    {
        "risk",
        "drawdown",
        "sharpe",
        "sortino",
        "var",
        "cvar",
        "hedge",
        "protect",
        "risk-adjust",
    }
)
_EXECUTION_KEYWORDS = frozenset(
    {
        "execution",
        "fill",
        "slippage",
        "latency",
        "order-to-trade",
        "ott",
        "fill rate",
    }
)


def classify_goal_weights(goal: str) -> ScoringWeights:
    """Classify a goal string into scoring weights via keyword matching.

    Returns the best-matching weight profile based on keyword hits in
    the goal text.  Falls back to the default (PnL-heavy) profile when
    no category dominates.
    """
    goal_lower = goal.lower()

    impact_hits = sum(1 for kw in _IMPACT_KEYWORDS if kw in goal_lower)
    risk_hits = sum(1 for kw in _RISK_KEYWORDS if kw in goal_lower)
    exec_hits = sum(1 for kw in _EXECUTION_KEYWORDS if kw in goal_lower)

    if impact_hits > risk_hits and impact_hits > exec_hits:
        return WEIGHT_PROFILES["impact_focused"]
    if risk_hits > impact_hits and risk_hits > exec_hits:
        return WEIGHT_PROFILES["risk_focused"]
    if exec_hits > impact_hits and exec_hits > risk_hits:
        return WEIGHT_PROFILES["execution_focused"]

    return WEIGHT_PROFILES["default"]


def build_scoring_rubric(weights: ScoringWeights) -> str:
    """Generate a scoring rubric for the judge prompt.

    The rubric instructs the judge to score each axis independently
    (1-10) and compute the final score as a weighted average using the
    given *weights*.
    """
    return f"""\
## Multi-Axis Scoring Rubric (weighted)

Score EACH axis independently from 1-10, then compute the final score as a
weighted average.  Report all four sub-scores in your reasoning.

### 1. Profitability (weight: {weights.profitability:.0%})
Measures absolute and relative PnL performance ($100K starting capital):
- 1-2: Loss or zero trades
- 3-4: PnL $0\u2013$200 (minimal edge captured)
- 5-6: PnL $200\u2013$1,500 (solid market-making or directional edge)
- 7-8: PnL $1,500\u2013$5,000 (strong performance)
- 9-10: PnL > $5,000 (exceptional)

### 2. Risk-Adjusted Performance (weight: {weights.risk_adjusted:.0%})
Measures Sharpe ratio, max drawdown, and inventory risk:
- 1-2: Sharpe < 0 or max drawdown > 50% of PnL
- 3-4: Sharpe 0\u20130.5, moderate drawdown
- 5-6: Sharpe 0.5\u20131.5, controlled drawdown
- 7-8: Sharpe 1.5\u20133.0, tight drawdown management
- 9-10: Sharpe > 3.0, minimal drawdown, low inventory std

### 3. Market Impact (weight: {weights.market_impact:.0%})
Measures whether the strategy is stabilising or destabilising:
- 1-2: Volatility increased > 10% AND spreads widened > 10%
- 3-4: Slight destabilisation (volatility or spread increase 5\u201310%)
- 5-6: Neutral impact (changes within \u00b15%)
- 7-8: Mildly stabilising (spread or volatility reduction 5\u201315%)
- 9-10: Strongly stabilising (measurable market quality improvements)

### 4. Execution Quality (weight: {weights.execution_quality:.0%})
Measures fill rate, order-to-trade ratio, and order management:
- 1-2: Fill rate < 5% or OTT > 50
- 3-4: Fill rate 5\u201315%, OTT 20\u201350
- 5-6: Fill rate 15\u201330%, OTT 10\u201320
- 7-8: Fill rate 30\u201350%, OTT 5\u201310
- 9-10: Fill rate > 50%, OTT < 5, clean order management

### Final Score Calculation
final_score = ({weights.profitability:.2f} \u00d7 profitability) + ({weights.risk_adjusted:.2f} \u00d7 risk_adjusted) + ({weights.market_impact:.2f} \u00d7 market_impact) + ({weights.execution_quality:.2f} \u00d7 execution_quality)

Round the final score to the nearest 0.5.  Report all four sub-scores
in your reasoning before stating the final score."""
