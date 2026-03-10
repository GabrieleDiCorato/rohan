"""Deterministic multi-axis scoring for strategy evaluation.

Replaces the former LLM-scored rubric with pure-Python formulas that
are baseline-relative, automatically calibrated, and noise-free.

Six axes are evaluated independently:
1. **Profitability** — opportunity capture rate (PnL / theoretical spread revenue)
2. **Risk-Adjusted Performance** — Sharpe ratio + drawdown penalty
3. **Volatility Impact** — baseline-relative volatility change
4. **Spread Impact** — baseline-relative spread change
5. **Liquidity Impact** — baseline-relative liquidity change (inverted: positive = good)
6. **Execution Quality** — fill rate + OTT penalty

Weight profiles shift emphasis based on the goal (e.g. a "minimise VPIN"
goal up-weights impact axes; a "maximise Sharpe" goal up-weights Risk).

All scores are clamped to [1.0, 10.0].
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Helpers ──────────────────────────────────────────────────────────────


def _lerp(value: float, in_lo: float, in_hi: float, out_lo: float, out_hi: float) -> float:
    """Linearly interpolate *value* from [in_lo, in_hi] to [out_lo, out_hi].

    Values outside the input range are clamped to the output range.
    """
    if in_hi == in_lo:
        return (out_lo + out_hi) / 2
    t = (value - in_lo) / (in_hi - in_lo)
    t = max(0.0, min(1.0, t))
    return out_lo + t * (out_hi - out_lo)


def _clamp(value: float, lo: float = 1.0, hi: float = 10.0) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


# ── Scoring weights ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScoringWeights:
    """Weights for each scoring axis.  Must sum to 1.0."""

    profitability: float
    risk_adjusted: float
    volatility_impact: float
    spread_impact: float
    liquidity_impact: float
    execution_quality: float

    def __post_init__(self) -> None:
        total = self.profitability + self.risk_adjusted + self.volatility_impact + self.spread_impact + self.liquidity_impact + self.execution_quality
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")


# ── Pre-defined weight profiles ──────────────────────────────────────────

WEIGHT_PROFILES: dict[str, ScoringWeights] = {
    "default": ScoringWeights(
        profitability=0.35,
        risk_adjusted=0.20,
        volatility_impact=0.07,
        spread_impact=0.08,
        liquidity_impact=0.05,
        execution_quality=0.25,
    ),
    "impact_focused": ScoringWeights(
        profitability=0.15,
        risk_adjusted=0.10,
        volatility_impact=0.20,
        spread_impact=0.20,
        liquidity_impact=0.15,
        execution_quality=0.20,
    ),
    "risk_focused": ScoringWeights(
        profitability=0.15,
        risk_adjusted=0.40,
        volatility_impact=0.07,
        spread_impact=0.08,
        liquidity_impact=0.05,
        execution_quality=0.25,
    ),
    "execution_focused": ScoringWeights(
        profitability=0.15,
        risk_adjusted=0.15,
        volatility_impact=0.07,
        spread_impact=0.08,
        liquidity_impact=0.05,
        execution_quality=0.50,
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


# ── Axis scores ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AxisScores:
    """Deterministic sub-scores for each scoring axis (1.0–10.0)."""

    profitability: float
    risk_adjusted: float
    volatility_impact: float
    spread_impact: float
    liquidity_impact: float
    execution_quality: float


# ── Axis scoring functions ───────────────────────────────────────────────


def _score_profitability(
    strategy_pnl: float | None,
    trade_count: int,
    baseline_mean_spread: float | None,
    baseline_traded_volume: float | None,
) -> float:
    """Score profitability as opportunity capture rate.

    opportunity = baseline_mean_spread × baseline_traded_volume / 2
    capture_rate = strategy_pnl / opportunity

    Piecewise mapping:
        capture_rate < 0        → 1–3   (losing money)
        0 → 0.001               → 3–5   (minimal capture)
        0.001 → 0.005           → 5–7   (solid capture)
        0.005 → 0.02            → 7–9   (strong capture)
        > 0.02                  → 9–10  (exceptional, capped)
    """
    if trade_count == 0 or strategy_pnl is None:
        return 1.0

    if not baseline_mean_spread or not baseline_traded_volume:
        return 1.0

    opportunity = baseline_mean_spread * baseline_traded_volume / 2.0
    if opportunity <= 0:
        return 1.0

    capture_rate = strategy_pnl / opportunity

    if capture_rate < 0:
        # Map most-negative (capped at -0.02) to 1.0, zero to 3.0
        return _clamp(_lerp(capture_rate, -0.02, 0.0, 1.0, 3.0))
    if capture_rate <= 0.001:
        return _clamp(_lerp(capture_rate, 0.0, 0.001, 3.0, 5.0))
    if capture_rate <= 0.005:
        return _clamp(_lerp(capture_rate, 0.001, 0.005, 5.0, 7.0))
    if capture_rate <= 0.02:
        return _clamp(_lerp(capture_rate, 0.005, 0.02, 7.0, 9.0))
    return _clamp(_lerp(capture_rate, 0.02, 0.04, 9.0, 10.0))


def _score_risk(
    sharpe_ratio: float | None,
    trade_count: int,
    max_drawdown: float | None,
    starting_capital_cents: int,
) -> float:
    """Score risk-adjusted performance from Sharpe ratio + drawdown penalty.

    Sharpe piecewise:
        None (< 5 trades)      → 3.0
        None (≥ 5 trades)      → 2.0
        < −1                   → 1.0
        −1 → 0                 → 1–3
        0 → 0.5                → 3–4
        0.5 → 1.5              → 4–6
        1.5 → 3.0              → 6–8
        > 3.0                  → 8–10

    Drawdown penalty (% of starting capital):
        > 5%   → −2
        > 2%   → −1
    """
    if sharpe_ratio is None:
        base = 3.0 if trade_count < 5 else 2.0
    elif sharpe_ratio < -1.0:
        base = 1.0
    elif sharpe_ratio <= 0.0:
        base = _lerp(sharpe_ratio, -1.0, 0.0, 1.0, 3.0)
    elif sharpe_ratio <= 0.5:
        base = _lerp(sharpe_ratio, 0.0, 0.5, 3.0, 4.0)
    elif sharpe_ratio <= 1.5:
        base = _lerp(sharpe_ratio, 0.5, 1.5, 4.0, 6.0)
    elif sharpe_ratio <= 3.0:
        base = _lerp(sharpe_ratio, 1.5, 3.0, 6.0, 8.0)
    else:
        base = _lerp(sharpe_ratio, 3.0, 5.0, 8.0, 10.0)

    # Drawdown penalty (as % of starting capital)
    if max_drawdown is not None and starting_capital_cents > 0:
        drawdown_pct = abs(max_drawdown) / starting_capital_cents
        if drawdown_pct > 0.05:
            base -= 2.0
        elif drawdown_pct > 0.02:
            base -= 1.0

    return _clamp(base)


def _score_negative_impact(delta_pct: float | None) -> float:
    """Score a baseline-relative impact where *negative* change is good.

    Used for volatility and spread: lower = better (stabilising / tightening).

    Piecewise:
        > +10%                 → 1.0
        +5% → +10%             → 1–4
        −5% → +5%              → 4–7
        −15% → −5%             → 7–9
        < −15%                 → 9.0 (capped)
    """
    if delta_pct is None:
        return 5.5  # no data → neutral

    if delta_pct > 0.10:
        return 1.0
    if delta_pct > 0.05:
        return _clamp(_lerp(delta_pct, 0.05, 0.10, 4.0, 1.0))
    if delta_pct > -0.05:
        return _clamp(_lerp(delta_pct, -0.05, 0.05, 7.0, 4.0))
    if delta_pct > -0.15:
        return _clamp(_lerp(delta_pct, -0.15, -0.05, 9.0, 7.0))
    return 9.0


def _score_liquidity_impact(
    delta_pct: float | None,
    availability_delta_pct: float | None = None,
) -> float:
    """Score liquidity change where *positive* change is good.

    Inverted vs volatility/spread: more liquidity = better.

    Piecewise:
        < −15%                 → 1.0
        −15% → −5%             → 1–4
        −5% → +5%              → 4–7
        +5% → +15%             → 7–9
        > +15%                 → 9.0 (capped)

    Availability penalty (when strategy reduces market availability):
        > 10% drop  → −2
        > 5% drop   → −1
    """
    if delta_pct is None:
        base = 5.5  # no data → neutral
    elif delta_pct < -0.15:
        base = 1.0
    elif delta_pct < -0.05:
        base = _clamp(_lerp(delta_pct, -0.15, -0.05, 1.0, 4.0))
    elif delta_pct < 0.05:
        base = _clamp(_lerp(delta_pct, -0.05, 0.05, 4.0, 7.0))
    elif delta_pct < 0.15:
        base = _clamp(_lerp(delta_pct, 0.05, 0.15, 7.0, 9.0))
    else:
        base = 9.0

    # Penalise strategies that reduce market availability
    if availability_delta_pct is not None and availability_delta_pct < 0:
        if availability_delta_pct < -0.10:
            base -= 2.0
        elif availability_delta_pct < -0.05:
            base -= 1.0

    return _clamp(base)


def _score_execution(
    fill_rate: float | None,
    order_to_trade_ratio: float | None,
) -> float:
    """Score execution quality from fill rate (primary) + OTT penalty.

    Fill rate piecewise:
        < 5%          → 1–2
        5–15%         → 2–4
        15–30%        → 4–6
        30–50%        → 6–8
        > 50%         → 8–10

    OTT penalty:
        > 200  → −2
        > 100  → −1
    """
    if fill_rate is None:
        base = 1.0
    elif fill_rate < 0.05:
        base = _lerp(fill_rate, 0.0, 0.05, 1.0, 2.0)
    elif fill_rate < 0.15:
        base = _lerp(fill_rate, 0.05, 0.15, 2.0, 4.0)
    elif fill_rate < 0.30:
        base = _lerp(fill_rate, 0.15, 0.30, 4.0, 6.0)
    elif fill_rate < 0.50:
        base = _lerp(fill_rate, 0.30, 0.50, 6.0, 8.0)
    else:
        base = _lerp(fill_rate, 0.50, 0.80, 8.0, 10.0)

    if order_to_trade_ratio is not None:
        if order_to_trade_ratio > 200:
            base -= 2.0
        elif order_to_trade_ratio > 100:
            base -= 1.0

    return _clamp(base)


# ── Public API ───────────────────────────────────────────────────────────


def compute_axis_scores(
    *,
    strategy_pnl: float | None,
    trade_count: int,
    sharpe_ratio: float | None,
    max_drawdown: float | None,
    fill_rate: float | None,
    order_to_trade_ratio: float | None,
    volatility_delta_pct: float | None,
    spread_delta_pct: float | None,
    bid_liquidity_delta_pct: float | None,
    ask_liquidity_delta_pct: float | None,
    starting_capital_cents: int,
    baseline_mean_spread: float | None,
    baseline_traded_volume: float | None,
    pct_time_two_sided_delta: float | None = None,
) -> AxisScores:
    """Compute all 6 deterministic axis scores from simulation metrics.

    All parameters are keyword-only to prevent positional argument errors.
    Returns an :class:`AxisScores` with each field clamped to [1.0, 10.0].
    """
    # Liquidity: average of bid and ask deltas
    if bid_liquidity_delta_pct is not None and ask_liquidity_delta_pct is not None:
        liquidity_delta = (bid_liquidity_delta_pct + ask_liquidity_delta_pct) / 2.0
    elif bid_liquidity_delta_pct is not None:
        liquidity_delta = bid_liquidity_delta_pct
    elif ask_liquidity_delta_pct is not None:
        liquidity_delta = ask_liquidity_delta_pct
    else:
        liquidity_delta = None

    return AxisScores(
        profitability=_score_profitability(strategy_pnl, trade_count, baseline_mean_spread, baseline_traded_volume),
        risk_adjusted=_score_risk(sharpe_ratio, trade_count, max_drawdown, starting_capital_cents),
        volatility_impact=_score_negative_impact(volatility_delta_pct),
        spread_impact=_score_negative_impact(spread_delta_pct),
        liquidity_impact=_score_liquidity_impact(liquidity_delta, pct_time_two_sided_delta),
        execution_quality=_score_execution(fill_rate, order_to_trade_ratio),
    )


def compute_final_score(axis_scores: AxisScores, weights: ScoringWeights) -> float:
    """Compute weighted average of axis scores, rounded to nearest 0.5.

    The result is clamped to [1.0, 10.0].
    """
    raw = (
        axis_scores.profitability * weights.profitability
        + axis_scores.risk_adjusted * weights.risk_adjusted
        + axis_scores.volatility_impact * weights.volatility_impact
        + axis_scores.spread_impact * weights.spread_impact
        + axis_scores.liquidity_impact * weights.liquidity_impact
        + axis_scores.execution_quality * weights.execution_quality
    )
    # Round to nearest 0.5
    rounded = round(raw * 2) / 2
    return _clamp(rounded)
