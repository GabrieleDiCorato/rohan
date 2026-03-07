"""Tests for prompt templates (Phase 2.1)."""

from rohan.llm.prompts import (
    AGGREGATOR_HUMAN,
    AGGREGATOR_SYSTEM,
    EXPLAINER_HUMAN,
    EXPLAINER_SYSTEM,
    HISTORY_ROW_TEMPLATE,
    HISTORY_TABLE_HEADER,
    WRITER_FEEDBACK_TEMPLATE,
    WRITER_HUMAN,
    WRITER_SYSTEM,
)
from rohan.llm.scoring import WEIGHT_PROFILES, build_scoring_rubric


class TestWriterPrompts:
    def test_system_prompt_contains_protocol(self):
        assert "StrategicAgent" in WRITER_SYSTEM
        assert "initialize" in WRITER_SYSTEM
        assert "on_market_data" in WRITER_SYSTEM
        assert "on_order_update" in WRITER_SYSTEM

    def test_system_prompt_mentions_types(self):
        assert "MarketState" in WRITER_SYSTEM
        assert "OrderAction" in WRITER_SYSTEM
        assert "AgentConfig" in WRITER_SYSTEM
        assert "integer cents" in WRITER_SYSTEM

    def test_system_prompt_mentions_allowed_imports(self):
        assert "math" in WRITER_SYSTEM
        assert "numpy" in WRITER_SYSTEM
        assert "pandas" in WRITER_SYSTEM

    def test_human_prompt_has_slots(self):
        assert "{goal}" in WRITER_HUMAN
        assert "{feedback_section}" in WRITER_HUMAN

    def test_human_prompt_renders(self):
        rendered = WRITER_HUMAN.format(
            goal="Make a market-maker",
            feedback_section="No feedback yet",
        )
        assert "Make a market-maker" in rendered
        assert "No feedback yet" in rendered

    def test_feedback_template_renders(self):
        rendered = WRITER_FEEDBACK_TEMPLATE.format(
            iteration_number=1,
            score=6.5,
            metrics_summary="- default: PnL=$23.18, Trades=140",
            strengths="- Good PnL",
            weaknesses="High drawdown",
            recommendations="Reduce size",
            previous_code="class X: pass",
            iteration_history="(No previous iterations)",
        )
        assert "6.5" in rendered
        assert "class X: pass" in rendered
        assert "$23.18" in rendered


class TestExplainerPrompts:
    def test_system_prompt_content(self):
        assert "quantitative analyst" in EXPLAINER_SYSTEM
        assert "code-level" in EXPLAINER_SYSTEM

    def test_human_prompt_has_slots(self):
        assert "{scenario_name}" in EXPLAINER_HUMAN
        assert "{interpreter_prompt}" in EXPLAINER_HUMAN
        assert "{strategy_code}" in EXPLAINER_HUMAN


class TestAggregatorPrompts:
    def test_system_prompt_content(self):
        rubric = build_scoring_rubric(WEIGHT_PROFILES["default"])
        rendered = AGGREGATOR_SYSTEM.format(scoring_rubric=rubric)
        assert "stop_converged" in rendered
        assert "stop_plateau" in rendered
        assert "1-10" in rendered

    def test_system_prompt_has_rubric_slot(self):
        assert "{scoring_rubric}" in AGGREGATOR_SYSTEM

    def test_system_prompt_renders_with_rubric(self):
        rubric = build_scoring_rubric(WEIGHT_PROFILES["default"])
        rendered = AGGREGATOR_SYSTEM.format(scoring_rubric=rubric)
        assert "Profitability" in rendered
        assert "Risk-Adjusted" in rendered
        assert "Market Impact" in rendered
        assert "Execution Quality" in rendered
        assert "weight:" in rendered

    def test_human_prompt_has_slots(self):
        assert "{goal}" in AGGREGATOR_HUMAN
        assert "{best_iteration_line}" in AGGREGATOR_HUMAN
        assert "{history_table}" in AGGREGATOR_HUMAN
        assert "{iteration_number}" in AGGREGATOR_HUMAN
        assert "{explanations}" in AGGREGATOR_HUMAN

    def test_history_table_header(self):
        assert "Iter" in HISTORY_TABLE_HEADER
        assert "PnL" in HISTORY_TABLE_HEADER
        assert "Score" in HISTORY_TABLE_HEADER

    def test_history_row_renders(self):
        row = HISTORY_ROW_TEMPLATE.format(
            iter=1,
            pnl="$1.00",
            trades="142",
            fill_rate="4.2%",
            vol_delta="+5.0%",
            spread_delta="-2.0%",
            score="7.0",
            summary="Good iteration",
        )
        assert "$1.00" in row
        assert "7.0" in row
        assert "4.2%" in row
