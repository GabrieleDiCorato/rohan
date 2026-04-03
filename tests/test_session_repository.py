"""Tests for SessionRepository — save / list / load / delete refinement runs.

Uses the in-memory SQLite database wired by conftest.py.
"""

import uuid

from rohan.framework.database import DatabaseConnector
from rohan.framework.repository import (
    IterationData,
    ScenarioRunData,
    SessionRepository,
)
from rohan.llm.models import IterationSummary, ScenarioMetrics


class TestSessionRepository:
    """Test suite for SessionRepository operations."""

    def _make_repo(self) -> SessionRepository:
        db = DatabaseConnector()
        db.create_tables()
        return SessionRepository(db)

    def _sample_iteration_data(self, *, n: int = 1) -> list[IterationData]:
        """Build *n* IterationData objects with one scenario run each."""
        result: list[IterationData] = []
        for i in range(1, n + 1):
            result.append(
                IterationData(
                    iteration_number=i,
                    strategy_code=f"class Strategy{i}: pass",
                    class_name=f"Strategy{i}",
                    reasoning=f"Iteration {i} reasoning",
                    judge_score=5.0 + i,
                    judge_reasoning=f"Good progress in iteration {i}",
                    aggregated_explanation=f"Explanation for iteration {i}",
                    axis_scores={
                        "profitability": 6.0 + i,
                        "risk": 5.0 + i,
                        "volatility_impact": 4.0 + i,
                        "spread_impact": 4.5 + i,
                        "liquidity_impact": 5.0 + i,
                        "execution": 7.0 + i,
                        "profile": "default",
                    },
                    scenario_runs=[
                        ScenarioRunData(
                            scenario_name="Default",
                            compiled_config={"agents": [{"type": "NoiseAgent"}]},
                            hasufel_summary={"total_volume": 1000 * i},
                            domain_metrics={
                                "market": {
                                    "vpin": 0.45,
                                    "lob_imbalance_mean": 0.12,
                                    "resilience_mean_ns": 5e8,
                                    "market_ott_ratio": 3.2,
                                    "pct_time_two_sided": 0.85,
                                },
                                "agent": {
                                    "total_pnl": 100.0 * i,
                                    "sharpe_ratio": 0.5 * i,
                                    "max_drawdown": -50.0,
                                    "trade_count": 10 * i,
                                    "fill_rate": 0.25,
                                    "order_to_trade_ratio": 8.5,
                                    "inventory_std": 120.0,
                                    "end_inventory": 5,
                                },
                                "impact": {
                                    "volatility_delta_pct": 0.02,
                                    "spread_delta_pct": -0.01,
                                },
                            },
                            price_chart_b64="cHJpY2U=",
                            spread_chart_b64="c3ByZWFk",
                            volume_chart_b64="dm9sdW1l",
                            pnl_chart_b64="cG5s",
                            inventory_chart_b64="aW52",
                            fill_scatter_b64="ZmlsbA==",
                        ),
                    ],
                )
            )
        return result

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def test_save_session_creates_record(self):
        repo = self._make_repo()
        session = repo.save_session(
            name="Test Run",
            goal="Make a market maker",
            max_iterations=3,
            scenario_configs=[{"name": "Default", "config_override": {}}],
            status="done",
            final_score=8.5,
            total_duration=120.0,
            progress_log=["step 1", "step 2"],
            final_code="class Strategy: pass",
            final_class_name="Strategy",
            final_reasoning="Good strategy",
            iterations=self._sample_iteration_data(n=2),
        )
        assert session.session_id is not None
        assert session.name == "Test Run"
        assert session.goal == "Make a market maker"
        assert session.final_score == 8.5
        assert len(session.iterations) == 2

    def test_save_session_with_no_iterations(self):
        repo = self._make_repo()
        session = repo.save_session(
            name="Empty Run",
            goal="Nothing",
            max_iterations=1,
            scenario_configs=[],
            status="failed",
            final_score=None,
            total_duration=5.0,
            progress_log=[],
            final_code=None,
            final_class_name=None,
            final_reasoning=None,
            iterations=[],
        )
        assert session.session_id is not None
        assert len(session.iterations) == 0

    def test_save_session_preserves_scenario_runs(self):
        repo = self._make_repo()
        session = repo.save_session(
            name="SR Check",
            goal="Test",
            max_iterations=1,
            scenario_configs=[],
            status="done",
            final_score=7.0,
            total_duration=60.0,
            progress_log=[],
            final_code="code",
            final_class_name="S",
            final_reasoning="r",
            iterations=self._sample_iteration_data(n=1),
        )
        it = session.iterations[0]
        assert len(it.scenario_runs) == 1
        sr = it.scenario_runs[0]
        assert sr.scenario_name == "Default"
        assert sr.domain_metrics["agent"]["total_pnl"] == 100.0
        assert sr.domain_metrics["agent"]["trade_count"] == 10
        assert sr.compiled_config == {"agents": [{"type": "NoiseAgent"}]}
        assert sr.hasufel_summary == {"total_volume": 1000}
        # Charts are stored as Artifact rows, loaded via dedicated method
        artifacts = repo.load_scenario_artifacts(session.session_id, 1, "Default")
        assert artifacts["price_chart_b64"] == "cHJpY2U="
        assert artifacts["pnl_chart_b64"] == "cG5s"
        assert artifacts["inventory_chart_b64"] == "aW52"
        assert artifacts["fill_scatter_b64"] == "ZmlsbA=="

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def test_list_sessions_empty(self):
        repo = self._make_repo()
        assert repo.list_sessions() == []

    def test_list_sessions_returns_summaries(self):
        repo = self._make_repo()
        repo.save_session(
            name="Run A",
            goal="Goal A",
            max_iterations=2,
            scenario_configs=[],
            status="done",
            final_score=7.0,
            total_duration=60.0,
            progress_log=[],
            final_code="c",
            final_class_name="C",
            final_reasoning="r",
            iterations=self._sample_iteration_data(n=1),
        )
        repo.save_session(
            name="Run B",
            goal="Goal B",
            max_iterations=3,
            scenario_configs=[],
            status="done",
            final_score=9.0,
            total_duration=90.0,
            progress_log=[],
            final_code="c",
            final_class_name="C",
            final_reasoning="r",
            iterations=self._sample_iteration_data(n=2),
        )
        summaries = repo.list_sessions()
        assert len(summaries) == 2
        names = {s.name for s in summaries}
        assert names == {"Run A", "Run B"}
        by_name = {s.name: s for s in summaries}
        assert by_name["Run B"].final_score == 9.0
        assert by_name["Run B"].iteration_count == 2
        assert by_name["Run A"].iteration_count == 1

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def test_load_session_returns_full_state(self):
        repo = self._make_repo()
        saved = repo.save_session(
            name="Loadable",
            goal="Test goal",
            max_iterations=3,
            scenario_configs=[{"name": "Default", "config_override": {}}],
            status="done",
            final_score=8.0,
            total_duration=100.0,
            progress_log=["log 1"],
            final_code="class S: pass",
            final_class_name="S",
            final_reasoning="because",
            iterations=self._sample_iteration_data(n=2),
        )

        loaded = repo.load_session(saved.session_id)
        assert loaded is not None

        # Top-level session state keys
        assert loaded["refine_goal"] == "Test goal"
        assert loaded["refine_max_iterations"] == 3
        assert loaded["refine_duration"] == 100.0
        assert loaded["refine_saved"] is True
        assert loaded["refine_saved_id"] == saved.session_id
        assert loaded["refine_is_dirty"] is False
        assert loaded["refine_progress"] == ["log 1"]

        # Final state dict
        fs = loaded["refine_final_state"]
        assert fs["goal"] == "Test goal"
        assert fs["current_code"] == "class S: pass"
        assert fs["current_class_name"] == "S"
        assert fs["status"] == "done"

        # Iteration summaries
        iterations = fs["iterations"]
        assert len(iterations) == 2
        assert isinstance(iterations[0], IterationSummary)
        assert iterations[0].iteration_number == 1
        assert iterations[0].judge_score == 6.0
        assert iterations[1].judge_score == 7.0

        # Scenario metrics inside iteration
        sm = iterations[0].scenario_metrics["Default"]
        assert isinstance(sm, ScenarioMetrics)
        assert sm.total_pnl == 100.0
        assert sm.trade_count == 10
        # Charts are lazy-loaded via load_scenario_artifacts(), not on ScenarioMetrics
        assert sm.price_chart_b64 is None
        assert sm.pnl_chart_b64 is None
        assert sm.inventory_chart_b64 is None
        assert sm.fill_scatter_b64 is None
        # Verify charts persisted via artifact loading
        arts = repo.load_scenario_artifacts(saved.session_id, 1, "Default")
        assert arts["price_chart_b64"] == "cHJpY2U="
        assert arts["pnl_chart_b64"] == "cG5s"
        assert arts["inventory_chart_b64"] == "aW52"
        assert arts["fill_scatter_b64"] == "ZmlsbA=="

    def test_load_session_preserves_sub_scores(self):
        """Sub-scores and new metric fields survive a save/load round-trip."""
        repo = self._make_repo()
        saved = repo.save_session(
            name="Sub-score Round-trip",
            goal="Test sub-scores",
            max_iterations=1,
            scenario_configs=[],
            status="done",
            final_score=7.5,
            total_duration=30.0,
            progress_log=[],
            final_code="class S: pass",
            final_class_name="S",
            final_reasoning="r",
            iterations=self._sample_iteration_data(n=1),
        )

        loaded = repo.load_session(saved.session_id)
        assert loaded is not None
        it = loaded["refine_final_state"]["iterations"][0]

        # Sub-scores on iteration
        assert it.profitability_score == 7.0
        assert it.risk_score == 6.0
        assert it.volatility_impact_score == 5.0
        assert it.spread_impact_score == 5.5
        assert it.liquidity_impact_score == 6.0
        assert it.execution_score == 8.0
        assert it.scoring_profile == "default"

        # Extended metrics on scenario result
        sm = it.scenario_metrics["Default"]
        assert sm.fill_rate == 0.25
        assert sm.order_to_trade_ratio == 8.5
        assert sm.inventory_std == 120.0
        assert sm.end_inventory == 5

    def test_load_session_preserves_hasufel_objects(self):
        """compiled_config and hasufel_summary survive a save/load round-trip."""
        repo = self._make_repo()
        saved = repo.save_session(
            name="Hasufel Round-trip",
            goal="Test hasufel objects",
            max_iterations=1,
            scenario_configs=[],
            status="done",
            final_score=7.0,
            total_duration=30.0,
            progress_log=[],
            final_code="class S: pass",
            final_class_name="S",
            final_reasoning="r",
            iterations=self._sample_iteration_data(n=1),
        )

        loaded = repo.load_session(saved.session_id)
        assert loaded is not None
        sm = loaded["refine_final_state"]["iterations"][0].scenario_metrics["Default"]
        assert sm.compiled_config == {"agents": [{"type": "NoiseAgent"}]}
        assert sm.hasufel_summary == {"total_volume": 1000}

    def test_load_nonexistent_returns_none(self):
        repo = self._make_repo()
        assert repo.load_session(uuid.uuid4()) is None

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def test_delete_session(self):
        repo = self._make_repo()
        saved = repo.save_session(
            name="Del",
            goal="g",
            max_iterations=1,
            scenario_configs=[],
            status="done",
            final_score=None,
            total_duration=1.0,
            progress_log=[],
            final_code=None,
            final_class_name=None,
            final_reasoning=None,
            iterations=[],
        )
        assert repo.delete_session(saved.session_id) is True
        assert repo.load_session(saved.session_id) is None

    def test_delete_nonexistent_returns_false(self):
        repo = self._make_repo()
        assert repo.delete_session(uuid.uuid4()) is False

    def test_delete_cascades_children(self):
        """Deleting a session should also remove its iterations and scenario runs."""
        repo = self._make_repo()
        saved = repo.save_session(
            name="Cascade",
            goal="g",
            max_iterations=1,
            scenario_configs=[],
            status="done",
            final_score=7.0,
            total_duration=30.0,
            progress_log=[],
            final_code="c",
            final_class_name="C",
            final_reasoning="r",
            iterations=self._sample_iteration_data(n=2),
        )
        sid = saved.session_id
        assert repo.delete_session(sid) is True
        # Verify the session is gone
        assert repo.list_sessions() == []
