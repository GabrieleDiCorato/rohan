"""Tests for the Single Iteration Pipeline (Phase 1.5.6).

Validates the end-to-end cycle:
    validate → execute → analyse → (optionally persist)

Also covers failure-path tests (§2.7.8a):
    - Baseline simulation failure
    - Strategy timeout
    - Runtime execution errors
"""

from unittest.mock import patch

import pytest

from rohan.config import SimulationSettings
from rohan.exceptions import SimulationTimeoutError, StrategyExecutionError
from rohan.framework import IterationPipeline, PipelineConfig
from rohan.framework.database import DatabaseConnector, RunStatus
from rohan.simulation.models import SimulationContext, SimulationResult

# A minimal valid strategy that buys 1 share whenever the market moves.
SIMPLE_STRATEGY = """\
from rohan.simulation.models.strategy_api import (
    AgentConfig, MarketState, OrderAction, OrderType, Side,
)


class SimpleStrategy:
    def initialize(self, config: AgentConfig) -> None:
        self.symbol = config.symbol

    def on_tick(self, state: MarketState) -> list[OrderAction]:
        return []

    def on_market_data(self, state: MarketState) -> list[OrderAction]:
        if state.best_bid is not None:
            return [
                OrderAction(
                    side=Side.BID,
                    quantity=1,
                    price=state.best_bid,
                    order_type=OrderType.LIMIT,
                ),
            ]
        return []

    def on_order_update(self, update) -> list[OrderAction]:
        return []

    def on_simulation_end(self, final_state: MarketState) -> None:
        pass
"""

# Strategy code with a forbidden import.
BAD_STRATEGY = """\
import os
class EvilStrategy:
    def initialize(self, config): pass
    def on_tick(self, state): return []
    def on_market_data(self, state): return []
    def on_order_update(self, update): return []
    def on_simulation_end(self, final_state): pass
"""

# Strategy that passes AST validation but raises at runtime in on_market_data.
RUNTIME_ERROR_STRATEGY = """\
from rohan.simulation.models.strategy_api import (
    AgentConfig, MarketState, OrderAction, OrderType, Side,
)


class RuntimeErrorStrategy:
    def initialize(self, config: AgentConfig) -> None:
        self.symbol = config.symbol

    def on_tick(self, state: MarketState) -> list[OrderAction]:
        return []

    def on_market_data(self, state: MarketState) -> list[OrderAction]:
        raise RuntimeError("deliberate runtime error")

    def on_order_update(self, update) -> list[OrderAction]:
        return []

    def on_simulation_end(self, final_state: MarketState) -> None:
        pass
"""


def _minimal_settings() -> SimulationSettings:
    s = SimulationSettings()
    s.start_time = "09:30:00"
    s.end_time = "09:35:00"
    s.agents.noise.num_agents = 2
    s.agents.value.num_agents = 1
    s.agents.momentum.num_agents = 0
    s.agents.adaptive_market_maker.num_agents = 0
    return s


class TestIterationPipeline:
    """Test suite for the IterationPipeline."""

    def test_invalid_strategy_returns_validation_errors(self):
        """Validation failures should be captured without running the simulation."""
        pipeline = IterationPipeline()
        config = PipelineConfig(settings=_minimal_settings())
        result = pipeline.run(BAD_STRATEGY, config)

        assert not result.success
        assert not result.validation.is_valid
        assert any("os" in e for e in result.validation.errors)
        assert result.comparison is None

    def test_simple_strategy_succeeds(self):
        """A valid strategy should produce metrics and an interpreter prompt."""
        pipeline = IterationPipeline()
        config = PipelineConfig(
            settings=_minimal_settings(),
            goal_description="Simple test strategy",
        )
        result = pipeline.run(SIMPLE_STRATEGY, config)

        assert result.success, f"Pipeline failed: {result.error}"
        assert result.validation.is_valid
        assert result.comparison is not None
        assert result.summary is not None
        assert result.interpreter_prompt is not None
        assert result.duration_seconds > 0

        # Verify comparison has expected structure
        comp = result.comparison
        assert comp.strategy_metrics is not None
        assert comp.market_impact is not None

    def test_pipeline_with_persistence(self):
        """When persist=True, iteration and run records should be stored."""
        db = DatabaseConnector()
        db.create_tables()

        from rohan.framework.repository import ArtifactStore

        repo = ArtifactStore(db)
        session_obj = repo.create_session("test pipeline persistence")

        pipeline = IterationPipeline(db=db)
        config = PipelineConfig(
            settings=_minimal_settings(),
            goal_description="Persistence test",
            persist=True,
            session_id=session_obj.session_id,
        )
        result = pipeline.run(SIMPLE_STRATEGY, config)

        assert result.success, f"Pipeline failed: {result.error}"
        assert result.iteration_id is not None

        # Verify DB records
        runs = repo.get_session_runs(session_obj.session_id)
        assert len(runs) >= 1
        assert runs[0].status == RunStatus.COMPLETED


# ---------------------------------------------------------------------------
# 2.7.8a — Failure-path tests
# ---------------------------------------------------------------------------


class TestIterationPipelineFailurePaths:
    """Failure-path tests for IterationPipeline (§2.7.8a)."""

    def test_baseline_failure_captured_as_error(self):
        """When the baseline simulation fails, the pipeline should capture the
        error in IterationResult.error (not crash)."""
        pipeline = IterationPipeline()
        config = PipelineConfig(settings=_minimal_settings())

        # Let the strategy run succeed, but make the baseline fail.
        original_run = pipeline._service.run_simulation

        def _patched_run(settings, strategy=None, **kw):
            if strategy is None:  # baseline call
                return SimulationResult(
                    context=SimulationContext(settings=settings),
                    duration_seconds=0.0,
                    error=RuntimeError("simulated baseline failure"),
                )
            return original_run(settings, strategy=strategy, **kw)

        with patch.object(pipeline._service, "run_simulation", side_effect=_patched_run):
            result = pipeline.run(SIMPLE_STRATEGY, config)

        assert not result.success
        assert result.error is not None
        assert "Baseline" in result.error or "baseline" in result.error

    def test_strategy_timeout_captured_as_error(self):
        """When execute_strategy_safely raises SimulationTimeoutError, the
        pipeline should capture it gracefully."""
        pipeline = IterationPipeline()
        config = PipelineConfig(settings=_minimal_settings(), timeout_seconds=1)

        with patch(
            "rohan.framework.iteration_pipeline.execute_strategy_safely",
            side_effect=SimulationTimeoutError("timed out after 1.0s (limit: 1s)"),
        ):
            result = pipeline.run(SIMPLE_STRATEGY, config)

        assert not result.success
        assert result.error is not None
        assert "timed out" in result.error.lower() or "timeout" in result.error.lower()

    def test_strategy_execution_error_captured(self):
        """When execute_strategy_safely raises StrategyExecutionError, the
        pipeline should capture it gracefully."""
        pipeline = IterationPipeline()
        config = PipelineConfig(settings=_minimal_settings())

        with patch(
            "rohan.framework.iteration_pipeline.execute_strategy_safely",
            side_effect=StrategyExecutionError("exec() failed: name 'undefined_var' is not defined"),
        ):
            result = pipeline.run(SIMPLE_STRATEGY, config)

        assert not result.success
        assert result.error is not None
        assert "exec() failed" in result.error or "Execution error" in result.error

    def test_runtime_error_strategy_returns_error_result(self):
        """A strategy that passes AST validation but raises at runtime should
        produce a failed IterationResult with a descriptive error."""
        pipeline = IterationPipeline()
        config = PipelineConfig(settings=_minimal_settings())
        result = pipeline.run(RUNTIME_ERROR_STRATEGY, config)

        assert not result.success
        assert result.validation.is_valid  # AST check passes
        assert result.error is not None
        assert "deliberate runtime error" in result.error or "Execution error" in result.error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
