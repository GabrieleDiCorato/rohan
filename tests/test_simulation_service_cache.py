from unittest.mock import MagicMock

import pandas as pd

from rohan.config import SimulationSettings
from rohan.simulation.models.simulation_output import SimulationOutput
from rohan.simulation.simulation_service import SimulationService


class _FakeRunner:
    def __init__(self, output):
        self._output = output
        self.run_calls = 0

    def validate(self):
        return []

    def run(self):
        self.run_calls += 1
        return self._output

    def cleanup(self):
        return None


class _FakeOutput(SimulationOutput):
    def get_order_book_l1(self):
        return pd.DataFrame()

    def get_order_book_l2(self, n_levels: int):
        _ = n_levels
        return pd.DataFrame()

    def get_logs_df(self):
        return pd.DataFrame()

    def get_logs_by_agent(self):
        return {}


def _settings(seed: int = 42) -> SimulationSettings:
    settings = SimulationSettings(
        seed=seed,
        start_time="09:30:00",
        end_time="09:35:00",
        stdout_log_level="OFF",
        baseline_cache_enabled=True,
        baseline_cache_max_entries=8,
    )
    settings.agents.noise.num_agents = 1
    settings.agents.value.num_agents = 0
    settings.agents.momentum.num_agents = 0
    settings.agents.adaptive_market_maker.num_agents = 0
    return settings


def test_baseline_cache_hit_reuses_output_without_second_run():
    service = SimulationService()
    service.clear_baseline_cache()

    output = _FakeOutput()
    runner = _FakeRunner(output)
    service._create_runner = MagicMock(return_value=runner)  # type: ignore[method-assign]

    settings = _settings()

    first = service.run_simulation(settings=settings, strategy=None)
    second = service.run_simulation(settings=settings, strategy=None)

    assert first.error is None
    assert second.error is None
    assert first.result is output
    assert second.result is output
    assert runner.run_calls == 1
    assert second.duration_seconds == 0.0


def test_strategy_run_does_not_use_baseline_cache():
    service = SimulationService()
    service.clear_baseline_cache()

    output = _FakeOutput()
    runner = _FakeRunner(output)
    service._create_runner = MagicMock(return_value=runner)  # type: ignore[method-assign]

    settings = _settings()

    strategy = MagicMock()
    first = service.run_simulation(settings=settings, strategy=strategy)
    second = service.run_simulation(settings=settings, strategy=strategy)

    assert first.error is None
    assert second.error is None
    assert runner.run_calls == 2


def test_baseline_cache_respects_max_entries():
    service = SimulationService()
    service.clear_baseline_cache()

    output = _FakeOutput()
    runner = _FakeRunner(output)
    service._create_runner = MagicMock(return_value=runner)  # type: ignore[method-assign]

    settings_a = _settings(seed=1)
    settings_a.baseline_cache_max_entries = 1
    settings_b = _settings(seed=2)
    settings_b.baseline_cache_max_entries = 1

    service.run_simulation(settings=settings_a, strategy=None)
    service.run_simulation(settings=settings_b, strategy=None)

    assert len(service._baseline_cache) == 1
