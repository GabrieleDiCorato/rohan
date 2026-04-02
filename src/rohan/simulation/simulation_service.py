"""Service module for simulation-related operations. Injects dependencies and manages simulation execution."""

import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any, cast

from rohan.config import OracleType, SimulationEngine, SimulationSettings, get_feature_flags
from rohan.llm.telemetry import emit_metric
from rohan.simulation.models import (
    SimulationContext,
    SimulationResult,
)
from rohan.simulation.models.simulation_output import SimulationOutput
from rohan.simulation.models.strategy_api import StrategicAgent
from rohan.simulation.models.strategy_spec import StrategySpec
from rohan.simulation.simulation_runner import SimulationRunner

logger = logging.getLogger(__name__)


class SimulationService:
    """Stateless service class for managing simulations.

    This service is designed to be reusable and can run multiple simulations
    with different settings without needing to create new instances.

    All methods return SimulationResult for uniform handling of success/failure.
    SimulationResult uses a Result-type pattern where either 'result' or 'error'
    is set, but never both. Check `result.error is None` for success.

    Example:
        >>> service = SimulationService()
        >>> result = service.run_simulation(settings)
        >>> if result.error is None:
        ...     data = result.result.get_order_book_l1()
        ... else:
        ...     print(f"Failed: {result.error}")
        >>>
        >>> # Batch processing
        >>> results = service.run_batch([settings1, settings2, settings3])
        >>> successful = [r.result for r in results if r.error is None]
        >>> failed = [r for r in results if r.error is not None]
    """

    _baseline_cache: OrderedDict[str, SimulationOutput] = OrderedDict()

    @staticmethod
    def _emit_simulation_metric(event: str, **fields: Any) -> None:
        """Emit structured telemetry for simulation service metrics."""
        emit_metric(event, component="rohan.simulation", **fields)

    def run_simulation(
        self,
        settings: SimulationSettings,
        context: SimulationContext | None = None,
        strategy_spec: StrategySpec | None = None,
        strategy: StrategicAgent | None = None,
    ) -> SimulationResult:
        """Runs a single simulation with the given settings.

        Args:
            settings: Configuration for the simulation.
            context: Optional context for tracking metadata. If not provided, one will be created.
            strategy_spec: Optional serializable StrategySpec — preferred path for
                LLM-generated strategies (goes through hasufel's config pipeline).
            strategy: Optional pre-built StrategicAgent instance — legacy path
                useful for tests that inspect strategy state after simulation.

        Returns:
            SimulationResult: Result object containing output, metadata, and status.

        Note:
            Always returns a SimulationResult, even on failure (with success=False).
        """
        # Create context if not provided
        if context is None:
            context = SimulationContext(settings=settings)

        feature_flags = get_feature_flags()
        cache_key: str | None = None
        if strategy_spec is None and strategy is None and settings.baseline_cache_enabled and feature_flags.baseline_cache_v1:
            cache_key = self._build_baseline_cache_key(settings)
            cached_output = self._baseline_cache.get(cache_key)
            if cached_output is not None:
                self._baseline_cache.move_to_end(cache_key)
                logger.info("Baseline cache hit: %s", cache_key[:12])
                if feature_flags.llm_telemetry_v1:
                    self._emit_simulation_metric("baseline_cache_hit", cache_key=cache_key[:12])
                return SimulationResult(
                    context=context,
                    duration_seconds=0.0,
                    result=cast(SimulationOutput, cached_output),
                )

        start_time = time.time()

        try:
            runner = self._create_runner(settings, strategy_spec=strategy_spec, strategy=strategy)

            # Validate before running
            errors = runner.validate()
            if errors:
                raise ValueError(f"Simulation validation failed: {', '.join(errors)}")

            try:
                output = runner.run()
                duration = time.time() - start_time

                if cache_key is not None:
                    self._baseline_cache[cache_key] = output
                    self._baseline_cache.move_to_end(cache_key)
                    while len(self._baseline_cache) > settings.baseline_cache_max_entries:
                        self._baseline_cache.popitem(last=False)
                    logger.info("Baseline cache stored: %s (size=%d)", cache_key[:12], len(self._baseline_cache))
                    if feature_flags.llm_telemetry_v1:
                        self._emit_simulation_metric(
                            "baseline_cache_store",
                            cache_key=cache_key[:12],
                            cache_size=len(self._baseline_cache),
                        )

                return SimulationResult(
                    context=context,
                    duration_seconds=duration,
                    result=output,
                )
            finally:
                # Always cleanup, even if simulation fails
                runner.cleanup()

        except Exception as e:
            duration = time.time() - start_time
            return SimulationResult(
                context=context,
                duration_seconds=duration,
                error=e,
            )

    def run_batch(
        self,
        settings_list: list[SimulationSettings],
        fail_fast: bool = False,
        n_workers: int | None = None,
        strategy_spec: StrategySpec | None = None,
    ) -> list[SimulationResult]:
        """Run multiple simulations with different settings.

        When *n_workers* > 1 and all settings use a non-historical oracle,
        simulations are executed in parallel via hasufel's
        ``run_batch()``.  Otherwise they run sequentially.

        Args:
            settings_list: List of simulation configurations to run.
            fail_fast: If True, stop on first failure. If False, continue and collect all results.
            n_workers: Number of parallel workers.  ``None`` or 1 for sequential
                execution, > 1 for parallel via hasufel ``run_batch()``.
            strategy_spec: Optional serializable StrategySpec to include in all
                batch configs.  Workers compile the spec in-process via the
                registered ``rohan_strategy`` agent type.

        Returns:
            List of SimulationResult objects containing outputs and metadata.

        Example:
            >>> service = SimulationService()
            >>> settings_list = [SimulationSettings(seed=i) for i in range(10)]
            >>> results = service.run_batch(settings_list, n_workers=4)
            >>> successful = [r for r in results if r.error is None]
            >>> failed = [r for r in results if r.error is not None]
        """
        use_parallel = n_workers is not None and n_workers > 1 and len(settings_list) > 1 and all(s.agents.oracle.oracle_type != OracleType.HISTORICAL for s in settings_list)

        if use_parallel:
            return self._run_batch_parallel(settings_list, n_workers=cast(int, n_workers), strategy_spec=strategy_spec)

        # Sequential fallback
        results = []

        for settings in settings_list:
            result = self.run_simulation(settings, strategy_spec=strategy_spec)
            results.append(result)

            if fail_fast and result.error is not None:
                raise result.error

        return results

    def _run_batch_parallel(
        self,
        settings_list: list[SimulationSettings],
        n_workers: int,
        strategy_spec: StrategySpec | None = None,
    ) -> list[SimulationResult]:
        """Run simulations in parallel via hasufel's ``run_batch()``.

        Builds a ``SimulationConfig`` from each settings object and delegates
        to hasufel for multi-process execution.  Results are wrapped in
        :class:`HasufelOutput` to satisfy the ``SimulationOutput`` interface.

        When *strategy_spec* is provided, each config includes the
        ``rohan_strategy`` agent type and a ``worker_initializer`` ensures
        the agent registration fires in each worker process.
        """
        from abides_markets.simulation import ResultProfile
        from abides_markets.simulation import run_batch as hasufel_run_batch

        from rohan.simulation.abides_impl.config_builder import create_simulation_builder
        from rohan.simulation.abides_impl.hasufel_output import HasufelOutput

        start_time = time.time()
        contexts = [SimulationContext(settings=s) for s in settings_list]

        try:
            configs = [create_simulation_builder(s, strategy_spec=strategy_spec).build() for s in settings_list]
        except Exception as e:
            # If any config build fails, return errors for all
            duration = time.time() - start_time
            return [SimulationResult(context=ctx, duration_seconds=duration, error=e) for ctx in contexts]

        # When strategy_spec is used, workers need the rohan_strategy
        # agent type registered in their process.
        worker_init = None
        if strategy_spec is not None:

            def _register_rohan_agents() -> None:
                import rohan.simulation.abides_impl.strategic_agent_config  # noqa: F401

            worker_init = _register_rohan_agents

        try:
            hasufel_results = hasufel_run_batch(
                configs,
                n_workers=n_workers,
                profile=ResultProfile.FULL,
                worker_initializer=worker_init,
            )
        except Exception as e:
            duration = time.time() - start_time
            return [SimulationResult(context=ctx, duration_seconds=duration, error=e) for ctx in contexts]

        duration = time.time() - start_time
        results: list[SimulationResult] = []

        for ctx, hr, settings in zip(contexts, hasufel_results, settings_list, strict=True):
            try:
                # Discover strategic agent ID from result
                strategic_agent_id: int | None = None
                if strategy_spec is not None:
                    strategy_agents = hr.get_agents_by_category("strategy")
                    if strategy_agents:
                        strategic_agent_id = strategy_agents[0].agent_id

                output: SimulationOutput = HasufelOutput(hr, ticker=settings.ticker, strategic_agent_id=strategic_agent_id)
                results.append(
                    SimulationResult(
                        context=ctx,
                        duration_seconds=duration / len(settings_list),
                        result=output,
                    )
                )
            except Exception as e:
                results.append(SimulationResult(context=ctx, duration_seconds=duration / len(settings_list), error=e))

        return results

    def _create_runner(
        self,
        settings: SimulationSettings,
        strategy_spec: StrategySpec | None = None,
        strategy: StrategicAgent | None = None,
    ) -> SimulationRunner:
        """Factory method to create the appropriate simulation runner.

        Args:
            settings: Simulation configuration.
            strategy_spec: Optional serializable StrategySpec (preferred).
            strategy: Optional pre-built StrategicAgent instance (legacy).

        Returns:
            SimulationRunner instance for the specified engine.

        Raises:
            ValueError: If the simulation engine is not supported.
        """
        if settings.engine == SimulationEngine.ABIDES:
            from rohan.simulation.abides_impl import SimulationRunnerAbides

            return SimulationRunnerAbides(settings, strategy_spec=strategy_spec, strategy=strategy)

        raise ValueError(f"Unsupported simulation engine: {settings.engine}")

    @classmethod
    def clear_baseline_cache(cls) -> None:
        """Clear all baseline cache entries."""
        cls._baseline_cache.clear()

    @staticmethod
    def _build_baseline_cache_key(settings: SimulationSettings) -> str:
        """Build deterministic cache key for baseline runs."""
        payload = settings.model_dump(mode="json")
        payload.pop("baseline_cache_enabled", None)
        payload.pop("baseline_cache_max_entries", None)
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
