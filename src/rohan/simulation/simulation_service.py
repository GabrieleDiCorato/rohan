"""Service module for simulation-related operations. Injects dependencies and manages simulation execution."""

import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import cast

from rohan.config import SimulationEngine, SimulationSettings, get_feature_flags
from rohan.llm.telemetry import emit_metric
from rohan.simulation.models import (
    SimulationContext,
    SimulationResult,
)
from rohan.simulation.models.simulation_output import SimulationOutput
from rohan.simulation.models.strategy_api import StrategicAgent
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

    def run_simulation(
        self,
        settings: SimulationSettings,
        context: SimulationContext | None = None,
        strategy: StrategicAgent | None = None,
    ) -> SimulationResult:
        """Runs a single simulation with the given settings.

        Args:
            settings: Configuration for the simulation.
            context: Optional context for tracking metadata. If not provided, one will be created.
            strategy: Optional StrategicAgent to inject into the simulation.

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
        if strategy is None and settings.baseline_cache_enabled and feature_flags.baseline_cache_v1:
            cache_key = self._build_baseline_cache_key(settings)
            cached_output = self._baseline_cache.get(cache_key)
            if cached_output is not None:
                self._baseline_cache.move_to_end(cache_key)
                logger.info("Baseline cache hit: %s", cache_key[:12])
                if feature_flags.llm_telemetry_v1:
                    emit_metric("baseline_cache_hit", cache_key=cache_key[:12])
                return SimulationResult(
                    context=context,
                    duration_seconds=0.0,
                    result=cast(SimulationOutput, cached_output),
                )

        start_time = time.time()

        try:
            runner = self._create_runner(settings, strategy=strategy)

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
                        emit_metric("baseline_cache_store", cache_key=cache_key[:12], cache_size=len(self._baseline_cache))

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

    def run_batch(self, settings_list: list[SimulationSettings], fail_fast: bool = False) -> list[SimulationResult]:
        """Run multiple simulations sequentially with different settings.

        Args:
            settings_list: List of simulation configurations to run.
            fail_fast: If True, stop on first failure. If False, continue and collect all results.

        Returns:
            List of SimulationResult objects containing outputs and metadata.

        Example:
            >>> service = SimulationService()
            >>> settings_list = [SimulationSettings(seed=i) for i in range(10)]
            >>> results = service.run_batch(settings_list)
            >>> successful = [r for r in results if r.error is None]
            >>> failed = [r for r in results if r.error is not None]
        """
        results = []

        for settings in settings_list:
            # Each simulation gets its own context
            result = self.run_simulation(settings)
            results.append(result)

            # If fail_fast is enabled and simulation failed, raise the error
            if fail_fast and result.error is not None:
                raise result.error

        return results

    def _create_runner(self, settings: SimulationSettings, strategy: StrategicAgent | None = None) -> SimulationRunner:
        """Factory method to create the appropriate simulation runner.

        Args:
            settings: Simulation configuration.
            strategy: Optional StrategicAgent to inject into the simulation.

        Returns:
            SimulationRunner instance for the specified engine.

        Raises:
            ValueError: If the simulation engine is not supported.
        """
        if settings.engine == SimulationEngine.ABIDES:
            from rohan.simulation.abides_impl import SimulationRunnerAbides

            return SimulationRunnerAbides(settings, strategy=strategy)

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
