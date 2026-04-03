"""Translation from Rohan SimulationSettings to hasufel SimulationConfig.

This module replaces the old AbidesConfigMapper (504 lines of manual agent
instantiation) with a thin translation layer that maps SimulationSettings
fields onto the hasufel SimulationBuilder API.  Agent instantiation, seed
derivation, and latency model construction are all handled by hasufel's
``compile()`` step.

Usage::

    builder = create_simulation_builder(settings)
    config = builder.build()

    # Or for pre-built scenario templates:
    runtime = compile_template("stable_day", seed=42)
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from abides_core.utils import str_to_ns
from abides_markets.config_system import SimulationBuilder, list_templates
from abides_markets.oracles import ExternalDataOracle
from abides_markets.oracles.data_providers import InterpolationStrategy

from rohan.config import (
    HistoricalOracleSettings,
    InterpolationMode,
    OracleType,
    ProviderType,
    SimulationSettings,
)
from rohan.simulation.data.api_provider import ApiDataProvider
from rohan.simulation.data.csv_provider import CsvDataProvider
from rohan.simulation.data.database_provider import DatabaseDataProvider
from rohan.simulation.data.provider_protocol import (
    FundamentalDataProvider,
    LazyLinearPointAdapter,
)
from rohan.simulation.models.strategy_spec import StrategySpec


def create_simulation_builder(
    settings: SimulationSettings,
    strategy_spec: StrategySpec | None = None,
) -> SimulationBuilder:
    """Create a configured hasufel SimulationBuilder from Rohan SimulationSettings.

    The returned builder can be used to obtain either the declarative config
    (``builder.build()``) or the compiled runtime dict
    (``builder.build_and_compile()``).  Any pre-built oracle instance (for
    historical mode) is stored inside the builder and passed through
    automatically by ``build_and_compile()``.

    Parameters
    ----------
    settings:
        Rohan simulation configuration.
    strategy_spec:
        Optional serializable strategy to inject.  When provided the
        ``rohan_strategy`` agent type is enabled in the config, and
        hasufel's ``compile()`` pipeline handles instantiation via
        :class:`StrategicAgentConfig`.
    """
    builder = SimulationBuilder()

    # ── Template bootstrap (if set) ──────────────────────────────
    # When a template is specified, agent composition and oracle params
    # come from hasufel's template; we only override market timing,
    # seed, latency, and strategy injection.
    if settings.template is not None:
        builder.from_template(settings.template)
        builder.market(
            ticker=settings.ticker,
            date=settings.date,
            start_time=settings.start_time,
            end_time=settings.end_time,
        )
        builder.latency(type=settings.latency.type.value)
        builder.computation_delay(settings.computation_delay_ns)
        builder.seed(settings.seed)
        builder.log_orders(settings.log_orders)
        log_level = settings.stdout_log_level
        if log_level.upper() == "OFF":
            log_level = "CRITICAL"
        builder.log_level(log_level)

        if strategy_spec is not None:
            import rohan.simulation.abides_impl.strategic_agent_config  # noqa: F401

            builder.enable_agent(
                "rohan_strategy",
                count=1,
                strategy_spec=strategy_spec,
                starting_cash=settings.starting_cash,
            )

        return builder

    # ── Custom config (no template) ──────────────────────────────

    # ── Market ────────────────────────────────────────────────────
    builder.market(
        ticker=settings.ticker,
        date=settings.date,
        start_time=settings.start_time,
        end_time=settings.end_time,
    )

    # ── Exchange ──────────────────────────────────────────────────
    exc = settings.agents.exchange
    builder.exchange(
        book_logging=exc.book_logging,
        book_log_depth=exc.book_log_depth,
        stream_history_length=exc.stream_history_length,
        log_orders=exc.exchange_log_orders,
        pipeline_delay=exc.pipeline_delay_ns,
        computation_delay=exc.computation_delay_ns,
    )

    # ── Oracle ────────────────────────────────────────────────────
    oracle_settings = settings.agents.oracle
    if oracle_settings.oracle_type == OracleType.HISTORICAL:
        oracle_instance = _build_historical_oracle(settings)
        builder.oracle_instance(oracle_instance)
    else:
        builder.oracle(
            type="sparse_mean_reverting",
            r_bar=settings.agents.value.r_bar,
            kappa=oracle_settings.kappa,
            sigma_s=oracle_settings.sigma_s,
            fund_vol=oracle_settings.fund_vol,
            megashock_lambda_a=oracle_settings.megashock_lambda_a,
            megashock_mean=oracle_settings.megashock_mean,
            megashock_var=oracle_settings.megashock_var,
        )

    # ── Background agents ─────────────────────────────────────────
    _add_agents(builder, settings)

    # ── Strategic agent (optional) ────────────────────────────────
    if strategy_spec is not None:
        # Ensure registration has fired (idempotent via allow_overwrite=True)
        import rohan.simulation.abides_impl.strategic_agent_config  # noqa: F401

        builder.enable_agent(
            "rohan_strategy",
            count=1,
            strategy_spec=strategy_spec,
            starting_cash=settings.starting_cash,
        )

    # ── Infrastructure ────────────────────────────────────────────
    builder.latency(type=settings.latency.type.value)
    builder.computation_delay(settings.computation_delay_ns)

    # ── Simulation metadata ───────────────────────────────────────
    builder.seed(settings.seed)
    builder.log_orders(settings.log_orders)
    # hasufel accepts DEBUG/INFO/WARNING/ERROR/CRITICAL; map "OFF" → "CRITICAL"
    log_level = settings.stdout_log_level
    if log_level.upper() == "OFF":
        log_level = "CRITICAL"
    builder.log_level(log_level)

    return builder


def compile_template(name: str, seed: int = 42, **overrides: Any) -> dict[str, Any]:
    """Compile one of hasufel's built-in scenario templates into a runtime dict.

    *name* must be a registered template (see ``available_templates()``).
    Optional *overrides* are applied after loading the template — for example
    ``compile_template("stable_day", seed=99, ticker="XYZ")``.
    """
    builder = SimulationBuilder()
    builder.from_template(name)
    builder.seed(seed)
    for key, value in overrides.items():
        getattr(builder, key)(value)
    return builder.build_and_compile()


def available_templates() -> list[dict[str, Any]]:
    """Return metadata for all registered hasufel templates."""
    return list_templates()


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------


def _add_agents(builder: SimulationBuilder, settings: SimulationSettings) -> None:
    """Add all background agent groups to the builder."""
    agents = settings.agents

    # Noise — set the wakeup window end to the simulation end time so agents
    # actually wake up during short simulations (default "16:00:00" is only
    # appropriate for full-day runs).
    builder.enable_agent(
        "noise",
        count=agents.noise.num_agents,
        noise_mkt_close_time=settings.end_time,
    )

    # Value — for synthetic oracle, let sigma_s auto-inherit from the oracle
    # context (the compiler converts fund_vol² → sigma_s).  For historical
    # oracle (ExternalDataOracle) the context has no sigma_s, so pass it
    # explicitly using the same fund_vol² convention.
    value_kwargs: dict[str, Any] = {
        "count": agents.value.num_agents,
        "r_bar": agents.value.r_bar,
        "kappa": agents.value.kappa,
        "lambda_a": agents.value.lambda_a,
    }
    if settings.agents.oracle.oracle_type == OracleType.HISTORICAL:
        value_kwargs["sigma_s"] = settings.agents.oracle.fund_vol**2
    builder.enable_agent("value", **value_kwargs)

    # Adaptive Market Maker
    mm = agents.adaptive_market_maker
    builder.enable_agent(
        "adaptive_market_maker",
        count=mm.num_agents,
        pov=mm.pov,
        min_order_size=mm.min_order_size,
        window_size=mm.window_size,
        num_ticks=mm.num_ticks,
        wake_up_freq=mm.wake_up_freq,
        poisson_arrival=mm.poisson_arrival,
        skew_beta=mm.skew_beta,
        price_skew_param=mm.price_skew,
        level_spacing=mm.level_spacing,
        spread_alpha=mm.spread_alpha,
        backstop_quantity=mm.backstop_quantity,
        cancel_limit_delay=mm.cancel_limit_delay_ns,
    )

    # Momentum
    mom = agents.momentum
    builder.enable_agent(
        "momentum",
        count=mom.num_agents,
        min_size=mom.min_size,
        max_size=mom.max_size,
        wake_up_freq=mom.wake_up_freq,
        poisson_arrival=mom.poisson_arrival,
    )


# ---------------------------------------------------------------------------
# Oracle helpers
# ---------------------------------------------------------------------------


def _build_historical_oracle(settings: SimulationSettings) -> ExternalDataOracle:
    """Build an ExternalDataOracle from historical settings."""
    hist = settings.agents.oracle.historical
    date_ns = int(pd.to_datetime(settings.date).value)
    mkt_open = date_ns + str_to_ns(settings.start_time)
    noise_mkt_close = date_ns + str_to_ns("16:00:00")

    r_bar = settings.agents.value.r_bar if hist.recenter_r_bar else None
    provider = _resolve_historical_provider(settings, hist, r_bar)

    interpolation_map = {
        InterpolationMode.FORWARD_FILL: InterpolationStrategy.FORWARD_FILL,
        InterpolationMode.NEAREST: InterpolationStrategy.NEAREST,
        InterpolationMode.LINEAR: InterpolationStrategy.LINEAR,
    }
    interpolation = interpolation_map[hist.interpolation]

    if hist.interpolation == InterpolationMode.LINEAR:
        provider = LazyLinearPointAdapter(provider, settings.ticker, mkt_open, noise_mkt_close)
        interpolation = InterpolationStrategy.FORWARD_FILL

    return ExternalDataOracle(
        mkt_open=mkt_open,
        mkt_close=noise_mkt_close,
        symbols=[settings.ticker],
        provider=provider,
        interpolation=interpolation,
    )


def _resolve_historical_provider(
    settings: SimulationSettings,
    hist: HistoricalOracleSettings,
    r_bar: int | None,
) -> FundamentalDataProvider:
    """Resolve the data provider for historical oracle mode."""
    if hist.provider_type == ProviderType.CSV:
        csv_cfg = hist.csv
        if not csv_cfg.csv_path:
            msg = "historical.csv.csv_path is required when provider_type=CSV"
            raise ValueError(msg)
        return CsvDataProvider(
            path=csv_cfg.csv_path,
            symbol=settings.ticker,
            r_bar=r_bar,
            price_unit=csv_cfg.price_unit,
            source_timezone=csv_cfg.source_timezone,
        )
    if hist.provider_type == ProviderType.DATABASE:
        db_cfg = hist.database
        if not db_cfg.dataset_id:
            msg = "historical.database.dataset_id is required when provider_type=DATABASE"
            raise ValueError(msg)
        return DatabaseDataProvider(
            dataset_id=db_cfg.dataset_id,
            symbol=settings.ticker,
            r_bar=r_bar,
        )
    if hist.provider_type == ProviderType.API:
        api_cfg = hist.api
        return ApiDataProvider(
            provider_name=api_cfg.provider_name,
            symbol=api_cfg.symbol or settings.ticker,
            api_key=api_cfg.api_key or "",
            price_unit=api_cfg.price_unit,
            source_timezone=api_cfg.source_timezone,
            r_bar=r_bar,
        )
    msg = f"Unsupported historical provider type: {hist.provider_type}"
    raise ValueError(msg)
