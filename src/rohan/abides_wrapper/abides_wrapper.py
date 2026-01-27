import numpy as np
import pandas as pd
from abides_core import LatencyModel
from abides_core.utils import get_wake_time, str_to_ns
from abides_markets.agents import (
    AdaptiveMarketMakerAgent,
    ExchangeAgent,
    MomentumAgent,
    NoiseAgent,
    ValueAgent,
)
from abides_markets.agents.financial_agent import FinancialAgent
from abides_markets.models import OrderSizeModel
from abides_markets.oracles import SparseMeanRevertingOracle
from abides_markets.utils import generate_uniform_random_pairwise_dist_on_line, meters_to_light_ns

from rohan.abides_wrapper.random_state_handler import RandomStateHandler
from rohan.config.agent_settings import AdaptiveMarketMakerSettings, AgentSettings, ExchangeAgentSettings, ValueAgentSettings
from rohan.config.latency_settings import LatencyModelSettings, LatencyType
from rohan.config.simulation_settings import SimulationSettings


class AbidesWrapper:
    """Wrapper for ABIDES simulation components.
    Handles the conversion of configuration settings to ABIDES-compatible objects.
    """

    def __init__(self, simulation_settings: SimulationSettings):
        self.simulation_settings = simulation_settings
        self.random_state_handler = RandomStateHandler(simulation_settings.seed)

    def build_configuration(self):
        settings: SimulationSettings = self.simulation_settings
        random_state_handler: RandomStateHandler = self.random_state_handler

        # Calculate time-related parameters
        date = pd.to_datetime(settings.date).value
        kernel_start_time = date
        mkt_open = date + str_to_ns(settings.start_time)
        mkt_close = date + str_to_ns(settings.end_time)
        noise_mkt_open = mkt_open - str_to_ns("00:30:00")
        noise_mkt_close = date + str_to_ns("16:00:00")
        kernel_stop_time = mkt_close + str_to_ns("1s")

        agents = self._build_agents(settings, mkt_open, mkt_close, noise_mkt_open, noise_mkt_close, random_state_handler)
        n_agents = len(agents)
        oracle = self._build_oracle(settings, mkt_open, noise_mkt_close, random_state_handler)
        latency_model = self._build_latency_model(n_agents, settings.latency, random_state_handler)

        return {
            "seed": settings.seed,
            "start_time": kernel_start_time,
            "stop_time": kernel_stop_time,
            "agents": agents,
            "agent_latency_model": latency_model,
            "default_computation_delay": settings.computation_delay_ns,
            "custom_properties": {"oracle": oracle},
            "random_state_kernel": random_state_handler.random_state_kernel,
            "stdout_log_level": settings.stdout_log_level,
        }

    @staticmethod
    def _build_agents(
        simulation_settings: SimulationSettings,
        mkt_open: int,
        mkt_close: int,
        noise_mkt_open: int,
        noise_mkt_close: int,
        random_state_handler: RandomStateHandler,
    ) -> list[FinancialAgent]:
        """Uses AgentSettings to create a list of ABIDES FinancialAgent instances.

        Arguments:
        simulation_settings: SimulationSettings instance containing configuration.
        mkt_open: Market open time in nanoseconds.
        mkt_close: Market close time in nanoseconds.
        noise_mkt_open: Noise agent market open time in nanoseconds.
        noise_mkt_close: Noise agent market close time in nanoseconds.
        random_state_handler: RandomStateHandler instance for managing random states.
        """

        agent_settings: AgentSettings = simulation_settings.agents
        agents: list[FinancialAgent] = []
        agent_count = 0

        # Get simulation-level settings
        ticker = simulation_settings.ticker
        starting_cash = simulation_settings.starting_cash
        log_orders = simulation_settings.log_orders

        # Create order size model (shared by trading agents)
        order_size_model = OrderSizeModel()

        # 1) Exchange Agent
        exchange_settings: ExchangeAgentSettings = agent_settings.exchange
        agents.append(
            ExchangeAgent(
                id=agent_count,
                name="EXCHANGE_AGENT",
                type=exchange_settings.type.value,
                mkt_open=mkt_open,
                mkt_close=mkt_close,
                symbols=[ticker],
                book_logging=exchange_settings.book_logging,
                book_log_depth=exchange_settings.book_log_depth,
                log_orders=exchange_settings.exchange_log_orders,
                pipeline_delay=exchange_settings.pipeline_delay_ns,
                computation_delay=exchange_settings.computation_delay_ns,
                stream_history=exchange_settings.stream_history_length,
                random_state=random_state_handler.exchange_agent_state,
            )
        )
        agent_count += 1

        # 2) Noise Agents
        agents.extend(
            [
                NoiseAgent(
                    id=j,
                    name=f"NoiseAgent {j}",
                    type=agent_settings.noise.type.value,
                    symbol=ticker,
                    starting_cash=starting_cash,
                    wakeup_time=get_wake_time(noise_mkt_open, noise_mkt_close),
                    log_orders=log_orders,
                    order_size_model=order_size_model,
                    random_state=random_state_handler.get_random_state(),  # Each noise agent gets its own random state
                )
                for j in range(agent_count, agent_count + agent_settings.noise.num_agents)
            ]
        )
        agent_count += agent_settings.noise.num_agents

        # 3) Value Agents
        value_agent_settings: ValueAgentSettings = agent_settings.value
        sigma_n = value_agent_settings.r_bar / 100  # observation noise variance

        agents.extend(
            [
                ValueAgent(
                    id=j,
                    name=f"Value Agent {j}",
                    type=value_agent_settings.type.value,
                    symbol=ticker,
                    starting_cash=starting_cash,
                    sigma_n=sigma_n,
                    r_bar=value_agent_settings.r_bar,
                    kappa=value_agent_settings.kappa,
                    lambda_a=value_agent_settings.lambda_a,
                    log_orders=log_orders,
                    order_size_model=order_size_model,
                    random_state=random_state_handler.get_random_state(),
                )
                for j in range(agent_count, agent_count + value_agent_settings.num_agents)
            ]
        )
        agent_count += value_agent_settings.num_agents

        # 4) Adaptive Market Maker Agents
        market_maker_settings: AdaptiveMarketMakerSettings = agent_settings.adaptive_market_maker
        mm_wake_up_freq = str_to_ns(market_maker_settings.wake_up_freq)
        agents.extend(
            [
                AdaptiveMarketMakerAgent(
                    id=j,
                    name=f"ADAPTIVE_POV_MARKET_MAKER_AGENT_{j}",
                    type=market_maker_settings.type.value,
                    symbol=ticker,
                    starting_cash=starting_cash,
                    pov=market_maker_settings.pov,
                    min_order_size=market_maker_settings.min_order_size,
                    window_size=market_maker_settings.window_size,
                    num_ticks=market_maker_settings.num_ticks,
                    wake_up_freq=mm_wake_up_freq,
                    poisson_arrival=market_maker_settings.poisson_arrival,
                    cancel_limit_delay=market_maker_settings.cancel_limit_delay_ns,
                    skew_beta=market_maker_settings.skew_beta,
                    price_skew_param=market_maker_settings.price_skew,
                    level_spacing=market_maker_settings.level_spacing,
                    spread_alpha=market_maker_settings.spread_alpha,
                    backstop_quantity=market_maker_settings.backstop_quantity,
                    log_orders=log_orders,
                    random_state=random_state_handler.get_random_state(),
                )
                for j in range(agent_count, agent_count + market_maker_settings.num_agents)
            ]
        )
        agent_count += market_maker_settings.num_agents

        # 5) Momentum Agents
        momentum_wake_up_freq = str_to_ns(agent_settings.momentum.wake_up_freq)
        agents.extend(
            [
                MomentumAgent(
                    id=j,
                    name=f"MOMENTUM_AGENT_{j}",
                    type=agent_settings.momentum.type.value,
                    symbol=ticker,
                    starting_cash=starting_cash,
                    min_size=agent_settings.momentum.min_size,
                    max_size=agent_settings.momentum.max_size,
                    wake_up_freq=momentum_wake_up_freq,
                    poisson_arrival=agent_settings.momentum.poisson_arrival,
                    log_orders=log_orders,
                    order_size_model=order_size_model,
                    random_state=random_state_handler.get_random_state(),
                )
                for j in range(agent_count, agent_count + agent_settings.momentum.num_agents)
            ]
        )
        agent_count += agent_settings.momentum.num_agents

        return agents

    @staticmethod
    def _build_oracle(
        settings: SimulationSettings,
        mkt_open: int,
        noise_mkt_close: int,
        random_state_handler: RandomStateHandler,
    ) -> SparseMeanRevertingOracle:
        agent_settings: AgentSettings = settings.agents

        symbols = {
            settings.ticker: {
                "r_bar": agent_settings.value.r_bar,
                "kappa": agent_settings.oracle.kappa,
                "sigma_s": agent_settings.oracle.sigma_s,
                "fund_vol": agent_settings.oracle.fund_vol,
                "megashock_lambda_a": agent_settings.oracle.megashock_lambda_a,
                "megashock_mean": agent_settings.oracle.megashock_mean,
                "megashock_var": agent_settings.oracle.megashock_var,
                "random_state": random_state_handler.oracle_state,
            }
        }
        return SparseMeanRevertingOracle(mkt_open, noise_mkt_close, symbols)

    @staticmethod
    def _build_latency_model(agent_count: int, latency_settings: LatencyModelSettings, random_state_handler: RandomStateHandler) -> LatencyModel:
        """Uses LatencyModelSettings to create an instance of LatencyModel from abides_core.

        Arguments:
        latency_settings: LatencyModelSettings instance containing configuration.
        """

        pairwise = (agent_count, agent_count)

        if latency_settings.type == LatencyType.NO_LATENCY:
            pairwise_latencies = np.zeros(pairwise, dtype=int)
        else:
            # All agents sit on line from Milan to Lisbon
            milan_to_lisbon_meters = 1687000
            pairwise_distances = generate_uniform_random_pairwise_dist_on_line(0.0, milan_to_lisbon_meters, agent_count, random_state=random_state_handler.latency_state)
            pairwise_latencies = meters_to_light_ns(pairwise_distances)

        return LatencyModel(
            random_state=random_state_handler.latency_state,
            min_latency=pairwise_latencies,
            latency_model=latency_settings.type,
            # Args for cubic latency model:
            connected=True,  # hardcoded like in abides_markets
            jitter=latency_settings.jitter,
            jitter_clip=latency_settings.jitter_clip,
            jitter_unit=latency_settings.jitter_unit,
        )
