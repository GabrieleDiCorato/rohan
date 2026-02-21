import pandas as pd

from rohan.config import SimulationSettings
from rohan.simulation import SimulationService


def run_simulation(seed):
    config = SimulationSettings()
    config.seed = seed
    # Use a 5 minute duration for testing
    config.start_time = "09:30:00"
    config.end_time = "09:35:00"
    config.date = "20260130"
    config.stdout_log_level = "OFF"  # Reduce noise

    # Run the simulation
    simulation_service = SimulationService()
    result = simulation_service.run_simulation(config)

    # Raise error if simulation failed
    if result.error is not None:
        raise result.error

    return result.result


def test_simulation_reproducibility():
    """Test that two simulations with the same seed produce identical results."""
    seed = 42

    # Run first simulation
    simulation1 = run_simulation(seed)

    # Run second simulation
    simulation2 = run_simulation(seed)

    # Compare L1
    assert simulation1 is not None
    assert simulation2 is not None
    l1_sim1 = simulation1.get_order_book_l1()
    l1_sim2 = simulation2.get_order_book_l1()

    pd.testing.assert_frame_equal(l1_sim1, l1_sim2)

    # Compare L2
    l2_sim1 = simulation1.get_order_book_l2(n_levels=2)
    l2_sim2 = simulation2.get_order_book_l2(n_levels=2)

    pd.testing.assert_frame_equal(l2_sim1, l2_sim2)
