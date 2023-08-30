
import copy
import time
import numpy as np
import memory_profiler
from floris.simulation import Floris


def run_floris(input_dict: dict) -> float:

    # Deep copy the input dictionary to ensure no side effects
    input_dict = copy.deepcopy(input_dict)

    # Start timing
    start = time.perf_counter()

    # Run floris calculation
    floris = Floris.from_dict(input_dict)
    floris.initialize_domain()
    floris.steady_state_atmospheric_condition()

    # Stop timing
    end = time.perf_counter()

    return end - start


def time_profile(input_dict, N_ITERATIONS=30) -> float:
    """
    Runs the wake calculation for N_ITERATIONS and returns
    the average of the timing results.
    An initial iteration of the calculation is run and not included in the
    timing results in order to allow Python to compile what it needs to and
    initialize the computer's memory.
    """

    # Run once to initialize Python and memory
    run_floris(input_dict)

    # Get timing for each iteration
    times = np.zeros(N_ITERATIONS)
    for i in range(N_ITERATIONS):
        times[i] = run_floris(input_dict)

    # Return the average
    return np.mean(times)


def memory_profile(input_dict):
    floris = Floris.from_dict(copy.deepcopy(input_dict["floris"]))
    floris.initialize_domain()  # TODO: include this in memory_usage()?
    mem_usage = memory_profiler.memory_usage(
        (floris.steady_state_atmospheric_condition, (), {}),
        max_usage=True
    )
    return mem_usage
