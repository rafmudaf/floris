

import copy
import time
from floris.simulation import Floris
from linux_perf import perf

def run_floris(input_dict):
    try:
        start = time.perf_counter()
        floris = Floris.from_dict(copy.deepcopy(input_dict.floris))
        floris.initialize_domain()
        floris.steady_state_atmospheric_condition()
        end = time.perf_counter()
        return end - start
    except KeyError:
        # Catch the errors when an invalid wake model was given because the model
        # was not yet implemented
        return -1.0

# def time_profile(input_dict):
#     """
#     Profile only the solver step
#     """
#     floris = Floris.from_dict(input_dict.floris)
#     floris.initialize_domain()
#     start = time.perf_counter()
#     floris.steady_state_atmospheric_condition()
#     end = time.perf_counter()
#     return end - start

def time_profile(input_dict):

    # Run once to initialize Python and memory
    run_floris(input_dict)

    times = np.zeros(N_ITERATIONS)
    for i in range(N_ITERATIONS):
        times[i] = run_floris(input_dict)

    return np.sum(times) / N_ITERATIONS

# def internal_probe(input_dict):
#     floris = Floris(input_dict=input_dict.floris)
#     internal_quantity = floris.steady_state_atmospheric_condition()
#     return internal_quantity


# def memory_profile(input_dict):
#     floris = Floris(input_dict=input_dict.floris)
#     mem_usage = memory_profiler.memory_usage(
#         (floris.steady_state_atmospheric_condition, (), {}),
#         max_usage=True
#     )
#     return mem_usage


def memory_profile(input_dict):
    # Run once to initialize Python and memory
    floris = Floris.from_dict(copy.deepcopy(input_dict.floris))
    floris.initialize_domain()
    floris.steady_state_atmospheric_condition()

    with perf():
        for i in range(N_ITERATIONS):
            floris = Floris.from_dict(copy.deepcopy(input_dict.floris))
            floris.initialize_domain()
            floris.steady_state_atmospheric_condition()

    print(
        "Size of one data array: "
        f"{64 * N_WIND_DIRECTIONS * N_WIND_SPEEDS * N_TURBINES * 25 / (1000 * 1000)} MB"
    )

