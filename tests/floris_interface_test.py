
from pathlib import Path

import numpy as np

from floris.tools.floris_interface import FlorisInterface


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full_v3.yaml"


def test_read_yaml():
    fi = FlorisInterface(configuration=YAML_INPUT)
    assert isinstance(fi, FlorisInterface)


def test_calculate_wake():

    """
    In FLORIS v3.2, running calculate_wake twice incorrectly set the yaw angles when the first time
    has non-zero yaw settings but the second run had all-zero yaw settings. The test below asserts
    that the yaw angles are correctly set in subsequent calls to calculate_wake.
    """
    fi = FlorisInterface(configuration=YAML_INPUT)
    yaw_angles = 20 * np.ones(
        (
            fi.floris.flow_field.n_wind_directions,
            fi.floris.flow_field.n_wind_speeds,
            fi.floris.farm.n_turbines
        )
    )
    fi.calculate_wake(yaw_angles=yaw_angles)
    assert np.array_equal(fi.floris.farm.yaw_angles, yaw_angles)

    yaw_angles = np.zeros(
        (
            fi.floris.flow_field.n_wind_directions,
            fi.floris.flow_field.n_wind_speeds,
            fi.floris.farm.n_turbines
        )
    )
    fi.calculate_wake(yaw_angles=yaw_angles)
    assert np.array_equal(fi.floris.farm.yaw_angles, yaw_angles)


def test_calculate_no_wake():
    """
    In FLORIS v3.2, running calculate_no_wake twice incorrectly set the yaw angles when the first
    time has non-zero yaw settings but the second run had all-zero yaw settings. The test below
    asserts that the yaw angles are correctly set in subsequent calls to calculate_no_wake.
    """
    fi = FlorisInterface(configuration=YAML_INPUT)
    yaw_angles = 20 * np.ones(
        (
            fi.floris.flow_field.n_wind_directions,
            fi.floris.flow_field.n_wind_speeds,
            fi.floris.farm.n_turbines
        )
    )
    fi.calculate_no_wake(yaw_angles=yaw_angles)
    assert np.array_equal(fi.floris.farm.yaw_angles, yaw_angles)

    yaw_angles = np.zeros(
        (
            fi.floris.flow_field.n_wind_directions,
            fi.floris.flow_field.n_wind_speeds,
            fi.floris.farm.n_turbines
        )
    )
    fi.calculate_no_wake(yaw_angles=yaw_angles)
    assert np.array_equal(fi.floris.farm.yaw_angles, yaw_angles)


def test_reset():
    """
    Test that the reset function correctly resets the given arguments, and that
    it correctly resets the low level FLORIS objects.

    When we change the value of the wind speed, FlowField should reinitialize the velocity
    arrays with the updated shear profile.
    When we change the value of the wind direction, Grid should sort the points, as needed,
    and reset the sorted_indeces that are used in Farm. Farm should also reset all of it's
    sorted and unsorted data arrays accordingly.

    When we change the size of either of these, all data arrays should be resized based on
    the number of wind speeds and wind directions.
    """

    # First, test that after resetting the wind speed and direction and layout, the calculation
    # results haven't changed.
    fi = FlorisInterface(configuration=YAML_INPUT)
    fi.calculate_wake()
    baseline_turbine_powers = fi.get_turbine_powers()

    fi.reset(
        layout_x=fi.floris.farm.layout_x,
        layout_y=fi.floris.farm.layout_y,
        wind_directions=fi.floris.flow_field.wind_directions,
        wind_speeds=fi.floris.flow_field.wind_speeds,
    )
    fi.calculate_wake()
    test_turbine_powers = fi.get_turbine_powers()

    assert np.array_equal(baseline_turbine_powers, test_turbine_powers)

    # Since the test layout is three turbines in a line, rotating 180 degrees should produce
    # the same farm power and flipped turbine powers
    fi.reset(
        layout_x=fi.floris.farm.layout_x,
        layout_y=fi.floris.farm.layout_y,
        wind_directions=fi.floris.flow_field.wind_directions + 180,
        wind_speeds=fi.floris.flow_field.wind_speeds,
    )
    fi.calculate_wake()
    rotated_turbine_powers = fi.get_turbine_powers()
    assert np.array_equal(
        baseline_turbine_powers,
        rotated_turbine_powers[:,:,::-1]  # turbine dimension is flipped below
    )

    # At 90 degree rotation, all turbines are unwaked so all turbine powers should be equal
    # to the upstream turbine in the baseline array
    fi.reset(
        layout_x=fi.floris.farm.layout_x,
        layout_y=fi.floris.farm.layout_y,
        wind_directions=fi.floris.flow_field.wind_directions + 90,
        wind_speeds=fi.floris.flow_field.wind_speeds,
    )
    fi.calculate_wake()
    unwaked_turbine_powers = fi.get_turbine_powers()
    n_turbines = fi.floris.farm.n_turbines
    assert np.array_equal(
        np.repeat(baseline_turbine_powers[:,:,0:1], n_turbines, axis=2),
        unwaked_turbine_powers
    )
