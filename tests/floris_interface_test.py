from pathlib import Path

import numpy as np
import pytest

from floris.tools.floris_interface import FlorisInterface


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full_v3.yaml"
JSON_INPUT = TEST_DATA / "input_full_v3.json"


def test_read_json():
    fi = FlorisInterface(configuration=JSON_INPUT)
    assert isinstance(fi, FlorisInterface)


def test_read_yaml():
    fi = FlorisInterface(configuration=YAML_INPUT)
    assert isinstance(fi, FlorisInterface)


def test_calculate_wake():
    pass


def test_reinitialize_flow_field():
    correct_layout_x = np.arange(10, dtype=float) * 150
    correct_layout_y = np.arange(10, dtype=float) * 200
    correct_wind_speeds = np.array([5.0, 10.0, 15.0, 20.0])
    correct_wind_directions = np.array([20.0, 40.0, 60.0, 80.0, 100.0])
    correct_turbine_id = ["nrel_5mw"] * correct_layout_x.shape[0]
    incorrect_turbine_id = ["nrel_5mw"]  # The original shape should fail
    correct_wtg_id = [f"WTG-{i + 1}" for i in range(correct_layout_x.shape[0])]
    incorrect_wtg_id = ["WTG-{1}"]
    correct_wind_shear = 0.13
    correct_wind_veer = 0.1
    correct_reference_height = 85.0
    correct_air_density = 1.4

    # Test that initialization works, specifically with no turbine_id or wtg_id provided
    #  for a single turbine type model
    fi = FlorisInterface(configuration=YAML_INPUT)
    fi.reinitialize_flow_field(
        wind_speed=correct_wind_speeds,
        wind_direction=correct_wind_directions,
        wind_shear=correct_wind_shear,
        wind_veer=correct_wind_veer,
        air_density=correct_air_density,
        specified_wind_height=correct_reference_height,
        layout_array=np.vstack((correct_layout_x, correct_layout_y)),
    )
    np.testing.assert_equal(fi.floris.farm.layout_x, correct_layout_x)
    np.testing.assert_equal(fi.floris.farm.layout_y, correct_layout_y)
    np.testing.assert_equal(fi.floris.farm.turbine_id, correct_turbine_id)
    np.testing.assert_equal(fi.floris.flow_field.wind_speeds, correct_wind_speeds)
    np.testing.assert_equal(fi.floris.flow_field.wind_directions, correct_wind_directions)
    assert fi.floris.flow_field.wind_shear == correct_wind_shear
    assert fi.floris.flow_field.wind_veer == correct_wind_veer
    assert fi.floris.flow_field.reference_wind_height == correct_reference_height
    assert fi.floris.flow_field.air_density == correct_air_density

    # Test that initialization works with assigning the turbine_id and wtg_id
    fi = FlorisInterface(configuration=YAML_INPUT)
    fi.reinitialize_flow_field(
        wind_speed=correct_wind_speeds,
        wind_direction=correct_wind_directions,
        wind_shear=correct_wind_shear,
        wind_veer=correct_wind_veer,
        specified_wind_height=correct_reference_height,
        layout_array=np.vstack((correct_layout_x, correct_layout_y)),
        turbine_id=correct_turbine_id,
        wtg_id=correct_wtg_id,
    )
    np.testing.assert_equal(fi.floris.farm.wtg_id, correct_wtg_id)
    np.testing.assert_equal(fi.floris.farm.turbine_id, correct_turbine_id)

    # Test that an incorrect turbine_id value will fail
    fi = FlorisInterface(configuration=YAML_INPUT)
    with pytest.raises(ValueError):
        fi.reinitialize_flow_field(
            wind_speed=correct_wind_speeds,
            wind_direction=correct_wind_directions,
            wind_shear=correct_wind_shear,
            wind_veer=correct_wind_veer,
            specified_wind_height=correct_reference_height,
            layout_array=np.vstack((correct_layout_x, correct_layout_y)),
            turbine_id=incorrect_turbine_id,
        )

    # Test that an incorrect wtg_id value will fail
    fi = FlorisInterface(configuration=YAML_INPUT)
    with pytest.raises(ValueError):
        fi.reinitialize_flow_field(
            wind_speed=correct_wind_speeds,
            wind_direction=correct_wind_directions,
            wind_shear=correct_wind_shear,
            wind_veer=correct_wind_veer,
            specified_wind_height=correct_reference_height,
            layout_array=np.vstack((correct_layout_x, correct_layout_y)),
            wtg_id=incorrect_wtg_id,
        )

    # TODO: Need a check for the multi turbine type model once fully implemented


def test_get_plane_of_points():
    fi = FlorisInterface(configuration=JSON_INPUT)
    with pytest.raises(NotImplementedError):
        fi.get_plane_of_points()


def test_get_set_of_points():
    fi = FlorisInterface(configuration=JSON_INPUT)
    with pytest.raises(NotImplementedError):
        fi.get_set_of_points([0, 1], [0, 1], [0, 1])


def test_get_hor_plane():
    pass


def test_get_cross_plane():
    pass


def test_get_y_plane():
    pass


def test_get_flow_data():
    pass


def test_get_yaw_angles():
    pass


def test_get_farm_power():
    pass


def test_get_turbine_layout():
    pass


def test_get_power_curve():
    pass


def test_get_turbine_ct():
    pass


def test_get_turbine_ti():
    pass


def test_get_farm_power_for_yaw_angle():
    pass


def test_get_farm_AEP():
    pass


def test_calc_one_AEP_case():
    pass


def test_get_farm_AEP_parallel():
    pass


def test_calc_AEP_wind_limit():
    pass


def test_calc_change_turbine():
    # Update single attribute: rotor diameter and not the reference wind height
    fi = FlorisInterface(configuration=YAML_INPUT)
    correct_rotor_diameter = 110.0
    turbine_ix = [0]
    turbine_map = fi.floris.farm._asdict()["turbine_map"]
    turbine_map["nrel_5mw_110m"] = turbine_map.pop("nrel_5mw")  # rename key to be unique
    turbine_map["nrel_5mw_110m"]["rotor_diameter"] = correct_rotor_diameter

    fi.change_turbine(turbine_indices=turbine_ix, new_turbine_map=turbine_map)
    turbine_to_check = fi.floris.farm.turbine_map[fi.floris.farm.turbine_id[0]]
    assert round(turbine_to_check.rotor_diameter, 10) == correct_rotor_diameter
    np.testing.assert_almost_equal(fi.floris.farm.rotor_diameter[:, :, turbine_ix].flatten(), [correct_rotor_diameter])

    # Update single attribute: hub height and the reference wind height
    fi = FlorisInterface(configuration=YAML_INPUT)
    correct_hub_height = 100.0
    turbine_ix = [0]
    turbine_map = fi.floris.farm._asdict()["turbine_map"]
    turbine_map["nrel_5mw_100m"] = turbine_map.pop("nrel_5mw")  # rename key to be unique
    turbine_map["nrel_5mw_100m"]["hub_height"] = correct_hub_height

    fi.change_turbine(turbine_indices=turbine_ix, new_turbine_map=turbine_map, update_specified_wind_height=True)
    turbine_to_check = fi.floris.farm.turbine_map[fi.floris.farm.turbine_id[0]]
    assert round(turbine_to_check.hub_height, 10) == correct_hub_height
    np.testing.assert_almost_equal(fi.floris.farm.hub_height[:, :, turbine_ix].flatten(), [correct_hub_height])
    assert fi.floris.flow_field.reference_wind_height == correct_hub_height

    # Update single attribute: hub height but not the reference wind height
    fi = FlorisInterface(configuration=YAML_INPUT)
    correct_hub_height = 100.0
    correct_reference_hub_height = 90.0

    turbine_ix = [0]
    turbine_map = fi.floris.farm._asdict()["turbine_map"]
    turbine_map["nrel_5mw_100m"] = turbine_map.pop("nrel_5mw")  # rename key to be unique
    turbine_map["nrel_5mw_100m"]["hub_height"] = correct_hub_height

    fi.change_turbine(turbine_indices=turbine_ix, new_turbine_map=turbine_map, update_specified_wind_height=False)
    turbine_to_check = fi.floris.farm.turbine_map[fi.floris.farm.turbine_id[0]]
    assert round(turbine_to_check.hub_height, 10) == correct_hub_height
    np.testing.assert_almost_equal(fi.floris.farm.hub_height[:, :, turbine_ix].flatten(), [correct_hub_height])
    assert fi.floris.flow_field.reference_wind_height == correct_reference_hub_height


def test_set_use_points_on_perimeter():
    pass


def test_set_gch():
    pass


def test_set_gch_yaw_added_recovery():
    pass


def test_set_gch_secondary_steering():
    pass


def test_layout_x():  # TODO
    pass


def test_layout_y():  # TODO
    pass


def test_TKE_to_TI():
    pass


def test_set_rotor_diameter():  # TODO
    pass


def test_show_model_parameters():  # TODO
    pass


def test_get_model_parameters():
    # TODO: Add in the turbulence model component once implemented!
    fi = FlorisInterface(configuration=JSON_INPUT)

    # Test to check that all parameters are returned
    correct_parameters = {
        "wake_deflection_parameters": {"ad": 0.0, "bd": 0.0, "kd": 0.05},
        "wake_velocity_parameters": {"we": 0.05},
    }
    parameters = fi.get_model_parameters(turbulence_model=False)
    assert parameters == correct_parameters

    # Test to check that only deflection parameters are returned
    correct_parameters = {"wake_deflection_parameters": {"ad": 0.0, "bd": 0.0, "kd": 0.05}}
    parameters = fi.get_model_parameters(turbulence_model=False, wake_velocity_model=False)
    assert parameters == correct_parameters

    # Test to check that only velocity models are returned
    correct_parameters = {"wake_velocity_parameters": {"we": 0.05}}
    parameters = fi.get_model_parameters(turbulence_model=False, wake_deflection_model=False)
    assert parameters == correct_parameters

    # Check that oly "ad" and "kd" values are returned
    correct_parameters = {"wake_deflection_parameters": {"ad": 0.0, "kd": 0.05}, "wake_velocity_parameters": {}}
    parameters = fi.get_model_parameters(turbulence_model=False, params=["ad", "kd"])
    assert parameters == correct_parameters


def test_set_model_parameters():
    # TODO: Add in the turbulence model component once implemented!
    fi = FlorisInterface(configuration=JSON_INPUT)

    correct_parameters = {
        "wake_deflection_parameters": {"ad": 0.1, "bd": 0.0, "kd": 0.05},
        "wake_velocity_parameters": {"we": 0.22},
    }
    update_parameters = {
        "Wake Deflection Parameters": {"ad": 0.1},  # Check v2 naming convention
        "wake velocity_Parameters": {"we": 0.22},  # Check mixed naming convention
    }
    fi.set_model_parameters(params=update_parameters)
    parameters = fi.get_model_parameters(turbulence_model=False)
    assert parameters == correct_parameters


def test_vis_layout():
    pass


def test_show_flow_field():
    pass
