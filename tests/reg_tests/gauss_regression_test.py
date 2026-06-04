
import numpy as np

from floris.core import (
    average_velocity,
    axial_induction,
    # Core,
    power,
    rotor_effective_velocity,
    thrust_coefficient,
)
from tests.conftest import (
    assert_results_arrays,
    N_FINDEX,
    N_TURBINES,
    print_test_values,
)


DEBUG = False
VELOCITY_MODEL = "gauss"
DEFLECTION_MODEL = "gauss"
BACKEND = "floraf"
if BACKEND == "floraf":
    from floris.core.cpp_core import CppCore as Core

baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736858, 0.7871515, 1753954.4591792, 0.2693224],
            [5.9186455, 0.8654743, 710441.9192938, 0.3166113],
            [6.0090150, 0.8604395, 741642.0177873, 0.3132110],
        ],
        # 9 m/s
        [
            [8.9703965, 0.7858774, 2496427.8618358, 0.2686331],
            [6.6606465, 0.8308044, 1034608.0101396, 0.2943330],
            [6.7947466, 0.8247058, 1094897.8563374, 0.2906592],
        ],
        # 10 m/s
        [
            [9.9671073, 0.7838789, 3417797.0050916, 0.2675559],
            [7.4045198, 0.8008441, 1405853.7207176, 0.2768656],
            [7.5868432, 0.7949439, 1511887.2179035, 0.2735844],
        ],
        # 11 m/s
        [
            [10.9638180, 0.7565157, 4519404.3072862, 0.2532794],
            [8.2046271, 0.7868643, 1924101.6501936, 0.2691669],
            [8.3491997, 0.7866780, 2032153.3223547, 0.2690660],
        ],
    ]
)

yawed_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736858, 0.7841561, 1741508.6722008, 0.2671213],
            [5.9521551, 0.8635694, 721623.6989382, 0.3153174],
            [6.0131307, 0.8602523, 743492.3616581, 0.3130858],
        ],
        # 9 m/s
        [
            [8.9703965, 0.7828869, 2480428.8963141, 0.2664440],
            [6.6982609, 0.8290938, 1051519.0079315, 0.2932960],
            [6.7996516, 0.8244827, 1097103.0727816, 0.2905261],
        ],
        # 10 m/s
        [
            [9.9671073, 0.7808960, 3395681.0032992, 0.2653854],
            [7.4461669, 0.7994645, 1429777.3846192, 0.2760940],
            [7.5922658, 0.7947730, 1515083.3259879, 0.2734901],
        ],
        # 11 m/s
        [
            [10.9638180, 0.7536370, 4488242.9153943, 0.2513413],
            [8.2481957, 0.7868081, 1956664.2629680, 0.2691365],
            [8.3531097, 0.7866729, 2035075.5955678, 0.2690633],
        ],
    ]
)

full_flow_baseline = np.array(
    [
        [
            [
                [7.88772361, 8.        , 8.10178821],
                [7.88772361, 8.        , 8.10178821],
                [7.88772361, 8.        , 8.10178821],
                [7.88772361, 8.        , 8.10178821],
                [7.88772361, 8.        , 8.10178821],
            ],
            [
                [7.88772264, 7.99999899, 8.10178721],
                [7.80183828, 7.91077933, 8.01357204],
                [4.05787708, 4.02142188, 4.16800363],
                [7.80183828, 7.91077933, 8.01357204],
                [7.88772264, 7.99999899, 8.10178721],
            ],
            [
                [7.88365433, 7.9958357 , 8.09760849],
                [7.54214774, 7.64551046, 7.74683377],
                [4.99852407, 5.0247459 , 5.13417881],
                [7.54214774, 7.64551046, 7.74683377],
                [7.88365433, 7.9958357 , 8.09760849],
            ],
            [
                [7.85066049, 7.96222083, 8.06371923],
                [7.39444624, 7.49602334, 7.5951238 ],
                [5.50716692, 5.55540288, 5.65662569],
                [7.39444624, 7.49602334, 7.5951238 ],
                [7.85066049, 7.96222083, 8.06371923],
            ],
            [
                [7.79761973, 7.90832696, 8.009239  ],
                [7.41896092, 7.52268669, 7.62030379],
                [6.98565022, 7.0811275 , 7.17523349],
                [7.41896092, 7.52268669, 7.62030379],
                [7.79761973, 7.90832696, 8.009239  ],
            ]
        ]
    ]
)


gch_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736858, 0.7841561, 1741508.6722008, 0.2671213],
            [5.9689340, 0.8626155, 727222.6050018, 0.3146730],
            [6.0360908, 0.8592082, 753814.9629960, 0.3123888],
            [6.1227167, 0.8552686, 792760.9909383, 0.3097821],
            [6.1410254, 0.8544359, 800992.3435818, 0.3092357],
            [6.1432541, 0.8543345, 801994.3564971, 0.3091693],
            [6.1442434, 0.8542896, 802439.1158526, 0.3091398],
            [6.1453974, 0.8542371, 802957.9497345, 0.3091054],
            [6.1465880, 0.8541829, 803493.2315348, 0.3090700],
            [6.1477750, 0.8541289, 804026.8701502, 0.3090347],
        ],
        # 9 m/s
        [
            [8.9703965, 0.7828869, 2480428.8963141, 0.2664440],
            [6.7170645, 0.8282386, 1059972.8615898, 0.2927795],
            [6.8249569, 0.8233319, 1108480.0451319, 0.2898405],
            [6.8927471, 0.8202489, 1138957.7499319, 0.2880147],
            [6.9051673, 0.8196840, 1144541.7037329, 0.2876819],
            [6.9063350, 0.8196309, 1145066.7068511, 0.2876506],
            [6.9070038, 0.8196005, 1145367.3983244, 0.2876327],
            [6.9079906, 0.8195556, 1145811.0578504, 0.2876063],
            [6.9091146, 0.8195045, 1146316.3678541, 0.2875762],
            [6.9103011, 0.8194506, 1146849.7967274, 0.2875444],
        ],
        # 10 m/s
        [
            [9.9671073, 0.7808960, 3395681.0032992, 0.2653854],
            [7.4669332, 0.7987766, 1441706.3550352, 0.2757103],
            [7.6196359, 0.7939336, 1531527.9847411, 0.2730273],
            [7.6698649, 0.7924100, 1561932.8306047, 0.2721897],
            [7.6769681, 0.7921945, 1566232.5614354, 0.2720716],
            [7.6778165, 0.7921687, 1566746.1707862, 0.2720574],
            [7.6784885, 0.7921484, 1567152.9075321, 0.2720463],
            [7.6793957, 0.7921208, 1567702.0865236, 0.2720312],
            [7.6804703, 0.7920882, 1568352.5711301, 0.2720133],
            [7.6816556, 0.7920523, 1569070.0707310, 0.2719936],
        ],
        # 11 m/s
        [
            [10.9638180, 0.7536370, 4488242.9153943, 0.2513413],
            [8.2691610, 0.7867811, 1972333.4291742, 0.2691218],
            [8.3808845, 0.7866371, 2055834.1618762, 0.2690439],
            [8.4335501, 0.7865693, 2095195.7177161, 0.2690072],
            [8.4435708, 0.7865563, 2102685.0693053, 0.2690002],
            [8.4462157, 0.7865529, 2104661.8573525, 0.2689983],
            [8.4475018, 0.7865513, 2105623.0376544, 0.2689974],
            [8.4486300, 0.7865498, 2106466.2862048, 0.2689967],
            [8.4498347, 0.7865483, 2107366.6617506, 0.2689958],
            [8.4511338, 0.7865466, 2108337.5277192, 0.2689949],
        ],
    ]
)

yaw_added_recovery_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736858, 0.7841561, 1741508.6722008, 0.2671213],
            [5.9689332, 0.8626156, 727222.3540334, 0.3146730],
            [6.0305406, 0.8594606, 751319.6495844, 0.3125571],
        ],
        # 9 m/s
        [
            [8.9703965, 0.7828869, 2480428.8963141, 0.2664440],
            [6.7170636, 0.8282387, 1059972.4826657, 0.2927795],
            [6.8187909, 0.8236123, 1105707.8700965, 0.2900073],
        ],
        # 10 m/s
        [
            [9.9671073, 0.7808960, 3395681.0032992, 0.2653854],
            [7.4669323, 0.7987766, 1441705.8203841, 0.2757103],
            [7.6128912, 0.7941382, 1527445.2805280, 0.2731400],
        ],
        # 11 m/s
        [
            [10.9638180, 0.7536370, 4488242.9153943, 0.2513413],
            [8.2691601, 0.7867811, 1972332.7278100, 0.2691218],
            [8.3736743, 0.7866464, 2050445.3384596, 0.2690489],
        ],
    ]
)

secondary_steering_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736858, 0.7841561, 1741508.6722008, 0.2671213],
            [5.9521559, 0.8635693, 721623.9542957, 0.3153174],
            [6.0187788, 0.8599955, 746031.6889128, 0.3129141],
        ],
        # 9 m/s
        [
            [8.9703965, 0.7828869, 2480428.8963141, 0.2664440],
            [6.6982618, 0.8290937, 1051519.3934629, 0.2932959],
            [6.8059255, 0.8241974, 1099923.7444659, 0.2903559],
        ],
        # 10 m/s
        [
            [9.9671073, 0.7808960, 3395681.0032992, 0.2653854],
            [7.4461678, 0.7994645, 1429777.9285494, 0.2760940],
            [7.5991268, 0.7945568, 1519127.2504621, 0.2733708],
        ],
        # 11 m/s
        [
            [10.9638180, 0.7536370, 4488242.9153943, 0.2513413],
            [8.2481967, 0.7868081, 1956664.9757307, 0.2691365],
            [8.3604363, 0.7866635, 2040551.4040835, 0.2690582],
        ],
    ]
)


# Note: compare the yawed vs non-yawed results. The upstream turbine
# power should be lower in the yawed case. The following turbine
# powers should higher in the yawed case.


def test_regression_tandem(sample_inputs_fixture):
    """
    Tandem turbines
    """
    if BACKEND == "floraf":
        sample_inputs_fixture.core["solver"]["backend"] = "cpp"
        sample_inputs_fixture.core["solver"]["device"] = "cpu"
        sample_inputs_fixture.core["solver"]["cpp_solver"] = "sequential"

    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    floris = Core.from_dict(sample_inputs_fixture.core)
    floris.initialize_domain()
    floris.steady_state_atmospheric_condition()

    n_turbines = floris.farm.n_turbines
    n_findex = floris.flow_field.n_findex

    velocities = floris.flow_field.u
    turbulence_intensities = floris.flow_field.turbulence_intensity_field
    air_density = floris.flow_field.air_density
    yaw_angles = floris.farm.yaw_angles
    tilt_angles = floris.farm.tilt_angles
    power_setpoints = floris.farm.power_setpoints
    awc_modes = floris.farm.awc_modes
    awc_amplitudes = floris.farm.awc_amplitudes
    test_results = np.zeros((n_findex, n_turbines, 4))

    farm_avg_velocities = average_velocity(
        velocities,
    )
    farm_cts = thrust_coefficient(
        velocities,
        turbulence_intensities,
        air_density,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_thrust_coefficient_functions,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    farm_powers = power(
        velocities,
        turbulence_intensities,
        air_density,
        floris.farm.turbine_power_functions,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_tilt_interps,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        turbulence_intensities,
        air_density,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_axial_induction_functions,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    for i in range(n_findex):
        for j in range(n_turbines):
            test_results[i, j, 0] = farm_avg_velocities[i, j]
            test_results[i, j, 1] = farm_cts[i, j]
            test_results[i, j, 2] = farm_powers[i, j]
            test_results[i, j, 3] = farm_axial_inductions[i, j]

    if DEBUG:
        print_test_values(
            farm_avg_velocities,
            farm_cts,
            farm_powers,
            farm_axial_inductions,
            max_findex_print=4,
        )

    assert_results_arrays(test_results[0:4], baseline)


def test_regression_rotation(sample_inputs_fixture):
    """
    Turbines in tandem and rotated.
    The result from 270 degrees should match the results from 360 degrees.

    Wind from the West (Left)

    ^
    |
    y

    1|1         3
     |
     |
     |
    0|0         2
     |----------|
      0         1  x->


    Wind from the North (Top), rotated

    ^
    |
    y

    1|3         2
     |
     |
     |
    0|1         0
     |----------|
      0         1  x->

    In 270, turbines 2 and 3 are waked. In 360, turbines 0 and 2 are waked.
    The test compares turbines 2 and 3 with 0 and 2 from 270 and 360.
    """
    TURBINE_DIAMETER = 126.0

    if BACKEND == "floraf":
        sample_inputs_fixture.core["solver"]["backend"] = "cpp"
        sample_inputs_fixture.core["solver"]["device"] = "cpu"
        sample_inputs_fixture.core["solver"]["cpp_solver"] = "sequential"

    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL
    sample_inputs_fixture.core["farm"]["layout_x"] = [
        0.0,
        0.0,
        5 * TURBINE_DIAMETER,
        5 * TURBINE_DIAMETER,
    ]
    sample_inputs_fixture.core["farm"]["layout_y"] = [
        0.0,
        5 * TURBINE_DIAMETER,
        0.0,
        5 * TURBINE_DIAMETER
    ]
    sample_inputs_fixture.core["flow_field"]["wind_directions"] = [270.0, 360.0]
    sample_inputs_fixture.core["flow_field"]["wind_speeds"] = [8.0, 8.0]
    sample_inputs_fixture.core["flow_field"]["turbulence_intensities"] = [0.1, 0.1]


    floris = Core.from_dict(sample_inputs_fixture.core)
    floris.initialize_domain()
    floris.steady_state_atmospheric_condition()

    farm_avg_velocities = average_velocity(floris.flow_field.u)

    t0_270 = farm_avg_velocities[0, 0]  # upstream
    t1_270 = farm_avg_velocities[0, 1]  # upstream
    t2_270 = farm_avg_velocities[0, 2]  # waked
    t3_270 = farm_avg_velocities[0, 3]  # waked

    t0_360 = farm_avg_velocities[1, 0]  # waked
    t1_360 = farm_avg_velocities[1, 1]  # upstream
    t2_360 = farm_avg_velocities[1, 2]  # waked
    t3_360 = farm_avg_velocities[1, 3]  # upstream

    assert np.allclose(t0_270, t1_360)
    assert np.allclose(t1_270, t3_360)
    assert np.allclose(t2_270, t0_360)
    assert np.allclose(t3_270, t2_360)


def test_regression_yaw(sample_inputs_fixture):
    """
    Tandem turbines with the upstream turbine yawed
    """
    if BACKEND == "floraf":
        sample_inputs_fixture.core["solver"]["backend"] = "cpp"
        sample_inputs_fixture.core["solver"]["device"] = "cpu"
        sample_inputs_fixture.core["solver"]["cpp_solver"] = "sequential"

    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    floris = Core.from_dict(sample_inputs_fixture.core)

    yaw_angles = np.zeros((N_FINDEX, N_TURBINES))
    yaw_angles[:,0] = 5.0
    floris.farm.yaw_angles = yaw_angles

    floris.initialize_domain()
    floris.steady_state_atmospheric_condition()

    n_turbines = floris.farm.n_turbines
    n_findex = floris.flow_field.n_findex

    velocities = floris.flow_field.u
    turbulence_intensities = floris.flow_field.turbulence_intensity_field
    air_density = floris.flow_field.air_density
    yaw_angles = floris.farm.yaw_angles
    tilt_angles = floris.farm.tilt_angles
    power_setpoints = floris.farm.power_setpoints
    awc_modes = floris.farm.awc_modes
    awc_amplitudes = floris.farm.awc_amplitudes
    test_results = np.zeros((n_findex, n_turbines, 4))

    farm_avg_velocities = average_velocity(
        velocities,
    )
    farm_cts = thrust_coefficient(
        velocities,
        turbulence_intensities,
        air_density,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_thrust_coefficient_functions,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    farm_powers = power(
        velocities,
        turbulence_intensities,
        air_density,
        floris.farm.turbine_power_functions,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_tilt_interps,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        turbulence_intensities,
        air_density,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_axial_induction_functions,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    for i in range(n_findex):
        for j in range(n_turbines):
            test_results[i, j, 0] = farm_avg_velocities[i, j]
            test_results[i, j, 1] = farm_cts[i, j]
            test_results[i, j, 2] = farm_powers[i, j]
            test_results[i, j, 3] = farm_axial_inductions[i, j]

    if DEBUG:
        print_test_values(
            farm_avg_velocities,
            farm_cts,
            farm_powers,
            farm_axial_inductions,
            max_findex_print=4,
        )

    assert_results_arrays(test_results[0:4], yawed_baseline)


def test_regression_gch(sample_inputs_fixture):
    """
    Turbines in a 10-turbine row with the upstream turbine yawed, yaw added
    recovery correction enabled, and secondary steering enabled.
    This test uses 10 turbines because the GCH components have an effect after the second turbine
    and deeper.
    """
    TURBINE_DIAMETER = 126.0
    N_TURBINES_GCH = 10
    layout_x = [i * 5 * TURBINE_DIAMETER for i in range(N_TURBINES_GCH)]
    layout_y = [0.0] * N_TURBINES_GCH

    if BACKEND == "floraf":
        sample_inputs_fixture.core["solver"]["backend"] = "cpp"
        sample_inputs_fixture.core["solver"]["device"] = "cpu"
        sample_inputs_fixture.core["solver"]["cpp_solver"] = "sequential"

    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL
    sample_inputs_fixture.core["farm"]["layout_x"] = layout_x
    sample_inputs_fixture.core["farm"]["layout_y"] = layout_y

    ### With GCH on, the results should be different from the regular Gauss baselines
    sample_inputs_fixture.core["wake"]["enable_transverse_velocities"] = True
    sample_inputs_fixture.core["wake"]["enable_secondary_steering"] = True
    sample_inputs_fixture.core["wake"]["enable_yaw_added_recovery"] = True

    floris = Core.from_dict(sample_inputs_fixture.core)

    yaw_angles = np.zeros((N_FINDEX, N_TURBINES_GCH))
    yaw_angles[:,0] = 5.0
    floris.farm.yaw_angles = yaw_angles

    floris.initialize_domain()
    floris.steady_state_atmospheric_condition()

    n_turbines = floris.farm.n_turbines
    n_findex = floris.flow_field.n_findex

    velocities = floris.flow_field.u
    turbulence_intensities = floris.flow_field.turbulence_intensity_field
    air_density = floris.flow_field.air_density
    yaw_angles = floris.farm.yaw_angles
    tilt_angles = floris.farm.tilt_angles
    power_setpoints = floris.farm.power_setpoints
    awc_modes = floris.farm.awc_modes
    awc_amplitudes = floris.farm.awc_amplitudes
    test_results = np.zeros((n_findex, n_turbines, 4))

    farm_avg_velocities = average_velocity(
        velocities,
    )
    farm_cts = thrust_coefficient(
        velocities,
        turbulence_intensities,
        air_density,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_thrust_coefficient_functions,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    farm_powers = power(
        velocities,
        turbulence_intensities,
        air_density,
        floris.farm.turbine_power_functions,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_tilt_interps,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        turbulence_intensities,
        air_density,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_axial_induction_functions,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    for i in range(n_findex):
        for j in range(n_turbines):
            test_results[i, j, 0] = farm_avg_velocities[i, j]
            test_results[i, j, 1] = farm_cts[i, j]
            test_results[i, j, 2] = farm_powers[i, j]
            test_results[i, j, 3] = farm_axial_inductions[i, j]

    if DEBUG:
        print_test_values(
            farm_avg_velocities,
            farm_cts,
            farm_powers,
            farm_axial_inductions,
            max_findex_print=4,
        )

    assert_results_arrays(test_results[0:4], gch_baseline)


def test_regression_yaw_added_recovery(sample_inputs_fixture):
    """
    Tandem turbines with the upstream turbine yawed and yaw added recovery
    correction enabled
    """
    if BACKEND == "floraf":
        sample_inputs_fixture.core["solver"]["backend"] = "cpp"
        sample_inputs_fixture.core["solver"]["device"] = "cpu"
        sample_inputs_fixture.core["solver"]["cpp_solver"] = "sequential"

    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    sample_inputs_fixture.core["wake"]["enable_transverse_velocities"] = True
    sample_inputs_fixture.core["wake"]["enable_secondary_steering"] = False
    sample_inputs_fixture.core["wake"]["enable_yaw_added_recovery"] = True

    floris = Core.from_dict(sample_inputs_fixture.core)

    yaw_angles = np.zeros((N_FINDEX, N_TURBINES))
    yaw_angles[:,0] = 5.0
    floris.farm.yaw_angles = yaw_angles

    floris.initialize_domain()
    floris.steady_state_atmospheric_condition()

    n_turbines = floris.farm.n_turbines
    n_findex = floris.flow_field.n_findex

    velocities = floris.flow_field.u
    turbulence_intensities = floris.flow_field.turbulence_intensity_field
    air_density = floris.flow_field.air_density
    yaw_angles = floris.farm.yaw_angles
    tilt_angles = floris.farm.tilt_angles
    power_setpoints = floris.farm.power_setpoints
    awc_modes = floris.farm.awc_modes
    awc_amplitudes = floris.farm.awc_amplitudes
    test_results = np.zeros((n_findex, n_turbines, 4))

    farm_avg_velocities = average_velocity(
        velocities,
    )
    farm_cts = thrust_coefficient(
        velocities,
        turbulence_intensities,
        air_density,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_thrust_coefficient_functions,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    farm_powers = power(
        velocities,
        turbulence_intensities,
        air_density,
        floris.farm.turbine_power_functions,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_tilt_interps,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        turbulence_intensities,
        air_density,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_axial_induction_functions,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    for i in range(n_findex):
        for j in range(n_turbines):
            test_results[i, j, 0] = farm_avg_velocities[i, j]
            test_results[i, j, 1] = farm_cts[i, j]
            test_results[i, j, 2] = farm_powers[i, j]
            test_results[i, j, 3] = farm_axial_inductions[i, j]

    if DEBUG:
        print_test_values(
            farm_avg_velocities,
            farm_cts,
            farm_powers,
            farm_axial_inductions,
            max_findex_print=4,
        )

    assert_results_arrays(test_results[0:4], yaw_added_recovery_baseline)


def test_regression_secondary_steering(sample_inputs_fixture):
    """
    Tandem turbines with the upstream turbine yawed and secondary steering enabled
    """
    if BACKEND == "floraf":
        sample_inputs_fixture.core["solver"]["backend"] = "cpp"
        sample_inputs_fixture.core["solver"]["device"] = "cpu"
        sample_inputs_fixture.core["solver"]["cpp_solver"] = "sequential"

    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    sample_inputs_fixture.core["wake"]["enable_transverse_velocities"] = True
    sample_inputs_fixture.core["wake"]["enable_secondary_steering"] = True
    sample_inputs_fixture.core["wake"]["enable_yaw_added_recovery"] = False

    floris = Core.from_dict(sample_inputs_fixture.core)

    yaw_angles = np.zeros((N_FINDEX, N_TURBINES))
    yaw_angles[:,0] = 5.0
    floris.farm.yaw_angles = yaw_angles

    floris.initialize_domain()
    floris.steady_state_atmospheric_condition()

    n_turbines = floris.farm.n_turbines
    n_findex = floris.flow_field.n_findex

    velocities = floris.flow_field.u
    turbulence_intensities = floris.flow_field.turbulence_intensity_field
    air_density = floris.flow_field.air_density
    yaw_angles = floris.farm.yaw_angles
    tilt_angles = floris.farm.tilt_angles
    power_setpoints = floris.farm.power_setpoints
    awc_modes = floris.farm.awc_modes
    awc_amplitudes = floris.farm.awc_amplitudes
    test_results = np.zeros((n_findex, n_turbines, 4))

    farm_avg_velocities = average_velocity(
        velocities,
    )
    farm_cts = thrust_coefficient(
        velocities,
        turbulence_intensities,
        air_density,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_thrust_coefficient_functions,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    farm_powers = power(
        velocities,
        turbulence_intensities,
        air_density,
        floris.farm.turbine_power_functions,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_tilt_interps,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        turbulence_intensities,
        air_density,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_axial_induction_functions,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    for i in range(n_findex):
        for j in range(n_turbines):
            test_results[i, j, 0] = farm_avg_velocities[i, j]
            test_results[i, j, 1] = farm_cts[i, j]
            test_results[i, j, 2] = farm_powers[i, j]
            test_results[i, j, 3] = farm_axial_inductions[i, j]

    if DEBUG:
        print_test_values(
            farm_avg_velocities,
            farm_cts,
            farm_powers,
            farm_axial_inductions,
            max_findex_print=4,
        )

    assert_results_arrays(test_results[0:4], secondary_steering_baseline)


def test_regression_small_grid_rotation(sample_inputs_fixture):
    """
    This utilizes a 5x5 wind farm with the layout in a regular grid oriented along the cardinal
    directions. The wind direction in this test is from 285 degrees which is slightly north of
    west. The objective of this test is to create a case with a very slight rotation of the wind
    farm to target the rotation and masking routines.

    Where wake models are masked based on the x-location of a turbine, numerical precision
    can cause masking to fail unexpectedly. For example, in the configuration here one of
    the turbines has these delta x values;

    [[4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13]
     [4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13]
     [4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13]
     [4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13]
     [4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13]]

    and therefore the masking statement is False when it should be True. This causes the current
    turbine to be affected by its own wake. This test requires that at least in this particular
    configuration the masking correctly filters grid points.
    """
    if BACKEND == "floraf":
        sample_inputs_fixture.core["solver"]["backend"] = "cpp"
        sample_inputs_fixture.core["solver"]["device"] = "cpu"
        sample_inputs_fixture.core["solver"]["cpp_solver"] = "sequential"

    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL
    X, Y = np.meshgrid(
        6.0 * 126.0 * np.arange(0, 5, 1),
        6.0 * 126.0 * np.arange(0, 5, 1)
    )
    X = X.flatten()
    Y = Y.flatten()

    sample_inputs_fixture.core["farm"]["layout_x"] = X
    sample_inputs_fixture.core["farm"]["layout_y"] = Y

    floris = Core.from_dict(sample_inputs_fixture.core)
    floris.initialize_domain()
    floris.steady_state_atmospheric_condition()

    # farm_avg_velocities = average_velocity(floris.flow_field.u)
    velocities = floris.flow_field.u
    turbulence_intensities = floris.flow_field.turbulence_intensity_field
    yaw_angles = floris.farm.yaw_angles
    tilt_angles = floris.farm.tilt_angles
    power_setpoints = floris.farm.power_setpoints
    awc_modes = floris.farm.awc_modes
    awc_amplitudes = floris.farm.awc_amplitudes

    farm_powers = power(
        velocities,
        turbulence_intensities,
        floris.flow_field.air_density,
        floris.farm.turbine_power_functions,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_tilt_interps,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )

    # A "column" is oriented parallel to the wind direction
    # Columns 1 - 4 should have the same power profile
    # Column 5 leading turbine is completely unwaked
    # and the rest of the turbines have a partial wake from their immediate upstream turbine
    assert np.allclose(farm_powers[8,0:5], farm_powers[8,5:10])
    assert np.allclose(farm_powers[8,0:5], farm_powers[8,10:15])
    assert np.allclose(farm_powers[8,0:5], farm_powers[8,15:20])
    assert np.allclose(farm_powers[8,20], farm_powers[8,0])
    assert np.allclose(farm_powers[8,21], farm_powers[8,21:25])


def test_full_flow_solver(sample_inputs_fixture):
    """
    Full flow solver test with the flow field planar grid.
    This requires one wind condition, and the grid is deliberately coarse to allow for
    visually comparing results, as needed.
    The u-component of velocity is compared, and the array has the shape
    (n_findex, n_turbines, n grid points in x, n grid points in y, 3 grid points in z).
    """
    # NOTE: flow_field_planar_grid is not supported in FLORAF
    from floris.core import Core

    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL
    sample_inputs_fixture.core["solver"] = {
        "type": "flow_field_planar_grid",
        "normal_vector": "z",
        "planar_coordinate": sample_inputs_fixture.core["farm"]["turbine_type"][0]["hub_height"],
        "flow_field_grid_points": [5, 5],
        "flow_field_bounds": [None, None],
    }
    sample_inputs_fixture.core["flow_field"]["wind_directions"] = [270.0]
    sample_inputs_fixture.core["flow_field"]["wind_speeds"] = [8.0]
    sample_inputs_fixture.core["flow_field"]["turbulence_intensities"] = [0.1]

    floris = Core.from_dict(sample_inputs_fixture.core)
    floris.solve_for_viz()

    velocities = floris.flow_field.u_sorted

    assert_results_arrays(velocities, full_flow_baseline)
