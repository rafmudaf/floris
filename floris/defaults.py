
DEFAULT_INPUTS = {
    'name': 'DEFAULT',
    'description': '',
    'floris_version': 'v3.4.0',
    'logging': {
        'console': {'enable': True, 'level': 'WARNING'},
        'file': {'enable': False, 'level': 'WARNING'}
    },
    'solver': {
        'type': 'turbine_grid',
        'turbine_grid_points': 3
    },
    'farm': {
        'layout_x': [0.0, 5 * 126.0],
        'layout_y': [0.0, 0.0],
        'turbine_type': ['nrel_5MW']
    },
    'flow_field': {
        'air_density': 1.225,
        'reference_wind_height': 90.0,
        'turbulence_intensity': 0.06,
        'wind_directions': [270.0],
        'wind_shear': 0.12,
        'wind_speeds': [8.0],
        'wind_veer': 0.0
    },
    'wake': {
        'model_strings': {
            'combination_model': 'sosfs',
            'deflection_model': 'gauss',
            'turbulence_model': 'crespo_hernandez',
            'velocity_model': 'gauss'
        },
        'enable_secondary_steering': False,
        'enable_yaw_added_recovery': False,
        'enable_transverse_velocities': False,
        'wake_deflection_parameters': {
            'gauss': {
                'ad': 0.0,
                'alpha': 0.58,
                'bd': 0.0,
                'beta': 0.077,
                'dm': 1.0,
                'ka': 0.38,
                'kb': 0.004
            },
            'jimenez': {
                'ad': 0.0,
                'bd': 0.0,
                'kd': 0.05
            }
        },
        'wake_velocity_parameters': {
            'gauss': {
                'alpha': 0.58,
                'beta': 0.077,
                'ka': 0.38,
                'kb': 0.004
            },
            'jensen': {
                'we': 0.05
            }
        },
        'wake_turbulence_parameters': {
            'crespo_hernandez': {
                'initial': 0.1,
                'constant': 0.5,
                'ai': 0.8,
                'downstream': -0.32
            }
        }
    }
}
