# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from __future__ import annotations

from copy import deepcopy
from typing import Any

from floris.simulation import WakeModelManager
from floris.simulation.base import BaseModel


"""
=====================================
Wake Velocity Model Parameters: gauss_legacy model
-------------------------------------
alpha = 0.58
beta = 0.077
calculate_VW_velocities = True
eps_gain = 0.2
ka = 0.38
kb = 0.004
logger = <Logger floris.simulation.wake_velocity.gaussianModels.gauss_legacy.LegacyGauss (DEBUG)>
use_yaw_added_recovery = True
-------------------------------------
"""

HEADER = "=".join(["="] * 19)
SEPARATOR = "-".join(["-"] * 19)
VERBOSE_HEADER = "=".join(["="] * 39)
VERBOSE_SEPARATOR = "-".join(["-"] * 39)


def show_params(
    wake: WakeModelManager,
    params: list[str] | None = None,
    verbose: bool = False,
    wake_velocity_model: bool = True,
    wake_deflection_model: bool = True,
    turbulence_model: bool = True,
):

    wake_parameters = wake._asdict()
    header = HEADER
    if verbose:
        header = VERBOSE_HEADER

    if wake_velocity_model:
        model = wake.model_strings["velocity_model"]
        title = f"Wake Velocity Model Parameters: {model} model"
        print(f"{header}\n{title}")

        parameters = [*wake_parameters["wake_velocity_parameters"][model]]
        if params is not None:
            parameters = get_paramater_subset(params, parameters)

        if verbose:
            print_model_docs(wake.velocity_model, wake_parameters["wake_velocity_parameters"][model], parameters)
        else:
            print_parameters(wake_parameters["wake_velocity_parameters"][model], parameters)

    if wake_deflection_model:
        model = wake.model_strings["deflection_model"]
        title = f"Wake Deflection Model Parameters: {model} model"
        print(f"{header}\n{title}")

        parameters = [*wake_parameters["wake_deflection_parameters"][model]]
        if params is not None:
            parameters = get_paramater_subset(params, parameters)

        if verbose:
            print_model_docs(wake.deflection_model, wake_parameters["wake_deflection_parameters"][model], parameters)
        else:
            print_parameters(wake_parameters["wake_deflection_parameters"][model], parameters)

    if turbulence_model:
        model = wake.model_strings["turbulence_model"]
        title = f"Wake Turbulence Model Parameters: {model} model"
        print(f"{header}\n{title}")

        parameters = [*wake_parameters["wake_turbulence_parameters"][model]]
        if params is not None:
            parameters = get_paramater_subset(params, parameters)

        if verbose:
            print_model_docs(wake.turbulence_model, wake_parameters["wake_turbulence_parameters"][model], parameters)
        else:
            print_parameters(wake_parameters["wake_turbulence_parameters"][model], parameters)


def get_params(
    wake: WakeModelManager,
    params: list[str] | None = None,
    wake_velocity_model: bool = True,
    wake_deflection_model: bool = True,
    turbulence_model: bool = True,
):
    model_params = wake._asdict()
    model_params.pop("model_strings")
    # TODO: Uncomment!
    # model_params.pop("wake_combination_parameters")

    if wake_velocity_model:
        model = wake.model_strings["velocity_model"]
        _parameters = model_params["wake_velocity_parameters"][model]
        subset = get_paramater_subset(params, [*_parameters])
        model_params["wake_velocity_parameters"] = drop_parameters(_parameters, subset)
    else:
        model_params.pop("wake_velocity_parameters")

    if wake_deflection_model:
        model = wake.model_strings["deflection_model"]
        _parameters = model_params["wake_deflection_parameters"][model]
        subset = get_paramater_subset(params, [*_parameters])
        model_params["wake_deflection_parameters"] = drop_parameters(_parameters, subset)
    else:
        model_params.pop("wake_deflection_parameters")

    if turbulence_model:
        model = wake.model_strings["turbulence_model"]
        _parameters = model_params["wake_turbulence_parameters"][model]
        subset = get_paramater_subset(params, [*_parameters])
        model_params["wake_turbulence_parameters"] = drop_parameters(_parameters, subset)
    # TODO: Uncomment!
    # else:
    #     model_params.pop("wake_turbulence_parameters")

    return model_params


def set_params(
    wake: WakeModelManager, model_parameter_dictionary: dict[str, dict[str, Any]], verbose: bool = True
) -> WakeModelManager:
    """Creates a new instance of the wake object with the provided modifications in the
    `model_parameter_dictionary`.

    Args:
        wake (WakeModelManager): The wake model manager, should be floris.wake.
        model_parameter_dictionary (dict[str, dict[str, Any]]): The dictionary of models
            and model parameters.
        verbose (bool, optional): Calls `show_params` with `verbose`=True. Defaults to True.

    Returns:
        WakeModelManager: The new wake model manager object.
    """

    # Make a new copy of the wake model manager
    wake = deepcopy(wake)

    # Update the key names from title case to the expected format
    for key in model_parameter_dictionary:
        new_key = key.lower().replace(" ", "_")
        model_parameter_dictionary[new_key] = model_parameter_dictionary.pop(key)

    # Update each of the model parameters dictionary within the wake model manager
    if "wake_velocity_parameters" in model_parameter_dictionary:
        model = wake.model_strings["velocity_model"]
        model_parameters = model_parameter_dictionary["wake_velocity_parameters"]
        wake.wake_velocity_parameters[model].update(model_parameters)

    if "wake_deflection_parameters" in model_parameter_dictionary:
        model = wake.model_strings["deflection_model"]
        model_parameters = model_parameter_dictionary["wake_deflection_parameters"]
        wake.wake_deflection_parameters[model].update(model_parameters)

    if "wake_turbulence_parameters" in model_parameter_dictionary:
        model = wake.model_strings["turbulence_model"]
        model_parameters = model_parameter_dictionary["wake_turbulence_parameters"]
        wake.wake_turbulence_parameters[model].update(model_parameters)

    # Recreate the wake models
    wake.model_generator()

    if verbose:
        # TODO: remove turbulence model filter!
        show_params(wake, verbose=verbose, turbulence_model=False)

    return wake


def get_paramater_subset(subset: list[str], parameters: list[str]) -> list[str]:
    """Retrieve a subset of the model parameters/properties

    Args:
        subset (list[str]): The subset of parameters desired.
        parameters (list[str]): The complete list of model parameters.

    Returns:
        list[str]: The valid subset of model parameters
    """
    if subset is None:
        return parameters
    return [p for p in subset if p in parameters]


def print_parameters(parameters_dict: dict[str, Any], parameter_subset: list[str]) -> None:
    """Prints the subset of wake model parameters.

    Args:
        parameters_dict (dict[str, Any]): The dictionary of available parameters and their values.
        parameter_subset (list[str]): The subset of model parameters to be printed.
    """
    print(SEPARATOR)
    for name in parameter_subset:
        if name == "model_string":
            continue
        print(f"{name} = {parameters_dict[name]}")
    print(SEPARATOR)


def print_model_docs(model: BaseModel, parameters_dict: dict[str, Any], parameters_subset: list[str]) -> None:
    """Prints the wake models documentation and the current values of the parameters in
    `parameters_subset`.

    Args:
        model (BaseModel): A valid wake model that is a subclass of the `BaseModel` base class.
        parameters_dict (dict[str, Any]): The dictionary of available parameters and their values.
        parameter_subset (list[str]): The subset of model parameters to be printed.
    """
    print(VERBOSE_SEPARATOR)
    print(model.__doc__)
    print_parameters(parameters_dict, parameters_subset)
    print(VERBOSE_SEPARATOR)


def drop_parameters(model_dictionary: dict[str, Any], parameter_subset: list[str]) -> dict[str, Any]:
    """Drops any unwanted parameters from the model parameter dictionary.

    Args:
        model_dictionary (dict[str, Any]): The model parameters dictionary, as defined by the input arguments.
        parameter_subset (list[str]): The subset of desired parameters.

    Returns:
        dict[str, Any]: The filtered model parameters dictionary.
    """
    for parameter in model_dictionary:
        if parameter not in parameter_subset:
            model_dictionary.pop(parameter)
    return model_dictionary
