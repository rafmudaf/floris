# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


from pathlib import Path

import matplotlib.pyplot as plt

import floris.tools as wfct


ROOT = Path(__file__).resolve().parent
FILE_NAME = "example_input.json"

root_parts = ROOT.parts
if root_parts[-1] == "_getting_started":
    INPUT_JSON = ROOT.parents[0] / FILE_NAME
elif root_parts[-1] == "examples":
    INPUT_JSON = ROOT / FILE_NAME
elif root_parts[-1] == "floris":
    INPUT_JSON = ROOT / "examples" / "example_input.json"
else:
    raise FileNotFoundError(
        "Examples must be run from with floris/, floris/examples/, or floris/examples/<topic-folder>!"
    )


# Initialize the FLORIS interface fi
fi = wfct.floris_interface.FlorisInterface(INPUT_JSON)

# Show the current model parameters
print("All the model parameters and their current values:\n")
fi.show_model_parameters(turbulence_model=False)
print("\n")

# Show the current model parameters with docstring info
print("All the model parameters, their current values, and docstrings:\n")
fi.show_model_parameters(verbose=True, turbulence_model=False)
print("\n")

# Show a specific model parameter with its docstring
print("A specific model parameter, its current value, and its docstring:\n")
fi.show_model_parameters(params=["ka"], verbose=False, turbulence_model=False)
print("\n")

# Get the current model parameters
model_params = fi.get_model_parameters(turbulence_model=False)
print("The current model parameters:\n")
print(model_params)
print("\n")

# Set parameters on the current model
print("Set specific model parameters on the current wake model:\n")
params = {
    "Wake Velocity Parameters": {"alpha": 0.2},
    "Wake Deflection Parameters": {"alpha": 0.2},
    # "Wake Turbulence Parameters": {"ti_constant": 1.0},
}
fi.set_model_parameters(params)
print("\n")

# Check that the parameters were changed
print("Observe that the requested paremeters changes have been made:\n")
model_params = fi.get_model_parameters(turbulence_model=False)
print(model_params)
print("\n")
