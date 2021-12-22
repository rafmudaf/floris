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


# This example illustrates changing the properties of some of the turbines
# This can be used to setup farms of different turbines

from pathlib import Path

import matplotlib.pyplot as plt

from floris.simulation import turbine
from floris.tools.visualization import visualize_cut_plane
from floris.tools.floris_interface import FlorisInterface


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
fi = FlorisInterface(INPUT_JSON)

# Set to 2x2 farm
layout_x = [0, 0, 600, 600]
layout_y = [0, 300, 0, 300]
fi.reinitialize_flow_field(layout_array=[layout_x, layout_y])


# Change turbine 0 and 3 to have a 35 m rotor diameter
turbine_rd35 = {"nrel_5mw_35m": fi.floris.turbine["nrel_5mw"]._asdict()}
turbine_rd35["nrel_5mw_35m"]["rotor_diameter"] = 35
fi.change_turbine(turbine_indices=[0, 3], new_turbine_map=turbine_rd35)

# Calculate wake
fi.calculate_wake()

# Get horizontal plane at default height (hub-height)
hor_plane = fi.get_hor_plane()

# Plot and show
fig, ax = plt.subplots()
visualize_cut_plane(hor_plane, ax=ax)
plt.show()
