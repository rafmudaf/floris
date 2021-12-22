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

# This example shows a quick illustration of using different wake models
# Note that for using the Jensen or Multizone model, even if not studying
# wake steering, it's important to couple with the Jimenez model of deflection
# to avoid software errors in functions only defined for gaussian models

from pathlib import Path

import matplotlib.pyplot as plt

import floris.tools as wfct


ROOT = Path(__file__).resolve().parent
root_parts = ROOT.parts
if root_parts[-1] == "_getting_started":
    OTHER_JSONS = ROOT.parents[0] / "other_jsons"
elif root_parts[-1] == "examples":
    OTHER_JSONS = ROOT / "other_jsons"
elif root_parts[-1] == "floris":
    OTHER_JSONS = ROOT / "examples" / "other_jsons"
else:
    raise FileNotFoundError(
        "Examples must be run from with floris/, floris/examples/, or floris/examples/<topic-folder>!"
    )


# Initialize the FLORIS interface for 4 seperate models defined as JSONS
# NOTE: the turbopark, multizone, and gauss legacy models are not currently impolemented, so will not be compared
fi_jensen = wfct.floris_interface.FlorisInterface(OTHER_JSONS / "jensen.json")
# fi_turbopark = wfct.floris_interface.FlorisInterface(OTHER_JSONS / "turbopark.json")
# fi_mz = wfct.floris_interface.FlorisInterface(OTHER_JSONS / "multizone.json")
# fi_gauss = wfct.floris_interface.FlorisInterface(OTHER_JSONS / "input_legacy.json")
fi_gch = wfct.floris_interface.FlorisInterface(OTHER_JSONS.parent / "example_input.json")

fig, axarr = plt.subplots(2, 5, figsize=(16, 4))


# Use a python for loop to iterate over the models and plot a horizontal cut through
# of the models for an aligned and yaw case to show some differences
for idx, (fi, name) in enumerate(zip([fi_jensen, fi_gch], ["Jensen", "GCH"])):

    # Aligned case
    fi.calculate_wake(yaw_angles=[0])
    ax = axarr[0, idx]
    hor_plane = fi.get_hor_plane()
    wfct.visualization.visualize_cut_plane(hor_plane, ax=ax, minSpeed=4, maxSpeed=8)
    ax.set_title(name)
    axarr[0, 0].set_ylabel("Aligned")

    # Yawed case
    fi.calculate_wake(yaw_angles=[25])
    ax = axarr[1, idx]
    hor_plane = fi.get_hor_plane()
    wfct.visualization.visualize_cut_plane(hor_plane, ax=ax, minSpeed=4, maxSpeed=8)
    axarr[1, 0].set_ylabel("Yawed")


# Show the figure
plt.show()
