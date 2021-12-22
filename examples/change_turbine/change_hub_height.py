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

import numpy as np
import matplotlib.pyplot as plt

import floris.tools as wfct


ROOT = Path(__file__).resolve().parent
FILE_NAME = "example_input.json"

root_parts = ROOT.parts
if root_parts[-1] == "change_turbine":
    INPUT_JSON = ROOT.parents[0] / FILE_NAME
elif root_parts[-1] == "examples":
    INPUT_JSON = ROOT / FILE_NAME
elif root_parts[-1] == "floris":
    INPUT_JSON = ROOT / "examples" / "example_input.json"
else:
    raise FileNotFoundError(
        "Examples must be run from with floris/, floris/examples/, or floris/examples/<topic-folder>!"
    )


# Side by Side, adjust T0 and T1 heights
fi = wfct.floris_interface.FlorisInterface(INPUT_JSON)
fi.reinitialize_flow_field(layout_array=[[0, 0], [0, 1000]])

heights = np.arange(70, 120, 1.0)
hub_height_turbine_map = {
    h: fi.copy_and_update_turbine_map(
        base_turbine_id="nrel_5mw",
        update_parameters={"hub_height": h},
        new_id=f"nrel_5mw_hh{h}m",
    )
    for h in heights
}


# Side by Side, adjust T0 and T1 heights
fi = wfct.floris_interface.FlorisInterface(INPUT_JSON)
fi.reinitialize_flow_field(layout_array=[[0, 0], [0, 1000]])

# Calculate wake
fi.calculate_wake()
init_power = np.array(fi.get_turbine_power()) / 1000.0

fig, axarr = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(15, 5))

# Show the hub-height slice in the 3rd pane
# hor_plane = fi.get_hor_plane()
# wfct.visualization.visualize_cut_plane(hor_plane, ax=axarr[2])


for t in range(2):

    ax = axarr[t]

    # Now sweep the heights for this turbine
    powers = np.zeros_like(heights)

    for h_idx, h in enumerate(heights):
        # Change the turbine "t" to be the nrel_5mw turbine with hub height "h"
        fi.change_turbine(turbine_indices=[t], new_turbine_map=hub_height_turbine_map[h])
        fi.calculate_wake()
        # There is only 1 wind direction and wind speed, so we can just grab all of them in this case
        powers[h_idx] = fi.get_turbine_power()[:, :, t] / 1000.0

    ax.plot(heights, powers, "k")
    ax.axhline(init_power[:, :, t], color="r", ls=":")
    ax.axvline(90, color="r", ls=":")
    ax.set_title("T%d" % t)
    ax.set_xlim([70, 120])
    ax.set_ylim([1000, 2000])
    ax.set_xlabel("Hub Height T%d" % t)
    ax.set_ylabel("Power")

plt.suptitle("Adjusting Both Turbine Heights")


# Waked, adjust T0 height
fi = wfct.floris_interface.FlorisInterface(INPUT_JSON)
fi.reinitialize_flow_field(layout_array=[[0, 500], [0, 0]])

# Calculate wake
fi.calculate_wake()
init_power = np.array(fi.get_turbine_power()) / 1000.0

fig, axarr = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(15, 5))

# Show the hub-height slice in the 3rd pane
# hor_plane = fi.get_hor_plane()
# wfct.visualization.visualize_cut_plane(hor_plane, ax=axarr[2])


for t in range(2):

    ax = axarr[t]

    # Now sweep the heights for this turbine
    powers = np.zeros_like(heights)

    for h_idx, h in enumerate(heights):
        # Change the turbine "0" to be the nrel_5mw turbine with hub height "h"
        fi.change_turbine(turbine_indices=[0], new_turbine_map=hub_height_turbine_map[h])
        fi.calculate_wake()
        # There is only 1 wind direction and wind speed, so we can just grab all of them in this case
        powers[h_idx] = fi.get_turbine_power()[:, :, t] / 1000.0

    ax.plot(heights, powers, "k")
    ax.axhline(init_power[:, :, t], color="r", ls=":")
    ax.axvline(90, color="r", ls=":")
    ax.set_title("T%d" % t)
    ax.set_xlim([50, 120])
    ax.set_xlabel("Hub Height T0")
    ax.set_ylabel("Power T%d" % t)

plt.suptitle("Adjusting T0 Height")

# Waked, adjust T1 height
fi = wfct.floris_interface.FlorisInterface(INPUT_JSON)
fi.reinitialize_flow_field(layout_array=[[0, 500], [0, 0]])

# Calculate wake
fi.calculate_wake()
init_power = np.array(fi.get_turbine_power()) / 1000.0

fig, axarr = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(15, 5))

# Show the hub-height slice in the 3rd pane
# hor_plane = fi.get_hor_plane()
# wfct.visualization.visualize_cut_plane(hor_plane, ax=axarr[2])


for t in range(2):

    ax = axarr[t]

    # Now sweep the heights for this turbine
    powers = np.zeros_like(heights)

    for h_idx, h in enumerate(heights):
        # Change the turbine "1" to be the nrel_5mw turbine with hub height "h"
        fi.change_turbine(turbine_indices=[1], new_turbine_map=hub_height_turbine_map[h])
        fi.calculate_wake()
        # There is only 1 wind direction and wind speed, so we can just grab all of them in this case
        powers[h_idx] = fi.get_turbine_power()[:, :, t] / 1000.0

    ax.plot(heights, powers, "k")
    ax.axhline(init_power[:, :, t], color="r", ls=":")
    ax.axvline(90, color="r", ls=":")
    ax.set_title("T%d" % t)
    ax.set_xlim([70, 120])
    ax.set_xlabel("Hub Height T1")
    ax.set_ylabel("Power T%d" % t)

plt.suptitle("Adjusting T1 Height")

plt.show()
