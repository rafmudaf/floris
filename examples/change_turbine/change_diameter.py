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


# Initializea and create the turbine mappings that we need for the example
fi = wfct.floris_interface.FlorisInterface(INPUT_JSON)
fi.reinitialize_flow_field(layout_array=[[0, 0], [0, 1000]])

# Create a dictionary to store the new turbine maps based the diameter we're updating to
diameters = np.arange(80, 160, 1.0)
diameter_turbine_map = {
    d: fi.copy_and_update_turbine_map(
        base_turbine_id="nrel_5mw",
        update_parameters={"rotor_diameter": d},
        new_id=f"nrel_5mw_rd{d}m",
    )
    for d in diameters
}


# Side by Side, adjust both T0 and T1 diameters
fi = wfct.floris_interface.FlorisInterface(INPUT_JSON)
fi.reinitialize_flow_field(layout_array=[[0, 0], [0, 1000]])

# Calculate wake
fi.calculate_wake()
init_power = np.array(fi.get_turbine_power()) / 1000.0

fig, axarr = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(15, 5))

# Show the hub-height slice in the 3rd pane
hor_plane = fi.get_hor_plane()
wfct.visualization.visualize_cut_plane(hor_plane, ax=axarr[2])

for t in range(2):

    ax = axarr[t]

    # Now sweep the heights for this turbine
    powers = np.zeros_like(diameters)

    for d_idx, d in enumerate(diameters):
        # Change the turbine "t" to be the nrel_5mw turbine with rotor diameter "d"
        fi.change_turbine(turbine_indices=[t], new_turbine_map=diameter_turbine_map[d])
        fi.calculate_wake()
        # There is only 1 wind direction and wind speed, so we can just grab all of them in this case
        powers[d_idx] = fi.get_turbine_power()[:, :, t] / 1000.0

    ax.plot(diameters, powers, "k")
    ax.axhline(init_power[:, :, t], color="r", ls=":")
    ax.axvline(126, color="r", ls=":")
    ax.set_title("T%d" % t)
    ax.set_xlim([80, 160])
    ax.set_ylim([200, 3000])
    ax.set_xlabel("Diameter T%d" % t)
    ax.set_ylabel("Power")

plt.suptitle("Adjusting Both T0 and T1 Diameters")

# Waked, adjust T0 diameter
fi = wfct.floris_interface.FlorisInterface(INPUT_JSON)
fi.reinitialize_flow_field(layout_array=[[0, 500], [0, 0]])

# Calculate wake
fi.calculate_wake()
init_power = np.array(fi.get_turbine_power()) / 1000.0

fig, axarr = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(15, 5))

# Show the hub-height slice in the 3rd pane
hor_plane = fi.get_hor_plane()
wfct.visualization.visualize_cut_plane(hor_plane, ax=axarr[2])

for t in range(2):

    ax = axarr[t]

    # Now sweep the heights for this turbine
    powers = np.zeros_like(diameters)

    for d_idx, d in enumerate(diameters):
        # Change the turbine 0 to be the nrel_5mw turbine with rotor diameter "d"
        fi.change_turbine(turbine_indices=[0], new_turbine_map=diameter_turbine_map[d])
        fi.calculate_wake()
        # There is only 1 wind direction and wind speed, so we can just grab all of them in this case
        powers[d_idx] = fi.get_turbine_power()[:, :, t] / 1000.0

    ax.plot(diameters, powers, "k")
    ax.axhline(init_power[:, :, t], color="r", ls=":")
    ax.axvline(126, color="r", ls=":")
    ax.set_title("T%d" % t)
    ax.set_xlim([80, 160])
    ax.set_ylim([200, 3000])
    ax.set_xlabel("Diameter T0")
    ax.set_ylabel("Power")

plt.suptitle("Adjusting T0 Diameter")

# Waked, adjust T1 diameter
fi = wfct.floris_interface.FlorisInterface(INPUT_JSON)
fi.reinitialize_flow_field(layout_array=[[0, 500], [0, 0]])

# Calculate wake
fi.calculate_wake()
init_power = np.array(fi.get_turbine_power()) / 1000.0

fig, axarr = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(15, 5))

# Show the hub-height slice in the 3rd pane
hor_plane = fi.get_hor_plane()
wfct.visualization.visualize_cut_plane(hor_plane, ax=axarr[2])

for t in range(2):

    ax = axarr[t]

    # Now sweep the heights for this turbine
    powers = np.zeros_like(diameters)

    for d_idx, d in enumerate(diameters):
        # Change the turbine 1 to be the nrel_5mw turbine with rotor diameter "d"
        fi.change_turbine(turbine_indices=[1], new_turbine_map=diameter_turbine_map[d])
        fi.calculate_wake()
        # There is only 1 wind direction and wind speed, so we can just grab all of them in this case
        powers[d_idx] = fi.get_turbine_power()[:, :, t] / 1000.0

    ax.plot(diameters, powers, "k")
    ax.axhline(init_power[:, :, t], color="r", ls=":")
    ax.axvline(126, color="r", ls=":")
    ax.set_title("T%d" % t)
    ax.set_xlim([80, 160])
    ax.set_ylim([200, 3000])
    ax.set_xlabel("Diameter T1")
    ax.set_ylabel("Power")

plt.suptitle("Adjusting T1 Diameter")

plt.show()
