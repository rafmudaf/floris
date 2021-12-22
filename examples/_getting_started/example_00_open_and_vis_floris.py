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

from floris.tools import FlorisInterface
from floris.tools.visualization import visualize_cut_plane


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
# For basic usage, the florice interface provides a simplified interface to
# the underlying classes
fi = FlorisInterface(INPUT_JSON)

yaw_angles = np.zeros((1,1,3))
yaw_angles[:,:,0] = 25.0
fi.floris.farm.yaw_angles = yaw_angles

# Calculate wake
fi.floris.solve_for_viz()

# Get horizontal plane at default height (hub-height)
hor_plane = fi.get_hor_plane()

# Plot and show
fig, ax = plt.subplots()
visualize_cut_plane(hor_plane, ax=ax)
plt.show()
