
import numpy as np
from floris import FlorisModel

fmodel = FlorisModel("inputs/cc.yaml")
n_turbines = 128
n_findex = 20
fmodel.set(
    layout_x=np.array([tt * 1000 for tt in range(n_turbines)]),
    layout_y=np.array(n_turbines * [0]),
    wind_directions=np.array(n_findex * [270]),
    wind_speeds=np.array(n_findex * [10]),
    turbulence_intensities=np.array(n_findex * [0.06]),
)
fmodel.run()
