# import numpy as np
import autograd.numpy as np  # thinly-wrapped numpy
from autograd import grad    # the only autograd function you may ever need

# import matplotlib.pyplot as plt
import floris.tools as wfct

fi = wfct.floris_interface.FlorisInterface("example_input.json")

yaw_angles = [1., 10.]

power = fi.get_farm_power_for_yaw_angle(yaw_angles=yaw_angles)

yaw_grad = grad(fi.get_farm_power_for_yaw_angle)

print('innnnnnnnnnnnn')
print(yaw_grad(yaw_angles))

