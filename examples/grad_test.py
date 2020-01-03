# import numpy as np
import autograd.numpy as np  # thinly-wrapped numpy
from autograd import grad    # the only autograd function you may ever need

# import matplotlib.pyplot as plt
import floris.tools as wfct

def grad_func(yaw_angles, fi):
    fi.calculate_wake(yaw_angles=yaw_angles)

    return fi.get_farm_power()

fi = wfct.floris_interface.FlorisInterface("example_input.json")

yaw_angles = [0.1, 0.1, 0.1]

# power = fi.get_farm_power_for_yaw_angle(yaw_angles=yaw_angles)

# yaw_grad = grad(fi.get_farm_power_for_yaw_angle)

yaw_grad = grad(grad_func)

print('--------------------------')
print('starting grad calculation')
print('--------------------------')
print(yaw_grad(yaw_angles, fi))

