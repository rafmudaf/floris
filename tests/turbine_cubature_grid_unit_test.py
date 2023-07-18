
import numpy as np
import pytest

from floris.utilities import Vec3
from tests.conftest import (
    N_TURBINES,
    N_WIND_DIRECTIONS,
    N_WIND_SPEEDS,
    TURBINE_GRID_RESOLUTION,
)


def test_cubaturegrid(cubature_grid_fixture):
    print(cubature_grid_fixture)
