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
from __future__ import annotations

import attr
import numpy as np
import numpy.typing as npt
from sklearn import neighbors
from attr.setters import convert
from numpy.lib.function_base import iterable

from floris.utilities import Vec3, attrs_array_converter


NDArrayFloat = npt.NDArray[np.float64]


@attr.s(auto_attribs=True)
class FlowData:
    """
    FlowData objects represent a saved 3D flow from a FLORIS simulation
    or other data source.

    Args:
        x (np.array): Cartesian coordinate data.
        y (np.array): Cartesian coordinate data.
        z (np.array): Cartesian coordinate data.
        u (np.array): x-component of velocity.
        v (np.array): y-component of velocity.
        w (np.array): z-component of velocity.
        spacing (Vec3, optional): Spatial resolution.
            Defaults to None.
        dimensions (Vec3, optional): Named dimensions
            (e.g. x1, x2, x3). Defaults to None.
        origin (Vec3, optional): Coordinates of origin.
            Defaults to None.
    """

    x: NDArrayFloat = attr.ib(converter=attrs_array_converter)
    y: NDArrayFloat = attr.ib(converter=attrs_array_converter)
    z: NDArrayFloat = attr.ib(converter=attrs_array_converter)
    u: NDArrayFloat = attr.ib(converter=attrs_array_converter)
    v: NDArrayFloat = attr.ib(converter=attrs_array_converter)
    w: NDArrayFloat = attr.ib(converter=attrs_array_converter)
    spacing: Vec3 | None = None
    dimensions: Vec3 | None = None
    origin: Vec3 | None = None
    resolution: Vec3 = attr.ib(init=False)

    # TODO handle none case, maybe default values apply like 0 origin and auto
    # determine spacing and dimensions
    def __attrs_post_init__(self) -> None:
        # Technically resolution is a restating of above, but it is useful to have
        self.resolution = Vec3(self.x.unique().size, self.y.unique().size, self.z.unique().size)

    def save_as_vtk(self, filename):
        """
        Save FlowData Object to vtk format.

        Args:
            filename (str): Write-to path for vtk file.
        """
        n_points = self.dimensions.x1 * self.dimensions.x2 * self.dimensions.x3

        ln = "\n"
        with open(filename, "w") as vtk_file:
            vtk_file.write(f"# vtk DataFile Version 3.0{ln}")
            vtk_file.write(f"array.mean0D{ln}")
            vtk_file.write(f"ASCII{ln}")
            vtk_file.write(f"DATASET STRUCTURED_POINTS{ln}")
            vtk_file.write(f"DIMENSIONS {self.dimensions}{ln}")
            vtk_file.write(f"ORIGIN {self.origin.x1} {self.origin.x2} {self.origin.x3}{ln}")
            vtk_file.write(f"SPACING {self.spacing}{ln}")
            vtk_file.write(f"POINT_DATA {n_points}{ln}")
            vtk_file.write(f"FIELD attributes 1{ln}")
            vtk_file.write(f"UAvg 3 {n_points} float{ln}")
            for u, v, w in zip(self.u, self.v, self.w):
                vtk_file.write(f"{Vec3(u, v, w)}{ln}")

    @staticmethod
    def crop(ff: "FlowData", x_bnds: iterable, y_bnds: iterable, z_bnds: iterable) -> "FlowData":
        """
        Crop FlowData object to within stated bounds.

        Args:
            ff (:py:class:`~.tools.flow_data.FlowData`):
                FlowData object.
            x_bnds (iterable): Min and max of x-coordinate.
            y_bnds (iterable): Min and max of y-coordinate.
            z_bnds (iterable): Min and max of z-coordinate.

        Returns:
            (:py:class:`~.tools.flow_data.FlowData`):
            Cropped FlowData object.
        """

        map_values = (
            (ff.x > x_bnds[0])
            & (ff.x < x_bnds[1])
            & (ff.y > y_bnds[0])
            & (ff.y < y_bnds[1])
            & (ff.z > z_bnds[0])
            & (ff.z < z_bnds[1])
        )

        x = ff.x[map_values]
        y = ff.y[map_values]
        z = ff.z[map_values]

        #  Work out new dimensions
        dimensions = Vec3(len(np.unique(x)), len(np.unique(y)), len(np.unique(z)))

        # Work out origin
        origin = Vec3(
            ff.origin.x1 + np.min(x),
            ff.origin.x2 + np.min(y),
            ff.origin.x3 + np.min(z),
        )

        return FlowData(
            x - np.min(x),
            y - np.min(y),
            z - np.min(z),
            ff.u[map_values],
            ff.v[map_values],
            ff.w[map_values],
            spacing=ff.spacing,  # doesn't change
            dimensions=dimensions,
            origin=origin,
        )

        # Define a quick function for getting arbitrary points from sowfa

    def get_points_from_flow_data(
        self, x_points: NDArrayFloat, y_points: NDArrayFloat, z_points: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Return the u-value of a set of points from with a FlowData object.
        Use a simple nearest neighbor regressor to do internal interpolation.

        Args:
            x_points (NDArrayFloat): Array of x-locations of points.
            y_points (NDArrayFloat): Array of y-locations of points.
            z_points (NDArrayFloat): Array of z-locations of points.

        Returns:
            NDArrayFloat: Array of u-velocity at specified points.
        """
        # print(x_points,y_points,z_points)
        # X = np.column_stack([self.x, self.y, self.z])
        n_neighbors = 1
        knn = neighbors.KNeighborsRegressor(n_neighbors)
        # y_ = knn.fit(X, self.u)  # .predict(T)

        # Predict new points
        T = np.column_stack([x_points, y_points, z_points])
        return knn.predict(T)
