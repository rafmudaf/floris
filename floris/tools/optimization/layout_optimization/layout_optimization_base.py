
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Polygon

from floris.tools import TimeSeries
from floris.tools.optimization.yaw_optimization.yaw_optimizer_geometric import (
    YawOptimizationGeometric,
)
from floris.tools.wind_data import WindDataBase

from ....logging_manager import LoggingManager


class LayoutOptimization(LoggingManager):
    """
    Base class for layout optimization. This class should not be used directly
    but should be subclassed by a specific optimization method.

    Args:
        fi (FlorisInterface): A FlorisInterface object.
        boundaries (iterable(float, float)): Pairs of x- and y-coordinates
            that represent the boundary's vertices (m).
        wind_data (TimeSeries | WindRose): A TimeSeries or WindRose object
            values.
        min_dist (float, optional): The minimum distance to be maintained
            between turbines during the optimization (m). If not specified,
            initializes to 2 rotor diameters. Defaults to None.
        enable_geometric_yaw (bool, optional): If True, enables geometric yaw
            optimization. Defaults to False.
    """
    def __init__(self, fi, boundaries, wind_data, min_dist=None, enable_geometric_yaw=False):
        self.fi = fi.copy()
        self.boundaries = boundaries
        self.enable_geometric_yaw = enable_geometric_yaw

        self._boundary_polygon = Polygon(self.boundaries)
        self._boundary_line = LineString(self.boundaries)

        self.xmin = np.min([tup[0] for tup in boundaries])
        self.xmax = np.max([tup[0] for tup in boundaries])
        self.ymin = np.min([tup[1] for tup in boundaries])
        self.ymax = np.max([tup[1] for tup in boundaries])

        # If no minimum distance is provided, assume a value of 2 rotor diamters
        if min_dist is None:
            self.min_dist = 2 * self.rotor_diameter
        else:
            self.min_dist = min_dist

        # Check that wind_data is a WindDataBase object
        if (not isinstance(wind_data, WindDataBase)):
            raise ValueError(
                "wind_data entry is not an object of WindDataBase"
                " (eg TimeSeries, WindRose, WindTIRose)"
            )
        self.wind_data = wind_data

        # Establish geometric yaw class
        if self.enable_geometric_yaw:
            self.yaw_opt = YawOptimizationGeometric(
                fi,
                minimum_yaw_angle=-30.0,
                maximum_yaw_angle=30.0,
            )

        self.initial_AEP = fi.get_farm_AEP_with_wind_data(self.wind_data)

    def __str__(self):
        return "layout"

    def _norm(self, val, x1, x2):
            return (val - x1) / (x2 - x1)

    def _unnorm(self, val, x1, x2):
        return np.array(val) * (x2 - x1) + x1

    def _get_geoyaw_angles(self):
        # NOTE: requires that child class saves x and y locations
        # as self.x and self.y and updates them during optimization.
        if self.enable_geometric_yaw:
            self.yaw_opt.fi_subset.set(layout_x=self.x, layout_y=self.y)
            df_opt = self.yaw_opt.optimize()
            self.yaw_angles = np.vstack(df_opt['yaw_angles_opt'])[:, :]
        else:
            self.yaw_angles = None

        return self.yaw_angles

    # Public methods

    def optimize(self):
        sol = self._optimize()
        return sol

    def plot_layout_opt_results(self):
        x_initial, y_initial, x_opt, y_opt = self._get_initial_and_final_locs()

        plt.figure(figsize=(9, 6))
        fontsize = 16
        plt.plot(x_initial, y_initial, "ob")
        plt.plot(x_opt, y_opt, "or")
        # plt.title('Layout Optimization Results', fontsize=fontsize)
        plt.xlabel("x (m)", fontsize=fontsize)
        plt.ylabel("y (m)", fontsize=fontsize)
        plt.axis("equal")
        plt.grid()
        plt.tick_params(which="both", labelsize=fontsize)
        plt.legend(
            ["Old locations", "New locations"],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            fontsize=fontsize,
        )

        verts = self.boundaries
        for i in range(len(verts)):
            if i == len(verts) - 1:
                plt.plot([verts[i][0], verts[0][0]], [verts[i][1], verts[0][1]], "b")
            else:
                plt.plot(
                    [verts[i][0], verts[i + 1][0]], [verts[i][1], verts[i + 1][1]], "b"
                )


    ###########################################################################
    # Properties
    ###########################################################################

    @property
    def nturbs(self):
        """
        This property returns the number of turbines in the FLORIS
        object.

        Returns:
            nturbs (int): The number of turbines in the FLORIS object.
        """
        self._nturbs = self.fi.floris.farm.n_turbines
        return self._nturbs

    @property
    def rotor_diameter(self):
        return self.fi.floris.farm.rotor_diameters_sorted[0][0]
