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


import math
from itertools import product
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from typing import Union

def plot_turbines(
    ax, layout_x, layout_y, yaw_angles, D, color=None, wind_direction=270.0
):
    """
    Plot wind plant layout from turbine locations.

    Args:
        ax (:py:class:`matplotlib.pyplot.axes`): Figure axes.
        layout_x (np.array): Wind turbine locations (east-west).
        layout_y (np.array): Wind turbine locations (north-south).
        yaw_angles (np.array): Yaw angles of each wind turbine.
        D (float): Wind turbine rotor diameter.
        color (str): Pyplot color option to plot the turbines.
        wind_direction (float): Wind direction (rotates farm)
    """

    # Correct for the wind direction
    yaw_angles = np.array(yaw_angles) - wind_direction - 270

    if color is None:
        color = "k"
    for x, y, yaw in zip(layout_x, layout_y, yaw_angles):
        R = D / 2.0
        x_0 = x + np.sin(np.deg2rad(yaw)) * R
        x_1 = x - np.sin(np.deg2rad(yaw)) * R
        y_0 = y - np.cos(np.deg2rad(yaw)) * R
        y_1 = y + np.cos(np.deg2rad(yaw)) * R
        ax.plot([x_0, x_1], [y_0, y_1], color=color)


def plot_turbines_with_fi(ax, fi, color=None):
    """
    Wrapper function to plot turbines which extracts the data
    from a FLORIS interface object

    Args:
        ax (:py:class:`matplotlib.pyplot.axes`): figure axes. Defaults
            to None.
        fi (:py:class:`floris.tools.flow_data.FlowData`):
                FlowData object.
        color (str, optional): Color to plot turbines
    """
    # Grab D
    for i, turbine in enumerate(fi.floris.farm.turbines):
        D = turbine.rotor_diameter
        break

    plot_turbines(
        ax,
        fi.layout_x,
        fi.layout_y,
        fi.get_yaw_angles(),
        D,
        color=color,
        wind_direction=fi.floris.farm.wind_map.input_direction,
    )


def line_contour_cut_plane(cut_plane, ax=None, levels=None, colors=None, **kwargs):
    """
    Visualize a cut_plane as a line contour plot.

    Args:
        cut_plane (:py:class:`~.tools.cut_plane.CutPlane`):
            CutPlane Object.
        ax (:py:class:`matplotlib.pyplot.axes`): Figure axes. Defaults
            to None.
        levels (np.array, optional): Contour levels for plot.
            Defaults to None.
        colors (list, optional): Strings of color specification info.
            Defaults to None.
        **kwargs: Additional parameters to pass to `ax.contour`.
    """

    if not ax:
        fig, ax = plt.subplots()

    # Reshape UMesh internally
    x1_mesh = cut_plane.df.x1.values.reshape(
        cut_plane.resolution[1], cut_plane.resolution[0]
    )
    x2_mesh = cut_plane.df.x2.values.reshape(
        cut_plane.resolution[1], cut_plane.resolution[0]
    )
    u_mesh = cut_plane.df.u.values.reshape(
        cut_plane.resolution[1], cut_plane.resolution[0]
    )
    Zm = np.ma.masked_where(np.isnan(u_mesh), u_mesh)
    rcParams["contour.negative_linestyle"] = "solid"

    # # Plot the cut-through
    ax.contour(x1_mesh, x2_mesh, Zm, levels=levels, colors=colors, **kwargs)

    # Make equal axis
    ax.set_aspect("equal")


def visualize_cut_plane(
    cut_plane, ax=None, minSpeed=None, maxSpeed=None, cmap="coolwarm", levels=None
):
    """
    Generate pseudocolor mesh plot of the cut_plane.

    Args:
        cut_plane (:py:class:`~.tools.cut_plane.CutPlane`): 2D
            plane through wind plant.
        ax (:py:class:`matplotlib.pyplot.axes`): Figure axes. Defaults
            to None.
        minSpeed (float, optional): Minimum value of wind speed for
            contours. Defaults to None.
        maxSpeed (float, optional): Maximum value of wind speed for
            contours. Defaults to None.
        cmap (str, optional): Colormap specifier. Defaults to
            'coolwarm'.

    Returns:
        im (:py:class:`matplotlib.plt.pcolormesh`): Image handle.
    """

    if not ax:
        fig, ax = plt.subplots()
    if minSpeed is None:
        minSpeed = cut_plane.df.u.min()
    if maxSpeed is None:
        maxSpeed = cut_plane.df.u.max()

    # Reshape to 2d for plotting
    x1_mesh = cut_plane.df.x1.values.reshape(
        cut_plane.resolution[1], cut_plane.resolution[0]
    )
    x2_mesh = cut_plane.df.x2.values.reshape(
        cut_plane.resolution[1], cut_plane.resolution[0]
    )
    u_mesh = cut_plane.df.u.values.reshape(
        cut_plane.resolution[1], cut_plane.resolution[0]
    )
    Zm = np.ma.masked_where(np.isnan(u_mesh), u_mesh)

    # Plot the cut-through
    im = ax.pcolormesh(
        x1_mesh, x2_mesh, Zm, cmap=cmap, vmin=minSpeed, vmax=maxSpeed, shading="nearest"
    )

    # Add line contour
    line_contour_cut_plane(
        cut_plane, ax=ax, levels=levels, colors="w", linewidths=0.8, alpha=0.3
    )

    # Make equal axis
    ax.set_aspect("equal")

    # Return im
    return im


def visualize_quiver(
    cut_plane, ax=None, minSpeed=None, maxSpeed=None, downSamp=1, **kwargs
):
    """
        Visualize the in-plane flows in a cut_plane using quiver.

        Args:
            cut_plane (:py:class:`~.tools.cut_plane.CutPlane`): 2D
                plane through wind plant.
            ax (:py:class:`matplotlib.pyplot.axes`): Figure axes. Defaults
                to None.
            minSpeed (float, optional): Minimum value of wind speed for
                contours. Defaults to None.
            maxSpeed (float, optional): Maximum value of wind speed for
                contours. Defaults to None.
            downSamp (int, optional): Down sample the number of quiver arrows
                from underlying grid.
            **kwargs: Additional parameters to pass to `ax.streamplot`.

        Returns:
            im (:py:class:`matplotlib.plt.pcolormesh`): Image handle.
        """
    if not ax:
        fig, ax = plt.subplots()

    # Reshape UMesh internally
    x1_mesh = cut_plane.df.x1.values.reshape(
        cut_plane.resolution[1], cut_plane.resolution[0]
    )
    x2_mesh = cut_plane.df.x2.values.reshape(
        cut_plane.resolution[1], cut_plane.resolution[0]
    )
    v_mesh = cut_plane.df.v.values.reshape(
        cut_plane.resolution[1], cut_plane.resolution[0]
    )
    w_mesh = cut_plane.df.w.values.reshape(
        cut_plane.resolution[1], cut_plane.resolution[0]
    )

    # plot the stream plot
    ax.streamplot(
        (x1_mesh[::downSamp, ::downSamp]),
        (x2_mesh[::downSamp, ::downSamp]),
        v_mesh[::downSamp, ::downSamp],
        w_mesh[::downSamp, ::downSamp],
        # scale=80.0,
        # alpha=0.75,
        # **kwargs
    )

    # ax.quiverkey(QV1, -.75, -0.4, 1, '1 m/s', coordinates='data')

    # Make equal axis
    # ax.set_aspect('equal')


def reverse_cut_plane_x_axis_in_plot(ax):
    """
    Shortcut method to reverse direction of x-axis.

    Args:
        ax (:py:class:`matplotlib.pyplot.axes`): Figure axes.
    """
    ax.invert_xaxis()


def plot_rotor_values(
    values: np.ndarray,
    titles: np.ndarray = None,
    max_width: int = 4,
    cmap: str = "coolwarm",
    return_fig_objects: bool = False,
    save_path: Union[str, None] = None,
) -> Union[None, tuple[plt.figure, plt.axes]]:
    """Plots the gridded turbine rotor values. This is intended to be used for
    understanding the differences between two sets of values, so each subplot can be
    used for inspection of what values are differing, and under what conditions.

    Parameters:
        values (np.ndarray): The 5-dimensional array of values to plot. Should be:
            N wind directions x N wind speeds x N turbines X N rotor points X N rotor points.
        titles (None | np.ndarray): The string values to label each plot, and should be of the
            same shape as `values`, by default None.
        max_width (int): The maximum number of subplots in one row, default 4.
        cmap (str): The matplotlib colormap to be used, default "coolwarm".
        return_fig_objects (bool): Indicator to return the primary figure objects for
            further editing, default False.
        save_path (str | None): Where to save the figure, if a value is provided.

    Returns:
        None | tuple[plt.figure, plt.axes, plt.axis, plt.colorbar]: If
        `return_fig_objects` is `False, then `None` is returned`, otherwise the primary
        figure objects are returned for custom editing.
    """

    cmap = plt.cm.get_cmap(name=cmap)

    n_wd, n_ws, n_turb, *_ = values.shape

    vmin = values.min()
    vmax = values.max()

    rounded_vmin = round(math.floor(vmin) * 2) / 2
    if vmin % 1 >= 0.5:
        rounded_vmin += 0.5

    rounded_vmax = round(math.ceil(vmax) * 2) / 2
    if vmax % 1 <= 0.5:
        rounded_vmax -= 0.5

    bounds = np.linspace(rounded_vmin, rounded_vmax, 61)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    n_plots = n_wd * n_ws
    extra = 0
    if n_plots <= max_width:
        nrows, ncols = 1, n_plots
    else:
        nrows, extra = divmod(n_plots, max_width)
        if extra > 0:
            nrows += 1
            extra = max_width - extra
        ncols = max_width

    fig = plt.figure(dpi=200, figsize=(16, 16))
    axes = fig.subplots(nrows, ncols)

    # Create the title values if they aren't already existing
    indices = list(product(range(n_wd), range(n_ws), range(n_turb)))
    if titles is None:
        titles = np.array([f"Wind Direction: {i}\nWind Speed: {j}\n Turbine: {k}" for i, j, k in indices])
        titles = titles.reshape(n_wd, n_ws, n_turb)

    for ax, t, (i, j, k) in zip(axes.flatten(), titles.flatten(), indices):
        ax.imshow(values[i, j, k], cmap=cmap, norm=norm)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(t)

    if extra > 0:
        for ax in axes[-1][-extra:]:
            fig.delaxes(ax)

    cbar_ax = fig.add_axes([0.05, 0.125, 0.03, 0.75])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    if return_fig_objects:
        return fig, axes, cbar_ax, cb
