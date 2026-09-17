# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from matplotlib.contour import QuadContourSet
from matplotlib.colorbar import Colorbar
from mpl_toolkits.mplot3d.art3d import Path3DCollection

from matplotlib.colors import ListedColormap, to_rgba

# Local Library Imports
from consts.consts import NP_INF, NP_LARGE

def plot_profile_2d_3d(meshgrid_2d: tuple, profile_2d: np.ndarray,
                        meshgrid_3d: tuple, profile_3d: np.ndarray, 
                        file_path: str, **kwargs):
    ### NOTE: We assume here that we are plotting nonnegative quanitities, where
    ### values closer to zero are transparent. The tol keyword is to avoid
    ### plotting transparent or nearly-transparent points.
    ## Handle kwargs
    default_kwargs: dict = {"title" : None,
                            "xlabel" : None,
                            "ylabel" : None,
                            "zlabel" : None,
                            "cbarlabel_2d" : None,
                            "cbarlabel_3d" : None,
                            "zdir" : "z",
                            "xscale" : "linear",
                            "yscale" : "linear",
                            "zscale" : "linear",
                            "cmin_2d" : None,
                            "cmax_2d" : None,
                            "cmin_3d" : None,
                            "cmax_3d" : None,
                            "cmap_2d" : "afmhot",
                            "cmap_3d" : "Reds",
                            "draw_style" : "default",
                            "tol" : 0.0,
                            "alpha" : 1.0}

    kwargs: dict = {**default_kwargs, **kwargs}

    ## Assertions on kwargs
    assert(kwargs["zdir"] in ["x", "y", "z"])

    ## Set up the figure
    fig, ax = plt.subplots(layout = "constrained", subplot_kw = {"projection" : "3d"})

    ## Disable computed zorder
    ax.computed_zorder = False

    ## Mask out values of 3D profile that are too small
    mask_3d: np.ndarray = (profile_3d > kwargs["tol"] * profile_3d.max())
    plt_meshgrid_3d: tuple = meshgrid_3d[:]
    for ii in range(0, 3):
        plt_meshgrid_3d[ii] = meshgrid_3d[ii][mask_3d]
    plt_profile_3d: np.ndarray = profile_3d[mask_3d]

    ## Set 2D and 3D colorbar and ticks
    ncbarticks: int = 7
    ncbarlevels: int = 256

    if kwargs["cmin_2d"] is not None:
        cmin_2d: float = kwargs["cmin_2d"]
    else:
        cmin_2d: np.float64 = profile_2d.min()
    
    if kwargs["cmax_2d"] is not None:
        cmax_2d: float = kwargs["cmax_2d"]
    else:
        cmax_2d: np.float64 = profile_2d.max()

    cbar_ticks_2d: np.ndarray = np.linspace(cmin_2d, cmax_2d, ncbarticks)
    cbar_levels_2d: np.ndarray = np.linspace(cmin_2d, cmax_2d, ncbarlevels)
    cbar_tick_labels_2d: list = ["{:1.3e}".format(tick) for tick in cbar_ticks_2d]

    if kwargs["cmin_3d"] is not None:
        cmin_3d: float = kwargs["cmin_3d"]
    else:
        cmin_3d: np.float64 = plt_profile_3d.min()
    
    if kwargs["cmax_3d"] is not None:
        cmax_3d: float = kwargs["cmax_3d"]
    else:
        cmax_3d: np.float64 = plt_profile_3d.max()

    cbar_ticks_3d: np.ndarray = np.linspace(cmin_3d, cmax_3d, ncbarticks)
    cbar_tick_labels_3d: list = ["{:1.3e}".format(tick) for tick in cbar_ticks_3d]

    ## Plot the 2d profile
    ctf_2d: QuadContourSet = ax.contourf(meshgrid_2d[0], meshgrid_2d[1], profile_2d, 
                                         zdir = kwargs["zdir"], offset = 0.0, zorder = 0,
                                         levels = cbar_levels_2d, cmap = kwargs["cmap_2d"],
                                         vmin = cmin_2d, vmax = cmax_2d)
    ctf_2d2: QuadContourSet = ax.contour(meshgrid_2d[0], meshgrid_2d[1], profile_2d,
                                         zdir = kwargs["zdir"], offset = 0.0, zorder = 1,
                                         levels = cbar_ticks_2d, colors = "black",
                                         linestyles = "--", linewidths = 0.5,
                                         vmin = cmin_2d, vmax = cmax_2d)
    
    ### Set the 2D colorbar
    cbar_2d: Colorbar = fig.colorbar(ctf_2d, ax = ax, pad = 0.0, location = "left")
    cbar_2d.ax.set_yticks(cbar_ticks_2d, cbar_tick_labels_2d)
    cbar_2d.add_lines(ctf_2d2)
    if kwargs["cbarlabel_2d"] is not None:
        cbar_2d.ax.set_ylabel(kwargs["cbarlabel_2d"])

    ## Plot the 3-D profile
    ctf_3d: Path3DCollection = \
        ax.scatter(plt_meshgrid_3d[0], plt_meshgrid_3d[1], plt_meshgrid_3d[2], c = plt_profile_3d,
                   cmap = kwargs["cmap_3d"], vmin = cmin_3d, vmax = cmax_3d, alpha = kwargs["alpha"], zorder = 2)

    ### Set the 3D colorbar
    cbar_3d: Colorbar = fig.colorbar(ctf_3d, ax = ax, pad = 0.15, location = "right")
    cbar_3d.solids.set(alpha = 1.0)
    cbar_3d.ax.set_yticks(cbar_ticks_3d, cbar_tick_labels_3d)
    if kwargs["cbarlabel_3d"] is not None:
        cbar_3d.ax.set_ylabel(kwargs["cbarlabel_3d"])

    ## Set the axis limits
    if kwargs["zdir"] == "x":
        x_min: float = meshgrid_3d[0].min()
        x_max: float = meshgrid_3d[0].max()

        y_min: float = min(meshgrid_2d[0].min(), meshgrid_3d[1].min())
        y_max: float = max(meshgrid_2d[0].max(), meshgrid_3d[1].max())

        z_min: float = min(meshgrid_2d[1].min(), meshgrid_3d[2].min())
        z_max: float = max(meshgrid_2d[1].max(), meshgrid_3d[2].max())
    elif kwargs["zdir"] == "y":
        x_min: float = min(meshgrid_2d[0].min(), meshgrid_3d[0].min())
        x_max: float = max(meshgrid_2d[0].max(), meshgrid_3d[0].max())

        y_min: float = meshgrid_3d[1].min()
        y_max: float = meshgrid_3d[1].max()

        z_min: float = min(meshgrid_2d[1].min(), meshgrid_3d[2].min())
        z_max: float = max(meshgrid_2d[1].max(), meshgrid_3d[2].max())
    elif kwargs["zdir"] == "z":
        x_min: float = min(meshgrid_2d[0].min(), meshgrid_3d[0].min())
        x_max: float = max(meshgrid_2d[0].max(), meshgrid_3d[0].max())

        y_min: float = min(meshgrid_2d[1].min(), meshgrid_3d[1].min())
        y_max: float = max(meshgrid_2d[1].max(), meshgrid_3d[1].max())

        z_min: float = meshgrid_3d[2].min()
        z_max: float = meshgrid_3d[2].max()
    

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    ## Set x-, y-, and z-scales
    ax.set_xscale(kwargs["xscale"])
    ax.set_yscale(kwargs["yscale"])
    ax.set_yscale(kwargs["zscale"])

    ## Label plot and axes
    if kwargs["xlabel"] is not None:
        ax.set_xlabel(kwargs["xlabel"])

    if kwargs["ylabel"] is not None:
        ax.set_ylabel(kwargs["ylabel"])

    if kwargs["zlabel"] is not None:
        ax.set_zlabel(kwargs["zlabel"])
    
    if kwargs["title"] is not None:
        ax.set_title(kwargs["title"])

    plt.savefig(file_path, dpi = 300, bbox_inches = "tight")
    plt.close(fig)