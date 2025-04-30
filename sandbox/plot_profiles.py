# Standard Library Imports
import typing

# Third-Party Library Imports
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.contour import QuadContourSet
from matplotlib.colorbar import Colorbar
from mpl_toolkits.mplot3d.art3d import Path3DCollection

from matplotlib.colors import ListedColormap, to_rgba

# Local Library Imports
from consts import np_INF, np_LARGE


def plot_profiles_1d(coord: np.ndarray, profiles: list, file_path: str, **kwargs):
    ## Handle kwargs
    default_kwargs: dict = {"title" : None,
                            "xlabel" : None,
                            "ylabel" : None,
                            "profile_labels" : None,
                            "xscale" : "linear",
                            "yscale" : "linear",
                            "coord_axis" : "x",
                            "viz" : "normal",
                            "draw_style" : "default"}

    kwargs: dict = {**default_kwargs, **kwargs}

    ## Assumptions about kwargs
    assert(kwargs["coord_axis"].lower() in ["x", "y"])

    ## Hold variables for axis bounds
    x_min: np.float64 = np_INF
    x_max: np.float64 = -np_INF
    y_min: np.float64 = np_INF
    y_max: np.float64 = -np_INF

    ## Implement pre-loop arguments
    if kwargs["coord_axis"].lower() == "x":
        x_data: np.ndarray = coord
        x_min: np.float64 = np.min([x_min, x_data.min()])
        x_max: np.float64 = np.max([x_max, x_data.max()])
    elif kwargs["coord_axis"].lower() == "y":
        y_data: np.ndarray = coord
        y_min: np.float64 = np.min([y_min, y_data.min()])
        y_max: np.float64 = np.max([y_max, y_data.max()])

    ## Set up the figure
    fig, ax = plt.subplots(layout = "constrained")

    ## Plot the profiles
    colors: list = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77",
                    "#CC6677", "#AA4499", "#882255"]
    ncolors: int = len(colors)

    for idx in range(0, len(profiles)):
        profile: np.ndarray = profiles[idx]
        if kwargs["profile_labels"] is not None:
            label: str = kwargs["profile_labels"][idx]
        else:
            label: typing.Optional[str] = None

        if kwargs["coord_axis"].lower() == "x":
            y_data: np.ndarray = profile
            y_min: np.float64 = np.min([y_min, y_data.min()])
            y_max: np.float64 = np.max([y_max, y_data.max()])
        elif kwargs["coord_axis"].lower() == "y":
            x_data: np.ndarray = profile
            x_min: np.float64 = np.min([x_min, x_data.min()])
            x_max: np.float64 = np.max([x_max, x_data.max()])

        ax.plot(x_data, y_data, color = colors[idx%ncolors], label = label,
                drawstyle = kwargs["draw_style"])

    if kwargs["profile_labels"] is not None:
        ax.legend()

    ## If we are looking at a difference, add a gridline to guide the eye
    if kwargs["viz"] == "difference":
        if kwargs["coord_axis"].lower() == "x":
            ax.hlines(0.0, -np_LARGE, np_LARGE, colors = "gray", linewidth = 0.2)
        elif kwargs["coord_axis"].lower() == "y":
            ax.vlines(0.0, -np_LARGE, np_LARGE, colors = "gray", linewidth = 0.2)

    ## Set x- and y-axis bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ## Set x- and y-scale
    ax.set_xscale(kwargs["xscale"])
    ax.set_yscale(kwargs["yscale"])

    ## Label plot and axes
    if kwargs["xlabel"] is not None:
        ax.set_xlabel(kwargs["xlabel"])

    if kwargs["ylabel"] is not None:
        ax.set_ylabel(kwargs["ylabel"])
    
    if kwargs["title"] is not None:
        ax.set_title(kwargs["title"])


    plt.savefig(file_path, dpi = 300, bbox_inches = "tight")
    plt.close(fig)


def plot_profile_2d(meshgrid: tuple, profile: np.ndarray, file_path: str, **kwargs):
    ## Handle kwargs
    default_kwargs: dict = {"title" : None,
                            "xlabel" : None,
                            "ylabel" : None,
                            "cbarlabel" : None,
                            "xscale" : "linear",
                            "yscale" : "linear",
                            "cmap" : "Wistia",
                            "cscale" : "normal",
                            "draw_style" : "default"}

    kwargs: dict = {**default_kwargs, **kwargs}

    ## Set up the figure
    fig, ax = plt.subplots(layout = "constrained")

    ## Set colorbar levels and ticks
    ncbarticks: int = 7
    ncbarlevels: int = 128

    if kwargs["cscale"] == "normal":
        cmax: np.float64 = profile.max()
        cmin: np.float64 = profile.min()
    elif kwargs["cscale"] == "difference":
        cmax: np.float64 = np.abs(profile).max()
        cmin: np.float64 = -1. * cmax

    cbar_ticks: np.ndarray = np.linspace(cmin, cmax, ncbarticks)
    cbar_levels: np.ndarray = np.linspace(cmin, cmax, ncbarlevels)
    cbar_tick_labels: list = ["{:.2f}".format(tick) for tick in cbar_ticks]

    ## Plot the profile
    ctf: QuadContourSet = ax.contourf(meshgrid[0], meshgrid[1], profile,
                                      cmap = kwargs["cmap"], levels = cbar_levels,
                                      zorder = 0)
    ctf2: QuadContourSet = ax.contour(ctf, levels = cbar_ticks, colors = "black",
                                      linestyles = "--", linewidths = 0.5, zorder = 1)

    ## Set the colorbar
    cbar: Colorbar = fig.colorbar(ctf, ax = ax)
    cbar.ax.set_yticks(cbar_ticks, cbar_tick_labels)
    cbar.add_lines(ctf2)
    if kwargs["cbarlabel"] is not None:
        cbar.ax.set_ylabel(kwargs["cbarlabel"])

    ## Set x- and y-scale
    ax.set_xscale(kwargs["xscale"])
    ax.set_yscale(kwargs["yscale"])

    ## Label plot and axes
    if kwargs["xlabel"] is not None:
        ax.set_xlabel(kwargs["xlabel"])

    if kwargs["ylabel"] is not None:
        ax.set_ylabel(kwargs["ylabel"])
    
    if kwargs["title"] is not None:
        ax.set_title(kwargs["title"])

    plt.savefig(file_path, dpi = 300, bbox_inches = "tight")
    plt.close(fig)

def plot_profile_3d(meshgrid: tuple, profile: np.ndarray, file_path: str, **kwargs):
    ## Handle kwargs
    default_kwargs: dict = {"title" : None,
                            "xlabel" : None,
                            "ylabel" : None,
                            "zlabel" : None,
                            "cbarlabel" : None,
                            "xscale" : "linear",
                            "yscale" : "linear",
                            "zscale" : "linear",
                            "cmap" : "Wistia",
                            "draw_style" : "default"}

    kwargs: dict = {**default_kwargs, **kwargs}

    ## Set up the figure
    fig, ax = plt.subplots(layout = "constrained", subplot_kw = {"projection" : "3d"})

    ## Set colorbar and ticks
    ncbarticks: int = 7
    ncbarlevels: int = 256

    cbar_ticks: np.ndarray = np.linspace(profile.min(), profile.max(), ncbarticks)
    cbar_tick_labels: list = ["{:.1f}".format(tick) for tick in cbar_ticks]

    ## Plot the profile
    ### We plot the profile in sections of different alphas
    color: tuple = plt.get_cmap(kwargs["cmap"])(1.0)
    cmap: ListedColormap = transparent_colormap(color, ncbarlevels)
    ctf: Path3DCollection = \
        ax.scatter(meshgrid[0], meshgrid[1], meshgrid[2], c = profile, cmap = cmap,
                   vmin = profile.min(), vmax = profile.max())
    
    ## Set the axis limits
    ax.set_xlim([meshgrid[0].min(), meshgrid[0].max()])
    ax.set_ylim([meshgrid[1].min(), meshgrid[1].max()])
    ax.set_zlim([meshgrid[2].min(), meshgrid[2].max()])

    ## Set the colorbar
    cbar: Colorbar = fig.colorbar(ctf, ax = ax, pad = 0.1)
    cbar.ax.set_yticks(cbar_ticks, cbar_tick_labels)
    if kwargs["cbarlabel"] is not None:
        cbar.ax.set_ylabel(kwargs["cbarlabel"])

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

def transparent_colormap(color: tuple, N: int =256) -> ListedColormap:
    """Create a colormap that fades from `color` to fully transparent."""

    base_color : np.ndarray= np.array(to_rgba(color))  # RGBA tuple
    alphas: np.ndarray = np.linspace(0., 1., N)  # From opaque to transparent
    colors: np.ndarray = np.tile(base_color, (N, 1))
    colors[:, 3] = alphas  # Set alpha values

    return ListedColormap(colors)
