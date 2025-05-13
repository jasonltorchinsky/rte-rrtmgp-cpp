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
                            "cmap" : "afmhot",
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
    cbar_tick_labels: list = ["{:.3f}".format(tick) for tick in cbar_ticks]

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
    ### NOTE: We assume here that we are plotting nonnegative quanitities, where
    ### values closer to zero are transparent. The tol keyword is to avoid
    ### plotting transparent or nearly-transparent points.
    ## Handle kwargs
    default_kwargs: dict = {"title" : None,
                            "xlabel" : None,
                            "ylabel" : None,
                            "zlabel" : None,
                            "cbarlabel" : None,
                            "xscale" : "linear",
                            "yscale" : "linear",
                            "zscale" : "linear",
                            "cmap" : "afmhot",
                            "draw_style" : "default",
                            "tol" : 0.0}

    kwargs: dict = {**default_kwargs, **kwargs}

    ## Set up the figure
    fig, ax = plt.subplots(layout = "constrained", subplot_kw = {"projection" : "3d"})

    ## Set colorbar and ticks
    ncbarticks: int = 7
    ncbarlevels: int = 256

    cbar_ticks: np.ndarray = np.linspace(profile.min(), profile.max(), ncbarticks)
    cbar_tick_labels: list = ["{:.3f}".format(tick) for tick in cbar_ticks]

    ## Mask out values that are too small
    mask: np.ndarray = (profile >= kwargs["tol"] * profile.max())
    plt_meshgrid: tuple = meshgrid[:]
    for ii in range(0, 3):
        plt_meshgrid[ii] = meshgrid[ii][mask]
    plt_profile: np.ndarray = profile[mask]

    ## Plot the profile
    ### We plot the profile in sections of different alphas
    color: tuple = plt.get_cmap(kwargs["cmap"])(1.0)
    cmap: ListedColormap = transparent_colormap(color, ncbarlevels)
    ctf: Path3DCollection = \
        ax.scatter(plt_meshgrid[0], plt_meshgrid[1], plt_meshgrid[2], c = plt_profile,
                   cmap = cmap, vmin = profile.min(), vmax = profile.max())
    
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

def plot_profiles_2d_3d(meshgrid_2d: tuple, profile_2d: np.ndarray,
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
                            "tol" : 0.0}

    kwargs: dict = {**default_kwargs, **kwargs}

    ## Assertion on kwargs
    assert(kwargs["zdir"] in ["x", "y", "z"])

    ## Set up the figure
    fig, ax = plt.subplots(layout = "constrained", subplot_kw = {"projection" : "3d"})

    ## Disable computed zorder
    ax.computed_zorder = False

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
    cbar_tick_labels_2d: list = ["{:.3f}".format(tick) for tick in cbar_ticks_2d]

    if kwargs["cmin_3d"] is not None:
        cmin_3d: float = kwargs["cmin_3d"]
    else:
        cmin_3d: np.float64 = profile_3d.min()
    
    if kwargs["cmax_3d"] is not None:
        cmax_3d: float = kwargs["cmax_3d"]
    else:
        cmax_3d: np.float64 = profile_3d.max()

    cbar_ticks_3d: np.ndarray = np.linspace(cmin_3d, cmax_3d, ncbarticks)
    cbar_tick_labels_3d: list = ["{:.3f}".format(tick) for tick in cbar_ticks_3d]

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
    ### Mask out values that are too small
    mask_3d: np.ndarray = (profile_3d >= kwargs["tol"] * cmax_3d)
    plt_meshgrid_3d: tuple = meshgrid_3d[:]
    for ii in range(0, 3):
        plt_meshgrid_3d[ii] = meshgrid_3d[ii][mask_3d]
    plt_profile_3d: np.ndarray = profile_3d[mask_3d]

    ### We plot the profile in sections of different alphas
    color: tuple = plt.get_cmap(kwargs["cmap_3d"])(1.0)
    cmap: ListedColormap = transparent_colormap(color, ncbarlevels)
    ctf_3d: Path3DCollection = \
        ax.scatter(plt_meshgrid_3d[0], plt_meshgrid_3d[1], plt_meshgrid_3d[2], c = plt_profile_3d,
                   cmap = cmap, vmin = cmin_2d, vmax = cmax_3d, zorder = 2)

    ### Set the 3D colorbar
    cbar_3d: Colorbar = fig.colorbar(ctf_3d, ax = ax, pad = 0.15, location = "right")
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

def transparent_colormap(color: tuple, N: int = 256) -> ListedColormap:
    """Create a colormap that fades from `color` to fully transparent."""

    base_color : np.ndarray= np.array(to_rgba(color))  # RGBA tuple
    alphas: np.ndarray = np.linspace(0., 1., N)  # From opaque to transparent
    colors: np.ndarray = np.tile(base_color, (N, 1))
    colors[:, 3] = alphas  # Set alpha values

    return ListedColormap(colors)

