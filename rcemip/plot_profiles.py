# Standard Library Imports
import typing

# Third-Party Library Imports
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.contour import QuadContourSet
from matplotlib.colorbar import Colorbar

# Local Library Imports


def plot_profiles_1d(coord: np.ndarray, profiles: list, file_path: str, **kwargs):
    ## Handle kwargs
    default_kwargs: dict = {"title" : None,
                            "xlabel" : None,
                            "ylabel" : None,
                            "profile_labels" : None,
                            "xscale" : "linear",
                            "yscale" : "linear",
                            "coord_axis" : "x",
                            "drawstyle" : "default"}

    kwargs: dict = {**default_kwargs, **kwargs}

    ## Assumptions about kwargs
    assert(kwargs["coord_axis"].lower() in ["x", "y"])

    ## Implement pre-loop arguments
    if kwargs["coord_axis"].lower() == "x":
        x_data: np.ndarray = coord
    elif kwargs["coord_axis"].lower() == "y":
        y_data: np.ndarray = coord

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
        elif kwargs["coord_axis"].lower() == "y":
            x_data: np.ndarray = profile

        ax.plot(x_data, y_data, color = colors[idx%ncolors], label = label,
                drawstyle = kwargs["drawstyle"])

    if kwargs["profile_labels"] is not None:
        ax.legend()

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

import numpy as np
import matplotlib.pyplot as plt
import typing


def plot_profile_2d(meshgrid: tuple, profile: np.ndarray, file_path: str, **kwargs):
    ## Handle kwargs
    default_kwargs: dict = {"title" : None,
                            "xlabel" : None,
                            "ylabel" : None,
                            "cbarlabel" : None,
                            "xscale" : "linear",
                            "yscale" : "linear",
                            "cmap" : "Wistia",
                            "drawstyle" : "default"}

    kwargs: dict = {**default_kwargs, **kwargs}

    ## Set up the figure
    fig, ax = plt.subplots(layout = "constrained")

    ## Set colorbar levels and ticks
    ncbarticks: int = 7
    ncbarlevels: int = 128

    cbar_ticks: np.ndarray = np.linspace(profile.min(), profile.max(), ncbarticks)
    cbar_levels: np.ndarray = np.linspace(profile.min(), profile.max(), ncbarlevels)
    cbar_tick_labels: list = ["{:.1f}".format(tick) for tick in cbar_ticks]

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
