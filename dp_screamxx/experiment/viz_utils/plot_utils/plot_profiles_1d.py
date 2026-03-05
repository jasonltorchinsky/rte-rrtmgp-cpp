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
from utils.consts import NP_INF, NP_LARGE

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
    assert(kwargs["viz"] in ["normal", "difference"])

    ## Hold variables for axis bounds
    x_min: np.float64 = NP_INF
    x_max: np.float64 = -NP_INF
    y_min: np.float64 = NP_INF
    y_max: np.float64 = -NP_INF

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

    linestyles: list[str] = ["solid", "dashed", "dotted"]
    nlinestyles: int = len(linestyles)

    for idx in range(0, len(profiles)):
        profile: np.ndarray = profiles[idx]
        if kwargs["profile_labels"] is not None:
            label: str = kwargs["profile_labels"][idx]
        else:
            label: Optional[str] = None

        if kwargs["coord_axis"].lower() == "x":
            y_data: np.ndarray = profile
            y_min: np.float64 = np.min([y_min, y_data.min()])
            y_max: np.float64 = np.max([y_max, y_data.max()])
        elif kwargs["coord_axis"].lower() == "y":
            x_data: np.ndarray = profile
            x_min: np.float64 = np.min([x_min, x_data.min()])
            x_max: np.float64 = np.max([x_max, x_data.max()])

        ax.plot(x_data, y_data, color = colors[idx%ncolors], linestyle = linestyles[idx%nlinestyles],
                label = label, drawstyle = kwargs["draw_style"])

    if (kwargs["profile_labels"] not in [None, []]):
        ax.legend()

    ## If we are looking at a difference, add a gridline to guide the eye
    if kwargs["viz"] == "difference":
        if kwargs["coord_axis"].lower() == "x":
            ax.hlines(0.0, -NP_LARGE, NP_LARGE, colors = "gray", linewidth = 0.2)
        elif kwargs["coord_axis"].lower() == "y":
            ax.vlines(0.0, -NP_LARGE, NP_LARGE, colors = "gray", linewidth = 0.2)

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