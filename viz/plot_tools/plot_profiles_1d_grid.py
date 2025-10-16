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
from consts import np_INF, np_LARGE

def plot_profiles_1d_grid(coord_grid: tuple[tuple[np.ndarray]],
    profiles_grid: tuple[tuple[tuple[np.ndarray]]], file_path: str, **kwargs):

    ## NOT FINISHED

    ## Handle kwargs
    default_kwargs: dict = {"title" : None,
                            "xlabels_grid" : None,
                            "ylabels_grid" : None,
                            "profile_labels_grid" : None,
                            "title_grid" : None,
                            "xscale" : "linear",
                            "yscale" : "linear",
                            "sharex" : True,
                            "sharey" : True,
                            "coord_axis" : "x",
                            "viz" : "normal",
                            "draw_style" : "default"}

    kwargs: dict = {**default_kwargs, **kwargs}

    ## Assumptions about kwargs
    assert(kwargs["coord_axis"].lower() in ["x", "y"])
    assert(kwargs["viz"] in ["normal", "difference"])

    ## Set up the figure
    nrow: int = len(coord_grid)
    ncol: int = len(coord_grid[0])
    fig, axs = plt.subplots(nrow, ncol, sharex = kwargs["sharex"], sharey = kwargs["sharey"], layout = "constrained")

    if nrow == 1:
        axs: np.ndarray = np.expand_dims(axs, 0)

    ## Set axis bounds
    x_min: np_float = np_INF
    x_max: np_float = -np_INF
    y_min: np_float = np_INF
    y_max: np_float = -np_INF

    for ii in range(0, nrow):
        for jj in range(0, ncol):
            if kwargs["coord_axis"].lower() == "x":
                x_min: np_float = min(x_min, coord_grid[ii][jj].min())
                x_max: np_float = max(x_max, coord_grid[ii][jj].max())
            elif kwargs["coord_axis"].lower() == "y":
                y_min: np_float = min(y_min, coord_grid[ii][jj].min())
                y_max: np_float = max(y_max, coord_grid[ii][jj].max())

            for profile in profiles_grid[ii][jj]:
                if kwargs["coord_axis"].lower() == "x":
                    y_min: np_float = min(y_min, profile.min())
                    y_max: np_float = max(y_max, profile.max())
                elif kwargs["coord_axis"].lower() == "y":
                    x_min: np_float = min(x_min, profile.min())
                    x_max: np_float = max(x_max, profile.max())

    ## Styles for profiles
    colors: list = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77",
                    "#CC6677", "#AA4499", "#882255"]
    ncolors: int = len(colors)

    linestyles: list[str] = ["solid", "dashed", "dotted"]
    nlinestyles: int = len(linestyles)

    ## Plot the profile grid
    for ii in range(0, nrow):
        for jj in range(0, ncol):
            coord: np.ndarray = coord_grid[ii][jj]
            profiles: tuple[np.ndarray] = profiles_grid[ii][jj]
            if kwargs["profile_labels_grid"] is not None:
                if kwargs["profile_labels_grid"][ii][jj] is not None:
                    profile_labels: Optional[tuple[str]] = kwargs["profile_labels_grid"][ii][jj]
                else:
                    profile_labels: Optional[tuple[str]] = None
            else:
                profile_labels: Optional[tuple[str]] = None

            for idx in range(0, len(profiles)):
                profile: np.ndarray = profiles[idx]

                if profile_labels is not None:
                    label: Optional[str] = profile_labels[idx]
                else:
                    label: Optional[str] = None

                if kwargs["coord_axis"].lower() == "x":
                    x_data: np.ndarray = coord
                    y_data: np.ndarray = profile
                elif kwargs["coord_axis"].lower() == "y":
                    y_data: np.ndarray = coord
                    x_data: np.ndarray = profile

                axs[ii, jj].plot(x_data, y_data, color = colors[idx%ncolors],
                    linestyle = linestyles[idx%nlinestyles], label = label,
                    drawstyle = kwargs["draw_style"])

            if profile_labels is not None:
                axs[ii, jj].legend()

            ## If we are looking at a difference, add a gridline to guide the eye
            if kwargs["viz"] == "difference":
                if kwargs["coord_axis"].lower() == "x":
                    axs[ii, jj].hlines(0.0, -np_LARGE, np_LARGE, colors = "gray", linewidth = 0.2)
                elif kwargs["coord_axis"].lower() == "y":
                    axs[ii, jj].vlines(0.0, -np_LARGE, np_LARGE, colors = "gray", linewidth = 0.2)

            ## Set x- and y-axis bounds
            axs[ii, jj].set_xlim(x_min, x_max)
            axs[ii, jj].set_ylim(y_min, y_max)

            ## Set x- and y-scale
            axs[ii, jj].set_xscale(kwargs["xscale"])
            axs[ii, jj].set_yscale(kwargs["yscale"])

            ## Set plot title
            if kwargs["title_grid"] is not None:
                if kwargs["title_grid"][ii][jj] is not None:
                    title: str = kwargs["title_grid"][ii][jj]

                    axs[ii, jj].set_title(title, fontsize = "small")

    ## Label plot and axes
    if kwargs["xlabel"] is not None:
        fig.supxlabel(kwargs["xlabel"])

    if kwargs["ylabel"] is not None:
        fig.supylabel(kwargs["ylabel"])
    
    if kwargs["title"] is not None:
        fig.suptitle(kwargs["title"])

    plt.savefig(file_path, dpi = 300, bbox_inches = "tight")
    plt.close(fig)