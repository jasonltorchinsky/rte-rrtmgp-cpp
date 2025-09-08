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
                            "alpha" : 1.0,
                            "draw_style" : "default",
                            "tol" : 0.0}

    kwargs: dict = {**default_kwargs, **kwargs}

    ## Set up the figure
    fig, ax = plt.subplots(layout = "constrained", subplot_kw = {"projection" : "3d"})

    ## Mask out values that are too small
    mask: np.ndarray = (profile > kwargs["tol"] * profile.max())
    plt_meshgrid: tuple = meshgrid[:]
    for ii in range(0, 3):
        plt_meshgrid[ii] = meshgrid[ii][mask]
    plt_profile: np.ndarray = profile[mask]

    ## Set colorbar and ticks
    ncbarticks: int = 7
    ncbarlevels: int = 256

    cbar_ticks: np.ndarray = np.linspace(plt_profile.min(), plt_profile.max(), ncbarticks)
    cbar_tick_labels: list = ["{:1.3e}".format(tick) for tick in cbar_ticks]

    ## Plot the profile
    ### We plot the profile in sections of different alphas
    ctf: Path3DCollection = \
        ax.scatter(plt_meshgrid[0], plt_meshgrid[1], plt_meshgrid[2], c = plt_profile,
                   cmap = kwargs["cmap"], vmin = plt_profile.min(), vmax = plt_profile.max(),
                   alpha = kwargs["alpha"])
    
    ## Set the axis limits
    ax.set_xlim([meshgrid[0].min(), meshgrid[0].max()])
    ax.set_ylim([meshgrid[1].min(), meshgrid[1].max()])
    ax.set_zlim([meshgrid[2].min(), meshgrid[2].max()])

    ## Set the colorbar
    cbar: Colorbar = fig.colorbar(ctf, ax = ax, pad = 0.1)
    cbar.solids.set(alpha = 1.0)
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