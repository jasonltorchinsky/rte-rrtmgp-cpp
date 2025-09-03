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

def plot_distribution(a: np.ndarray, file_path: str, nbins: int = 10, **kwargs) -> None:
    
    default_kwargs: dict = {"title" : None,
                            "xlabel" : None,
                            "ylabel" : None,
                            "zlabel" : None,
                            "cbarlabel" : None,
                            "xscale" : "linear",
                            "yscale" : "linear",
                            "cmin" : None,
                            "cmax" : None,
                            "cmap" : "afmhot",
                            "tol" : 0.0}

    kwargs: dict = {**default_kwargs, **kwargs}

    ## Assertions on kwargs
    assert(kwargs["xscale"] in ["linear", "log"])

    ## Set up the figure
    fig, ax = plt.subplots(layout = "constrained")

    if kwargs["xscale"] == "linear":
        bins: np.ndarray = np.linspace(a.min(), a.max(), nbins)
    elif kwargs["xscale"] == "log":
        assert(a.min() >= 0.0)
        bins: np.ndarray = np.logspace(np.log10(a[a > 0.0].min()), np.log10(a.max()), nbins)

    hist: np.ndarray
    hist, _ = np.histogram(a, bins)

    ## Plot the profile
    _: tuple = ax.hist(a.flatten(), bins, edgecolor = "black", facecolor = "white", zorder = 0)

    ## Include number of zero points if we are using a xscale = "log"
    if kwargs["xscale"] == "log":
        _ = ax.text(0.7, 0.95, "Zero Count: {}".format(np.sum(a == 0.0)),
                    transform = ax.transAxes)
    
    ## Plot the vertical line at the tolerance bound
    _: LineCollection = ax.vlines(kwargs["tol"] * a.max(), ymin = 0.0, ymax = np.sqrt(np_INF),
                                       colors = "black", linewidths = 0.5, zorder = 1)

    ## Set the y-scale
    ax.set_xlim([bins.min(), bins.max()])
    ax.set_ylim([0, 1.1 * hist.max()])

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