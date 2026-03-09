# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import numpy as np
import matplotlib.pyplot as plt

# Local Library Imports
from consts.consts import NP_REAL, NP_INT, NP_ARRAY, NP_INF, NP_EPS, NP_LARGE, \
    MPL_AXES, MPL_FIGURE

def plot_distribution(data: NP_ARRAY[NP_INT | NP_REAL], file_path: str,
    nbins: int | NP_INT = 10, **kwargs) -> None:
    
    default_kwargs: dict = {"xmin" : None,
        "xmax" : None,
        "ymax" : None,
        "title" : None,
        "xlabel" : None,
        "ylabel" : None,
        "xscale" : "linear",
        "yscale" : "linear",
        "density" : False,
        "tol" : 0.0}

    kwargs: dict = {**default_kwargs, **kwargs}

    ## Assertions on kwargs
    assert(kwargs["xscale"] in ["linear", "log"])

    ## Calculate the historgram
    xmin: NP_INT | NP_REAL
    xmax: NP_INT | NP_REAL
    if kwargs["xmin"] is not None:
        xmin = kwargs["xmin"]
    else:
        xmin = data.min()
    if kwargs["xmax"] is not None:
        xmax = kwargs["xmax"]
    else:
        xmax = data.max()
    assert(xmax > xmin)

    if kwargs["xscale"] == "linear":
        bins: NP_ARRAY[NP_REAL] = np.linspace(xmin, xmax, nbins,
            dtype = NP_REAL)
    elif kwargs["xscale"] == "log":
        assert(xmin > 0. and data.min() >= 0.0)
        bins: NP_ARRAY[NP_REAL] = np.logspace(np.log10(xmin), np.log10(xmax), nbins,
            dtype = NP_REAL)

    ymax: NP_REAL
    if kwargs["ymax"] is not None:
        ymax = kwargs["ymax"]
    else:
        hist: NP_ARRAY[NP_REAL]
        hist, _ = np.histogram(data, bins, density = kwargs["density"])
        ymax = hist.max()
    
    ## Set up the figure
    fig: MPL_FIGURE
    ax: MPL_AXES
    fig, ax = plt.subplots(layout = "constrained")

    ## Plot the profile
    _ = ax.hist(data.flatten(), bins, density = kwargs["density"], edgecolor = "black",
        facecolor = "white", zorder = 0)

    ## Include number of zero points if we are using a xscale = "log"
    if kwargs["xscale"] == "log":
        _ = ax.text(0.7, 0.95, "Zero Count: {}".format(np.sum(data <= NP_EPS)),
            transform = ax.transAxes)
    
    ## Plot the vertical line at the tolerance bound
    _ = ax.vlines(kwargs["tol"] * data.max(), ymin = 0.0, ymax = NP_LARGE,
        colors = "black", linewidths = 0.5, zorder = 1)

    ## Set the y-scale
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([0.0, 1.1 * ymax])

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