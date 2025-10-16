# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt


from matplotlib.colorbar import Colorbar
from matplotlib.contour import QuadContourSet

# Local Library Imports


def plot_profile_2d(meshgrid: tuple, profile: np.ndarray, file_path: str, **kwargs):
    ## Handle kwargs
    default_kwargs: dict = {"title" : None,
                            "xlabel" : None,
                            "ylabel" : None,
                            "cbarlabel" : None,
                            "xscale" : "linear",
                            "yscale" : "linear",
                            "cmin" : None,
                            "cmax" : None,
                            "cmap" : "afmhot",
                            "cscale" : "normal",
                            "draw_style" : "default"}

    kwargs: dict = {**default_kwargs, **kwargs}

    ## kwargs asserts
    assert(kwargs["cscale"] in ["normal", "difference", "log"])

    ## Set up the figure
    fig, ax = plt.subplots(layout = "constrained")

    ## Set colorbar levels and ticks
    ncbarticks: int = 7
    ncbarlevels: int = 256

    if kwargs["cmin"] is None:
        if kwargs["cscale"] in ["normal", "log"]:
            cmin: np.float64 = profile.min()
    else:
        cmin: float = kwargs["cmin"]

    if kwargs["cmax"] is None:
        if kwargs["cscale"] in ["normal", "log"]:
            cmax: np.float64 = profile.max()
    else:
        cmax: float = kwargs["cmax"]

    if (kwargs["cmin"] is None) and (kwargs["cmax"] is None):
        if kwargs["cscale"] in ["difference"]:
            cmax: np.float64 = np.abs(profile).max()
            cmin: np.float64 = -1. * cmax

    if kwargs["cscale"] in ["log"]:
        norm: Optional[colors.Normalize] = colors.LogNorm(vmin = cmin, vmax = cmax)
        cbar_ticks: np.ndarray = np.logspace(np.log10(cmin), np.log10(cmax), ncbarticks)
        cbar_levels: np.ndarray = np.logspace(np.log10(cmin), np.log10(cmax), ncbarlevels)
        cbar_tick_labels: list = ["{:1.3e}".format(tick) for tick in cbar_ticks]
    else:
        norm: Optional[colors.Normalize] = None
        cbar_ticks: np.ndarray = np.linspace(cmin, cmax, ncbarticks)
        cbar_levels: np.ndarray = np.linspace(cmin, cmax, ncbarlevels)
        cbar_tick_labels: list = ["{:1.3e}".format(tick) for tick in cbar_ticks]

    ## Plot the profile
    ctf: QuadContourSet = ax.contourf(meshgrid[0], meshgrid[1], profile,
        cmap = kwargs["cmap"], levels = cbar_levels, norm = norm, zorder = 0)
    ctf2: QuadContourSet = ax.contour(ctf, levels = cbar_ticks, colors = "black",
        linestyles = "--", linewidths = 0.5, norm = norm, zorder = 1)

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