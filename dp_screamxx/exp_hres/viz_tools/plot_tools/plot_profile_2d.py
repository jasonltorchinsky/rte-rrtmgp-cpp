# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt

# Local Library Imports
from utils.conts import NP_REAL, NP_ARRAY, MPL_AXES, MPL_COLORBAR, MPL_FIGURE, \
    MPL_CONTOUR


def plot_profile_2d(meshgrid: list[NP_ARRAY[NP_REAL]], profile: NP_ARRAY[NP_REAL],
    file_path: str, **kwargs):
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
        "plot_style" : "contour"}

    kwargs: dict = {**default_kwargs, **kwargs}

    ## kwargs asserts
    assert(kwargs["cscale"] in ["normal", "difference", "log"])

    ## Set up the figure
    fig: MPL_FIGURE
    ax: MPL_AXES
    fig, ax = plt.subplots(layout = "constrained")

    ## Set colorbar levels and ticks
    ncbarticks: NP_INT = NP_INT(7)
    ncbarlevels: NP_INT = NP_INT(256)

    if kwargs["cmin"] is None:
        if kwargs["cscale"] in ["normal", "log"]:
            cmin: NP_REAL = profile.min()
    else:
        cmin: NP_REAL = NP_REAL(kwargs["cmin"])

    if kwargs["cmax"] is None:
        if kwargs["cscale"] in ["normal", "log"]:
            cmax: NP_REAL = profile.max()
    else:
        cmax: NP_REAL = NP_REAL(kwargs["cmax"])

    if (kwargs["cmin"] is None) and (kwargs["cmax"] is None):
        if kwargs["cscale"] in ["difference"]:
            cmax: NP_REAL = np.abs(profile).max()
            cmin: NP_REAL = -1. * cmax

    if kwargs["cscale"] in ["log"]:
        norm: Optional[colors.Normalize] = colors.LogNorm(vmin = cmin, vmax = cmax)
        cbar_ticks: NP_ARRAY[NP_REAL] = np.logspace( \
            np.log10(cmin), np.log10(cmax), ncbarticks, dtype = NP_REAL)
        cbar_levels: NP_ARRAY[NP_REAL] = np.logspace( \
            np.log10(cmin), np.log10(cmax), ncbarlevels, dtype = NP_REAL)
        cbar_tick_labels: list[str] = ["{:1.3e}".format(tick) for tick in cbar_ticks]
    else:
        norm: Optional[colors.Normalize] = None
        cbar_ticks: NP_ARRAY[NP_REAL] = np.linspace( \
            cmin, cmax, ncbarticks, dtype = NP_REAL)
        cbar_levels: NP_ARRAY[NP_REAL] = np.linspace( \
            cmin, cmax, ncbarlevels, dtype = NP_REAL)
        cbar_tick_labels: list[str] = ["{:1.3e}".format(tick) for tick in cbar_ticks]

    ## Plot the profile
    if kwargs["plot_style"] == "colormesh":
        ctf: MPL_CONTOUR = ax.pcolormesh(meshgrid[0], meshgrid[1], profile,
            cmap = kwargs["cmap"], levels = cbar_levels, norm = norm, zorder = 0)
    else: # default to "contour"
        ctf: MPL_CONTOUR = ax.contourf(meshgrid[0], meshgrid[1], profile,
            cmap = kwargs["cmap"], levels = cbar_levels, norm = norm, zorder = 0)
        ctf2: MPL_CONTOUR = ax.contour(ctf, levels = cbar_ticks, colors = "black",
            linestyles = "--", linewidths = 0.5, norm = norm, zorder = 1)

    ## Set the colorbar
    cbar: MPL_COLORBAR = fig.colorbar(ctf, ax = ax)
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