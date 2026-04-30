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
from consts.consts import NP_INF, NP_LARGE

def plot_profile_2d_grid(meshgrid_grid: tuple, profile_grid: tuple, file_path: str, **kwargs):
    ## Handle kwargs
    default_kwargs: dict = {"title" : None,
                            "xlabel" : None,
                            "ylabel" : None,
                            "cbarlabel" : None,
                            "profile_label_grid" : None,
                            "xscale" : "linear",
                            "yscale" : "linear",
                            "cmin" : None,
                            "cmax" : None,
                            "cmap" : "afmhot",
                            "cscale" : "normal",
                            "draw_style" : "default",
                            "figsize" : None}

    kwargs: dict = {**default_kwargs, **kwargs}

    ## kwargs asserts
    assert(kwargs["cscale"] in ["normal", "difference"])

    ## Set up the figure
    nrow: int = len(meshgrid_grid)
    ncol: int = len(meshgrid_grid[0])
    fig, axs = plt.subplots(nrow, ncol, sharex = True, sharey = True, layout = "constrained")

    ## Set colorbar levels and ticks
    ncbarticks: int = 7
    ncbarlevels: int = 128

    if kwargs["cmin"] is None:
        if kwargs["cscale"] == "normal":
            cmin: np.float64 = NP_INF
            for ii in range(0, nrow):
                for jj in range(0, ncol):
                    cmin: np.float64 = min(cmin, profile_grid[ii][jj].min())
    else:
        cmin: float = kwargs["cmin"]

    if kwargs["cmax"] is None:
        if kwargs["cscale"] == "normal":
            cmax: np.float64 = -NP_INF
            for ii in range(0, nrow):
                for jj in range(0, ncol):
                    cmax: np.float64 = max(cmax, profile_grid[ii][jj].max())
    else:
        cmax: float = kwargs["cmax"]

    if (kwargs["cmin"] is None) and (kwargs["cmax"] is None):
        if kwargs["cscale"] == "difference":
            cmax: np.float64 = -NP_INF
            for ii in range(0, nrow):
                for jj in range(0, ncol):
                    cmax: np.float64 = max(cmax, np.abs(profile_grid[ii][jj]).max())
            cmin: np.float64 = -1. * cmax

    cbar_ticks: np.ndarray = np.linspace(cmin, cmax, ncbarticks)
    cbar_levels: np.ndarray = np.linspace(cmin, cmax, ncbarlevels)
    cbar_tick_labels: list = ["{:1.3e}".format(tick) for tick in cbar_ticks]

    ## Plot the profile_grid
    for ii in range(0, nrow):
        for jj in range(0, ncol):
            ctf: QuadContourSet = axs[ii, jj].contourf(meshgrid_grid[ii][jj][0], meshgrid_grid[ii][jj][1], profile_grid[ii][jj],
                                                       cmap = kwargs["cmap"], levels = cbar_levels,
                                                       vmin = cmin, vmax = cmax, zorder = 0)
            ctf2: QuadContourSet = axs[ii, jj].contour(ctf, levels = cbar_ticks, colors = "black",
                                                       linestyles = "--", linewidths = 0.5, zorder = 1)

            ## Set x- and y-scale
            axs[ii, jj].set_xscale(kwargs["xscale"])
            axs[ii, jj].set_yscale(kwargs["yscale"])

            ## Label plot
            if kwargs["profile_label_grid"] is not None:
                axs[ii, jj].set_title(kwargs["profile_label_grid"][ii][jj], fontsize = "small")

    ## Set the colorbar
    cbar: Colorbar = fig.colorbar(ctf, ax = axs)
    cbar.ax.set_yticks(cbar_ticks, cbar_tick_labels)
    cbar.add_lines(ctf2)
    if kwargs["cbarlabel"] is not None:
        cbar.ax.set_ylabel(kwargs["cbarlabel"])

    ## Label plot and axes
    if kwargs["xlabel"] is not None:
        fig.supxlabel(kwargs["xlabel"])

    if kwargs["ylabel"] is not None:
        fig.supylabel(kwargs["ylabel"])
    
    if kwargs["title"] is not None:
        fig.suptitle(kwargs["title"])

    if kwargs["figsize"] is not None:
        fig.set_figwidth(kwargs["figsize"][0])
        fig.set_figheight(kwargs["figsize"][1])
        
    plt.savefig(file_path, dpi = 300, bbox_inches = "tight")
    plt.close(fig)