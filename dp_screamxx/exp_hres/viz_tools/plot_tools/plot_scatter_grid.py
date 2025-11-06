# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import PathCollection

# Local Library Imports
from utils.consts import NP_REAL, NP_INF, NP_LARGE

def plot_scatter_grid(xdata_grid: tuple[tuple[tuple[np.ndarray]]],
    ydata_grid: tuple[tuple[tuple[np.ndarray]]], file_path: str, **kwargs):
    
    ## Handle kwargs
    default_kwargs: dict = {"title" : None,
                            "xlabel" : None,
                            "ylabel" : None,
                            "subplot_label_grid" : None, 
                            "data_labels_grid" : None,
                            "xscale" : "linear",
                            "yscale" : "linear",
                            "xlim" : None,
                            "ylim" : None,
                            "show_idenitity" : False,
                            "figsize" : None}

    kwargs: dict = {**default_kwargs, **kwargs}

    ## kwargs asserts

    ## Set up the figure
    nrow: int = len(xdata_grid)
    ncol: int = len(xdata_grid[0])
    fig, axs = plt.subplots(nrow, ncol, sharex = True, sharey = True, layout = "constrained")
    
    if nrow == 1:
        axs: np.ndarray = np.expand_dims(axs, 0)

    ## Set axis bounds
    ### If xlim or ylim not set, determine automatically
    if kwargs["xlim"] is not None:
        x_min: NP_REAL = kwargs["xlim"][0]
        x_max: NP_REAL = kwargs["xlim"][1]
    else:
        x_min: NP_REAL = NP_INF
        x_max: NP_REAL = -NP_INF
    
    if kwargs["ylim"] is not None:
        y_min: NP_REAL = kwargs["ylim"][0]
        y_max: NP_REAL = kwargs["ylim"][1]
    else:
        y_min: NP_REAL = NP_INF
        y_max: NP_REAL = -NP_INF

    if ((kwargs["xlim"] is None) or (kwargs["ylim"] is None)):
        for ii in range(0, nrow):
            for jj in range(0, ncol):
                for kk in range(0, len(xdata_grid[ii][jj])):
                    if (kwargs["xlim"] is None):
                        x_min: NP_REAL = min(x_min, xdata_grid[ii][jj][kk].min())
                        x_max: NP_REAL = max(x_max, xdata_grid[ii][jj][kk].max())

                    if (kwargs["ylim"] is None):
                        y_min: NP_REAL = min(y_min, ydata_grid[ii][jj][kk].min())
                        y_max: NP_REAL = max(y_max, ydata_grid[ii][jj][kk].max())

    ## Styles for profiles
    colors: list = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77",
                    "#CC6677", "#AA4499", "#882255"]
    ncolors: int = len(colors)

    markerstyles: list[str] = ["o", "v", "s", "*", "d"]
    nmarkerstyles: int = len(markerstyles)

    ## Plot the profile_grid
    for ii in range(0, nrow):
        for jj in range(0, ncol):
            if kwargs["data_labels_grid"] is not None:
                if kwargs["data_labels_grid"][ii][jj] is not None:
                    data_labels: Optional[tuple[str]] = kwargs["data_labels_grid"][ii][jj]
                else:
                    data_labels: Optional[tuple[str]] = None
            else:
                data_labels: Optional[tuple[str]] = None

            for kk in range(0, len(xdata_grid[ii][jj])):
                xdata: np.ndarray = xdata_grid[ii][jj][kk]
                ydata: np.ndarray = ydata_grid[ii][jj][kk]
                if data_labels is not None:
                    data_label: Optional[str] = data_labels[kk]
                else:
                    data_label: Optional[str] = None

                axs[ii, jj].scatter(xdata, ydata,
                    s = 10, c = colors[kk % ncolors],
                    marker = markerstyles[kk % nmarkerstyles], label = data_label)

            if kwargs["show_identity"]:
                axs[ii, jj].plot([x_min, x_max], [y_min, y_max], color = "black", linestyle = "solid")

            if data_labels is not None:
                axs[ii, jj].legend()

            ## Set xlim and ylim
            axs[ii, jj].set_xlim([x_min, x_max])
            axs[ii, jj].set_ylim([y_min, y_max])

            ## Set x- and y-scale
            axs[ii, jj].set_xscale(kwargs["xscale"])
            axs[ii, jj].set_yscale(kwargs["yscale"])

            ## Label plot
            if kwargs["subplot_label_grid"] is not None:
                axs[ii, jj].set_title(kwargs["subplot_label_grid"][ii][jj], fontsize = "small")

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