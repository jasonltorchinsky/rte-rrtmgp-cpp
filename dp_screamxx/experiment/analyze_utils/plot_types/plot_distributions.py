# Standard Library Imports
from typing import Sequence

# Third-Party Library Imports
import numpy as np
import matplotlib.pyplot as plt

# Local Library Imports
from consts.consts import NP_REAL, NP_INT, NP_ARRAY, NP_EPS, NP_LARGE, \
    MPL_AXES, MPL_FIGURE


def plot_distributions(
    data: NP_ARRAY[NP_INT | NP_REAL] | Sequence[NP_ARRAY[NP_INT | NP_REAL]],
    file_path: str,
    nbins: int | NP_INT = 10,
    **kwargs
) -> None:
    """
    - Multiple datasets supported (single array or sequence of arrays).
    - Plots only the top line of histogram bins (step line).
    - kwarg "data_labels": labels per dataset (enables legend).
    - kwarg "ignore_zeros": if True, remove values <= NP_EPS before binning.
    - kwarg "normalize": if True, normalize each dataset histogram so sum(hist)=1
      (probability mass per bin). This is different from density=True (pdf).
    - Default line colors cycle through a fixed palette.
    """

    colors: list = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77",
                    "#CC6677", "#AA4499", "#882255"]
    ncolors: int = len(colors)

    default_kwargs: dict = {
        "xmin": None,
        "xmax": None,
        "ymax": None,
        "title": None,
        "xlabel": None,
        "ylabel": None,
        "xscale": "linear",
        "yscale": "linear",
        "density": False,        # passed to np.histogram unless normalize=True
        "normalize": False,      # NEW
        "tol": 0.0,
        "data_labels": None,
        "linewidth": 1.5,
        "ignore_zeros": False,
    }
    kwargs: dict = {**default_kwargs, **kwargs}

    assert kwargs["xscale"] in ["linear", "log"]
    if kwargs["normalize"] and kwargs["density"]:
        raise ValueError("Use either normalize=True (PMF) or density=True (PDF), not both.")

    # Normalize input to a list of 1D arrays
    if isinstance(data, (list, tuple)):
        original_datasets = [np.asarray(d).flatten() for d in data]
    else:
        original_datasets = [np.asarray(data).flatten()]

    assert len(original_datasets) > 0
    assert all(d.size > 0 for d in original_datasets)

    # Optionally remove zeros (and near-zeros)
    if kwargs["ignore_zeros"]:
        datasets = [d[d > NP_EPS] for d in original_datasets]
        assert all(d.size > 0 for d in datasets), \
            "One or more datasets empty after ignore_zeros filtering."
    else:
        datasets = original_datasets

    labels = kwargs["data_labels"]
    if labels is not None:
        assert len(labels) == len(datasets), "data_labels must match number of datasets"

    # Determine global xmin/xmax unless user specifies
    all_data = np.concatenate(datasets)
    xmin: NP_INT | NP_REAL = kwargs["xmin"] if kwargs["xmin"] is not None else all_data.min()
    xmax: NP_INT | NP_REAL = kwargs["xmax"] if kwargs["xmax"] is not None else all_data.max()
    assert xmax > xmin

    # Build bins
    if kwargs["xscale"] == "linear":
        bins: NP_ARRAY[NP_REAL] = np.linspace(xmin, xmax, nbins, dtype=NP_REAL)
    else:
        assert xmin > 0.0 and all_data.min() >= 0.0
        bins: NP_ARRAY[NP_REAL] = np.logspace(np.log10(xmin), np.log10(xmax), nbins, dtype=NP_REAL)

    # Helper to compute histogram with requested normalization
    def _histogram(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if kwargs["normalize"]:
            counts, edges = np.histogram(d, bins, density=False)
            total = float(np.sum(counts))
            hist = (counts / total) if total > 0.0 else counts.astype(float)
            return hist, edges
        else:
            hist, edges = np.histogram(d, bins, density=kwargs["density"])
            return hist.astype(float), edges

    # Compute ymax if not provided (max across datasets)
    if kwargs["ymax"] is not None:
        ymax: NP_REAL = kwargs["ymax"]
    else:
        ymax = 0.0
        for d in datasets:
            hist, _ = _histogram(d)
            ymax = max(ymax, float(hist.max()) if hist.size else 0.0)

    # Set up the figure
    fig: MPL_FIGURE
    ax: MPL_AXES
    fig, ax = plt.subplots(layout="constrained")

    # Plot only the "top line" of the distribution bins for each dataset
    for i, d in enumerate(datasets):
        hist, edges = _histogram(d)
        label = labels[i] if labels is not None else None
        color = colors[i % ncolors]

        ax.step(
            edges,
            np.r_[hist, hist[-1]] if hist.size else np.array([0.0]),
            where="post",
            linewidth=kwargs["linewidth"],
            color=color,
            label=label,
            zorder=2,
        )

    # Include number of zero points if xscale="log" and not ignoring zeros
    if kwargs["xscale"] == "log" and (not kwargs["ignore_zeros"]):
        zero_counts = [int(np.sum(d <= NP_EPS)) for d in original_datasets]
        msg = "Zero Count(s): " + ", ".join(map(str, zero_counts))
        _ = ax.text(0.7, 0.95, msg, transform=ax.transAxes)

    # Plot the vertical line at the tolerance bound (based on global max)
    _ = ax.vlines(kwargs["tol"] * all_data.max(), ymin=0.0, ymax=NP_LARGE,
                  colors="black", linewidths=0.5, zorder=1)

    # Axes limits/scales
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([0.0, 1.1 * ymax if ymax > 0 else 1.0])

    ax.set_xscale(kwargs["xscale"])
    ax.set_yscale(kwargs["yscale"])

    # Labels
    if kwargs["xlabel"] is not None:
        ax.set_xlabel(kwargs["xlabel"])
    if kwargs["ylabel"] is not None:
        ax.set_ylabel(kwargs["ylabel"])
    if kwargs["title"] is not None:
        ax.set_title(kwargs["title"])

    if labels is not None:
        ax.legend()

    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)