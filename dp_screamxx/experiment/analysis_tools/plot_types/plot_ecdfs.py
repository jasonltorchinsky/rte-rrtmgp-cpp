# Standard Library Imports
from typing import Sequence

# Third-Party Library Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ecdf, ks_2samp  # SciPy >= 1.11

# Local Library Imports
from consts.consts import NP_REAL, NP_INT, NP_ARRAY, NP_EPS, NP_LARGE, \
    MPL_AXES, MPL_FIGURE


def plot_ecdfs(
    data: NP_ARRAY[NP_INT | NP_REAL] | Sequence[NP_ARRAY[NP_INT | NP_REAL]],
    file_path: str,
    **kwargs
) -> None:
    """
    - Multiple datasets supported (single array or sequence of arrays).
    - Plots empirical CDFs computed by scipy.stats.ecdf via ax.plot.
    - kwarg "data_labels": labels per dataset (enables legend).
    - kwarg "ignore_zeros": if True, remove values <= NP_EPS before ECDF.
    - If kwarg "calculate_kolmogorov_smirnov" is True AND there are exactly two
      datasets, compute the two-sample Kolmogorov–Smirnov statistic and display
      it in a text box at the bottom-right of the plot.
    """

    colors: list = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77",
                    "#CC6677", "#AA4499", "#882255"]
    ncolors: int = len(colors)

    default_kwargs: dict = {
        "xmin": None,
        "xmax": None,
        "ymin": 0.0,
        "ymax": 1.0,
        "title": None,
        "xlabel": None,
        "ylabel": "Empirical CDF",
        "xscale": "linear",
        "yscale": "linear",
        "tol": 0.0,
        "data_labels": None,
        "linewidth": 1.5,
        "ignore_zeros": False,
        "draw_points": False,
        "markersize": 2.5,
        "calculate_kolmogorov_smirnov": False,
        "ecdf_complementary": False,  # if True, plot 1-F(x)
        "ks_text_fmt": r"Kolmogorov-Smirnov: {stat:.4g}",  # formatting for annotation
        "ks_show_pvalue": False,
        "ks_pvalue_fmt": r"p: {pvalue:.4g}",
    }
    kwargs: dict = {**default_kwargs, **kwargs}

    assert kwargs["xscale"] in ["linear", "log"]
    assert kwargs["yscale"] in ["linear", "log"]

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

    # Set up the figure
    fig: MPL_FIGURE
    ax: MPL_AXES
    fig, ax = plt.subplots(layout="constrained")

    # Plot ECDF for each dataset using scipy.stats.ecdf + ax.plot
    for i, d in enumerate(datasets):
        res = ecdf(d.astype(float))
        x = np.asarray(res.cdf.quantiles, dtype=float)
        y = np.asarray(res.cdf.probabilities, dtype=float)
        if kwargs["ecdf_complementary"]:
            y = 1.0 - y

        label = labels[i] if labels is not None else None
        color = colors[i % ncolors]

        ax.plot(
            x,
            y,
            linewidth=kwargs["linewidth"],
            color=color,
            label=label,
            zorder=2,
        )

        if kwargs["draw_points"]:
            ax.plot(
                x,
                y,
                linestyle="none",
                marker=".",
                markersize=kwargs["markersize"],
                color=color,
                zorder=3,
            )

    # Optional KS statistic (only defined/implemented here for exactly 2 datasets)
    if kwargs["calculate_kolmogorov_smirnov"]:
        if len(datasets) != 2:
            raise ValueError(
                "calculate_kolmogorov_smirnov=True requires exactly two datasets."
            )

        # Two-sample KS statistic: max_x |F1(x) - F2(x)|
        ks_res = ks_2samp(
            datasets[0].astype(float),
            datasets[1].astype(float),
            alternative="two-sided",
            method="auto",
        )

        lines = [kwargs["ks_text_fmt"].format(stat=float(ks_res.statistic))]
        if kwargs["ks_show_pvalue"]:
            lines.append(kwargs["ks_pvalue_fmt"].format(pvalue=float(ks_res.pvalue)))
        text = "\n".join(lines)

        ax.text(
            0.98, 0.02, text,
            transform=ax.transAxes,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", alpha=0.85),
            zorder=10,
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
    ax.set_ylim([kwargs["ymin"], kwargs["ymax"]])
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