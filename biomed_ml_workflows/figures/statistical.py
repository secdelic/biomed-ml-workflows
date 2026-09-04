"""Reusable statistical and image plotting functions.

All functions consume explicit, already-computed values and return a
``(Figure, Axes)`` pair. They never fit models, load data, or save files.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle


FigureResult = tuple[Figure, Axes | np.ndarray]


def _array(name: str, values: Any, *, ndim: int | tuple[int, ...] | None = None) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    allowed = (ndim,) if isinstance(ndim, int) else ndim
    if allowed is not None and array.ndim not in allowed:
        raise ValueError(f"{name} must have {allowed} dimensions; got {array.ndim}.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return array


def _numeric(name: str, values: Any, *, ndim: int | tuple[int, ...] | None = None) -> np.ndarray:
    array = _array(name, values, ndim=ndim)
    try:
        result = array.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _labels(labels: Sequence[str] | None, count: int, prefix: str) -> list[str]:
    if labels is None:
        return [f"{prefix} {index + 1}" for index in range(count)]
    result = [str(value) for value in labels]
    if len(result) != count:
        raise ValueError(f"labels must contain {count} values.")
    return result


def _finish(ax: Axes, *, title: str | None, xlabel: str | None, ylabel: str | None) -> None:
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def plot_line(
    x: Any,
    y: Any,
    *,
    labels: Sequence[str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> FigureResult:
    """Plot one or more numeric series; ``x`` is [N], ``y`` is [N] or [S,N]."""
    x_array = _numeric("x", x, ndim=1)
    y_array = _numeric("y", y, ndim=(1, 2))
    series = y_array[np.newaxis, :] if y_array.ndim == 1 else y_array
    if series.shape[1] != x_array.size:
        raise ValueError("The last y dimension must equal len(x).")
    names = _labels(labels, series.shape[0], "Series")
    fig, ax = plt.subplots(constrained_layout=True)
    for name, values in zip(names, series, strict=True):
        ax.plot(x_array, values, label=name if series.shape[0] > 1 or labels else None)
    if series.shape[0] > 1 or labels:
        ax.legend(loc="best")
    _finish(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    return fig, ax


def plot_scatter(
    x: Any,
    y: Any,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> FigureResult:
    """Plot paired observations; ``x`` and ``y`` must both be [N]."""
    x_array = _numeric("x", x, ndim=1)
    y_array = _numeric("y", y, ndim=1)
    if x_array.size != y_array.size:
        raise ValueError("x and y must have equal lengths.")
    fig, ax = plt.subplots(constrained_layout=True)
    ax.scatter(x_array, y_array)
    _finish(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    return fig, ax


def plot_histogram(
    values: Any,
    *,
    labels: Sequence[str] | None = None,
    bins: int = 10,
    title: str | None = None,
) -> FigureResult:
    """Plot a histogram for [N] data or separate histograms for [N,P] columns."""
    array = _numeric("values", values, ndim=(1, 2))
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer.")
    matrix = array[:, np.newaxis] if array.ndim == 1 else array
    names = _labels(labels, matrix.shape[1], "Variable")
    ncols = min(3, matrix.shape[1])
    nrows = ceil(matrix.shape[1] / ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, constrained_layout=True)
    flat = axes.ravel()
    for index, name in enumerate(names):
        flat[index].hist(matrix[:, index], bins=bins)
        flat[index].set_title(name)
        flat[index].set_ylabel("Count")
    for ax in flat[matrix.shape[1] :]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title)
    return fig, axes if matrix.shape[1] > 1 else flat[0]


def plot_density(
    values: Any,
    *,
    labels: Sequence[str] | None = None,
    bandwidth: float | None = None,
    title: str | None = None,
) -> FigureResult:
    """Plot Gaussian-kernel densities for [N] or [S,N] prevalidated samples."""
    array = _numeric("values", values, ndim=(1, 2))
    series = array[np.newaxis, :] if array.ndim == 1 else array
    if series.shape[1] < 2:
        raise ValueError("Each density series requires at least two observations.")
    names = _labels(labels, series.shape[0], "Series")
    minimum, maximum = float(series.min()), float(series.max())
    if minimum == maximum:
        raise ValueError("Density data must contain variation.")
    grid = np.linspace(minimum, maximum, 256)
    fig, ax = plt.subplots(constrained_layout=True)
    for name, sample in zip(names, series, strict=True):
        resolved = bandwidth
        if resolved is None:
            std = float(np.std(sample, ddof=1))
            resolved = 1.06 * std * sample.size ** (-0.2)
        if resolved is None or resolved <= 0:
            raise ValueError("bandwidth must be positive.")
        scaled = (grid[:, None] - sample[None, :]) / resolved
        density = np.exp(-0.5 * scaled**2).mean(axis=1) / (resolved * np.sqrt(2 * np.pi))
        ax.plot(grid, density, label=name if series.shape[0] > 1 or labels else None)
    if series.shape[0] > 1 or labels:
        ax.legend(loc="best")
    _finish(ax, title=title, xlabel="Value", ylabel="Density")
    return fig, ax


def plot_bar(
    categories: Sequence[Any],
    values: Any,
    *,
    series_labels: Sequence[str] | None = None,
    stacked: bool = False,
    title: str | None = None,
    ylabel: str | None = None,
) -> FigureResult:
    """Plot simple, grouped, or stacked bars; values are [K] or [S,K]."""
    category_names = [str(value) for value in categories]
    if not category_names:
        raise ValueError("categories must not be empty.")
    array = _numeric("values", values, ndim=(1, 2))
    series = array[np.newaxis, :] if array.ndim == 1 else array
    if series.shape[1] != len(category_names):
        raise ValueError("The last values dimension must equal len(categories).")
    names = _labels(series_labels, series.shape[0], "Series")
    positions = np.arange(len(category_names), dtype=float)
    fig, ax = plt.subplots(constrained_layout=True)
    if stacked:
        baseline = np.zeros(len(category_names))
        for name, row in zip(names, series, strict=True):
            ax.bar(positions, row, bottom=baseline, label=name if series.shape[0] > 1 else None)
            baseline = baseline + row
    else:
        width = 0.8 / series.shape[0]
        for index, (name, row) in enumerate(zip(names, series, strict=True)):
            offset = (index - (series.shape[0] - 1) / 2) * width
            ax.bar(positions + offset, row, width=width, label=name if series.shape[0] > 1 else None)
    ax.set_xticks(positions, category_names)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    if series.shape[0] > 1:
        ax.legend(loc="best")
    _finish(ax, title=title, xlabel=None, ylabel=ylabel)
    return fig, ax


def plot_boxplot(
    groups: Sequence[Any], *, labels: Sequence[str] | None = None, title: str | None = None
) -> FigureResult:
    """Plot distributions supplied as a non-empty sequence of one-dimensional groups."""
    arrays = [_numeric(f"groups[{index}]", values, ndim=1) for index, values in enumerate(groups)]
    if not arrays:
        raise ValueError("groups must not be empty.")
    names = _labels(labels, len(arrays), "Group")
    fig, ax = plt.subplots(constrained_layout=True)
    ax.boxplot(arrays, tick_labels=names)
    _finish(ax, title=title, xlabel=None, ylabel="Value")
    return fig, ax


def plot_violin(
    groups: Sequence[Any], *, labels: Sequence[str] | None = None, title: str | None = None
) -> FigureResult:
    """Plot violin distributions from a non-empty sequence of one-dimensional groups."""
    arrays = [_numeric(f"groups[{index}]", values, ndim=1) for index, values in enumerate(groups)]
    if not arrays:
        raise ValueError("groups must not be empty.")
    if any(values.size < 2 for values in arrays):
        raise ValueError("Each violin group requires at least two observations.")
    names = _labels(labels, len(arrays), "Group")
    fig, ax = plt.subplots(constrained_layout=True)
    ax.violinplot(arrays, showmedians=True)
    ax.set_xticks(np.arange(1, len(arrays) + 1), names)
    _finish(ax, title=title, xlabel=None, ylabel="Value")
    return fig, ax


def plot_correlation_heatmap(
    values: Any,
    *,
    labels: Sequence[str] | None = None,
    matrix_is_correlation: bool = False,
    annotate: bool = True,
    title: str | None = None,
) -> FigureResult:
    """Plot a [P,P] correlation matrix or compute it from an [N,P] data matrix."""
    array = _numeric("values", values, ndim=2)
    matrix = array if matrix_is_correlation else np.corrcoef(array, rowvar=False)
    matrix = np.atleast_2d(matrix)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A correlation matrix must be square.")
    names = _labels(labels, matrix.shape[0], "Variable")
    fig, ax = plt.subplots(constrained_layout=True)
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(names)), names)
    if annotate:
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                ax.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Correlation")
    _finish(ax, title=title, xlabel=None, ylabel=None)
    return fig, ax


def plot_joint_distribution(
    x: Any,
    y: Any,
    *,
    bins: int = 15,
    xlabel: str = "x",
    ylabel: str = "y",
) -> FigureResult:
    """Plot paired [N] observations with marginal histograms."""
    x_array = _numeric("x", x, ndim=1)
    y_array = _numeric("y", y, ndim=1)
    if x_array.size != y_array.size:
        raise ValueError("x and y must have equal lengths.")
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer.")
    fig = plt.figure(constrained_layout=True)
    grid = fig.add_gridspec(4, 4)
    joint = fig.add_subplot(grid[1:, :3])
    top = fig.add_subplot(grid[0, :3], sharex=joint)
    right = fig.add_subplot(grid[1:, 3], sharey=joint)
    joint.scatter(x_array, y_array)
    top.hist(x_array, bins=bins)
    right.hist(y_array, bins=bins, orientation="horizontal")
    joint.set_xlabel(xlabel)
    joint.set_ylabel(ylabel)
    top.tick_params(labelbottom=False)
    right.tick_params(labelleft=False)
    return fig, np.asarray([joint, top, right], dtype=object)


def plot_pairwise_relationships(
    data: Any, *, labels: Sequence[str] | None = None
) -> FigureResult:
    """Plot a pairwise scatter/histogram matrix for numeric data shaped [N,P]."""
    matrix = _numeric("data", data, ndim=2)
    if matrix.shape[1] < 2:
        raise ValueError("data must contain at least two columns.")
    names = _labels(labels, matrix.shape[1], "Variable")
    fig, axes = plt.subplots(matrix.shape[1], matrix.shape[1], squeeze=False, constrained_layout=True)
    for row in range(matrix.shape[1]):
        for column in range(matrix.shape[1]):
            ax = axes[row, column]
            if row == column:
                ax.hist(matrix[:, column], bins=10)
            else:
                ax.scatter(matrix[:, column], matrix[:, row], s=12)
            if row == matrix.shape[1] - 1:
                ax.set_xlabel(names[column])
            if column == 0:
                ax.set_ylabel(names[row])
    return fig, axes


def plot_relationship_scatter(
    x: Any,
    y: Any,
    *,
    groups: Sequence[Any] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
) -> FigureResult:
    """Plot an [N]-by-[N] relationship, optionally separated by explicit groups."""
    x_array = _numeric("x", x, ndim=1)
    y_array = _numeric("y", y, ndim=1)
    if x_array.size != y_array.size:
        raise ValueError("x and y must have equal lengths.")
    fig, ax = plt.subplots(constrained_layout=True)
    if groups is None:
        ax.scatter(x_array, y_array)
    else:
        group_array = _array("groups", groups, ndim=1)
        if group_array.size != x_array.size:
            raise ValueError("groups must have the same length as x and y.")
        for group in np.unique(group_array):
            selected = group_array == group
            ax.scatter(x_array[selected], y_array[selected], label=str(group))
        ax.legend(loc="best")
    _finish(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    return fig, ax


def plot_swarm_facet(
    values: Any,
    categories: Sequence[Any],
    *,
    facets: Sequence[Any] | None = None,
    ylabel: str | None = None,
) -> FigureResult:
    """Plot deterministic swarm-like category points, optionally faceted; inputs are [N]."""
    value_array = _numeric("values", values, ndim=1)
    category_array = _array("categories", categories, ndim=1)
    if value_array.size != category_array.size:
        raise ValueError("values and categories must have equal lengths.")
    facet_array = np.repeat("All", value_array.size) if facets is None else _array("facets", facets, ndim=1)
    if facet_array.size != value_array.size:
        raise ValueError("facets must have the same length as values.")
    facet_names = np.unique(facet_array)
    fig, axes = plt.subplots(1, len(facet_names), squeeze=False, constrained_layout=True)
    for ax, facet in zip(axes.ravel(), facet_names, strict=True):
        selected_facet = facet_array == facet
        category_names = np.unique(category_array[selected_facet])
        for position, category in enumerate(category_names):
            selected = selected_facet & (category_array == category)
            count = int(selected.sum())
            jitter = np.linspace(-0.15, 0.15, count) if count > 1 else np.zeros(1)
            ax.scatter(position + jitter, value_array[selected], s=18)
        ax.set_xticks(np.arange(len(category_names)), [str(value) for value in category_names])
        ax.set_title(str(facet))
        if ylabel:
            ax.set_ylabel(ylabel)
    return fig, axes if len(facet_names) > 1 else axes.ravel()[0]


def plot_regularization_errorbar(
    parameters: Any,
    mean_scores: Any,
    errors: Any,
    *,
    selected_parameter: float | None = None,
    xlabel: str = "Regularization parameter",
    ylabel: str = "Score",
) -> FigureResult:
    """Plot [N] regularization scores with [N] uncertainty values."""
    x = _numeric("parameters", parameters, ndim=1)
    means = _numeric("mean_scores", mean_scores, ndim=1)
    error = _numeric("errors", errors, ndim=1)
    if not (x.size == means.size == error.size):
        raise ValueError("parameters, mean_scores, and errors must have equal lengths.")
    if (error < 0).any():
        raise ValueError("errors must be non-negative.")
    fig, ax = plt.subplots(constrained_layout=True)
    ax.errorbar(x, means, yerr=error, marker="o")
    if selected_parameter is not None:
        ax.axvline(float(selected_parameter), linestyle="--")
    _finish(ax, title=None, xlabel=xlabel, ylabel=ylabel)
    return fig, ax


def plot_coefficient_path(
    parameters: Any,
    coefficients: Any,
    *,
    feature_names: Sequence[str] | None = None,
    selected_parameter: float | None = None,
) -> FigureResult:
    """Plot coefficient trajectories; parameters are [L], coefficients are [F,L]."""
    x = _numeric("parameters", parameters, ndim=1)
    matrix = _numeric("coefficients", coefficients, ndim=2)
    if matrix.shape[1] != x.size:
        raise ValueError("coefficients must have shape [features, len(parameters)].")
    names = _labels(feature_names, matrix.shape[0], "Feature")
    fig, ax = plt.subplots(constrained_layout=True)
    for name, row in zip(names, matrix, strict=True):
        ax.plot(x, row, label=name)
    if selected_parameter is not None:
        ax.axvline(float(selected_parameter), linestyle="--")
    ax.legend(loc="best")
    _finish(ax, title=None, xlabel="Regularization parameter", ylabel="Coefficient")
    return fig, ax


def plot_activation_curves(
    x: Any, curves: Mapping[str, Any], *, title: str | None = None
) -> FigureResult:
    """Plot named, precomputed activation curves sharing an [N] x-axis."""
    x_array = _numeric("x", x, ndim=1)
    if not isinstance(curves, Mapping) or not curves:
        raise ValueError("curves must be a non-empty mapping.")
    items = [(str(name), _numeric(f"curves[{name}]", values, ndim=1)) for name, values in curves.items()]
    if any(values.size != x_array.size for _, values in items):
        raise ValueError("Every activation curve must have len(x) values.")
    ncols = min(3, len(items))
    nrows = ceil(len(items) / ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, constrained_layout=True)
    flat = axes.ravel()
    for ax, (name, values) in zip(flat, items, strict=False):
        ax.plot(x_array, values)
        ax.set_title(name)
        ax.set_axisbelow(True)
        ax.grid(alpha=0.25)
    for ax in flat[len(items) :]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title)
    return fig, axes if len(items) > 1 else flat[0]


def plot_cluster_elbow(
    cluster_counts: Any, inertia: Any, *, title: str = "Cluster elbow"
) -> FigureResult:
    """Plot candidate cluster counts [K] against precomputed inertia [K]."""
    counts = _numeric("cluster_counts", cluster_counts, ndim=1)
    scores = _numeric("inertia", inertia, ndim=1)
    if counts.size != scores.size:
        raise ValueError("cluster_counts and inertia must have equal lengths.")
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(counts, scores, marker="o")
    _finish(ax, title=title, xlabel="Number of clusters", ylabel="Inertia")
    return fig, ax


def plot_cluster_silhouette(
    sample_scores: Any,
    cluster_labels: Sequence[Any],
    *,
    average_score: float | None = None,
) -> FigureResult:
    """Plot precomputed per-sample silhouette scores [N] by cluster [N]."""
    scores = _numeric("sample_scores", sample_scores, ndim=1)
    labels = _array("cluster_labels", cluster_labels, ndim=1)
    if scores.size != labels.size:
        raise ValueError("sample_scores and cluster_labels must have equal lengths.")
    fig, ax = plt.subplots(constrained_layout=True)
    lower = 0
    ticks: list[float] = []
    tick_labels: list[str] = []
    for label in np.unique(labels):
        values = np.sort(scores[labels == label])
        upper = lower + values.size
        ax.fill_betweenx(np.arange(lower, upper), 0, values, alpha=0.7)
        ticks.append((lower + upper - 1) / 2)
        tick_labels.append(str(label))
        lower = upper + 1
    if average_score is not None:
        ax.axvline(float(average_score), linestyle="--", label="Average")
        ax.legend(loc="best")
    ax.set_yticks(ticks, tick_labels)
    _finish(ax, title="Silhouette profile", xlabel="Silhouette score", ylabel="Cluster")
    return fig, ax


def plot_cluster_scatter(
    x: Any,
    y: Any,
    cluster_labels: Sequence[Any],
    *,
    centers: Any | None = None,
) -> FigureResult:
    """Plot two-dimensional observations [N] colored by cluster labels [N]."""
    x_array = _numeric("x", x, ndim=1)
    y_array = _numeric("y", y, ndim=1)
    labels = _array("cluster_labels", cluster_labels, ndim=1)
    if not (x_array.size == y_array.size == labels.size):
        raise ValueError("x, y, and cluster_labels must have equal lengths.")
    fig, ax = plt.subplots(constrained_layout=True)
    for label in np.unique(labels):
        selected = labels == label
        ax.scatter(x_array[selected], y_array[selected], label=str(label))
    if centers is not None:
        center_array = _numeric("centers", centers, ndim=2)
        if center_array.shape[1] != 2:
            raise ValueError("centers must have shape [K,2].")
        ax.scatter(center_array[:, 0], center_array[:, 1], marker="x", s=80, label="Centers")
    ax.legend(loc="best")
    _finish(ax, title="Cluster assignment", xlabel="Dimension 1", ylabel="Dimension 2")
    return fig, ax


def plot_embedding(
    embedding: Any,
    *,
    labels: Sequence[Any] | None = None,
    method_name: str = "Embedding",
) -> FigureResult:
    """Plot a precomputed two-dimensional embedding shaped [N,2]."""
    points = _numeric("embedding", embedding, ndim=2)
    if points.shape[1] != 2:
        raise ValueError("embedding must have shape [N,2].")
    fig, ax = plt.subplots(constrained_layout=True)
    if labels is None:
        ax.scatter(points[:, 0], points[:, 1])
    else:
        label_array = _array("labels", labels, ndim=1)
        if label_array.size != points.shape[0]:
            raise ValueError("labels must contain one value per embedding row.")
        for label in np.unique(label_array):
            selected = label_array == label
            ax.scatter(points[selected, 0], points[selected, 1], label=str(label))
        ax.legend(loc="best")
    _finish(ax, title=method_name, xlabel="Dimension 1", ylabel="Dimension 2")
    return fig, ax


def plot_forecast(
    time: Any,
    observed: Any,
    forecast: Any,
    *,
    lower: Any | None = None,
    upper: Any | None = None,
) -> FigureResult:
    """Plot observed and forecast series [T], with optional [T] interval bounds."""
    time_array = _numeric("time", time, ndim=1)
    observed_array = _numeric("observed", observed, ndim=1)
    forecast_array = _numeric("forecast", forecast, ndim=1)
    if not (time_array.size == observed_array.size == forecast_array.size):
        raise ValueError("time, observed, and forecast must have equal lengths.")
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(time_array, observed_array, label="Observed")
    ax.plot(time_array, forecast_array, label="Forecast")
    if (lower is None) != (upper is None):
        raise ValueError("lower and upper must be supplied together.")
    if lower is not None and upper is not None:
        low = _numeric("lower", lower, ndim=1)
        high = _numeric("upper", upper, ndim=1)
        if low.size != time_array.size or high.size != time_array.size:
            raise ValueError("Forecast bounds must have len(time) values.")
        if np.any(low > high):
            raise ValueError("lower must not exceed upper.")
        ax.fill_between(time_array, low, high, alpha=0.2, label="Interval")
    ax.legend(loc="best")
    _finish(ax, title="Forecast", xlabel="Time", ylabel="Value")
    return fig, ax


def plot_forecast_diagnostics(
    residuals: Any, *, fitted: Any | None = None, max_lag: int = 20
) -> FigureResult:
    """Plot residual sequence, distribution, autocorrelation, and residual-vs-fitted diagnostics."""
    residual = _numeric("residuals", residuals, ndim=1)
    if residual.size < 3:
        raise ValueError("residuals must contain at least three values.")
    if not isinstance(max_lag, int) or max_lag < 1:
        raise ValueError("max_lag must be a positive integer.")
    fitted_array = np.arange(residual.size) if fitted is None else _numeric("fitted", fitted, ndim=1)
    if fitted_array.size != residual.size:
        raise ValueError("fitted must have the same length as residuals.")
    lag_count = min(max_lag, residual.size - 1)
    centered = residual - residual.mean()
    denominator = float(np.dot(centered, centered))
    autocorrelation = np.asarray(
        [1.0 if lag == 0 else np.dot(centered[:-lag], centered[lag:]) / denominator for lag in range(lag_count + 1)]
    )
    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    axes[0, 0].plot(np.arange(residual.size), residual)
    axes[0, 0].axhline(0, linewidth=1)
    axes[0, 0].set_title("Residual sequence")
    axes[0, 1].hist(residual, bins=min(12, residual.size))
    axes[0, 1].set_title("Residual distribution")
    axes[1, 0].stem(np.arange(lag_count + 1), autocorrelation)
    axes[1, 0].set_title("Residual autocorrelation")
    axes[1, 1].scatter(fitted_array, residual)
    axes[1, 1].axhline(0, linewidth=1)
    axes[1, 1].set_title("Residual vs fitted")
    return fig, axes


def plot_signal_spectrum(
    time: Any, signal: Any, frequencies: Any, magnitude: Any
) -> FigureResult:
    """Plot a [N] signal and a precomputed [F] frequency magnitude spectrum."""
    time_array = _numeric("time", time, ndim=1)
    signal_array = _numeric("signal", signal, ndim=1)
    frequency_array = _numeric("frequencies", frequencies, ndim=1)
    magnitude_array = _numeric("magnitude", magnitude, ndim=1)
    if time_array.size != signal_array.size or frequency_array.size != magnitude_array.size:
        raise ValueError("Signal and spectrum coordinates must align with their values.")
    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    axes[0].plot(time_array, signal_array)
    axes[0].set_title("Signal")
    axes[0].set_xlabel("Time")
    axes[1].plot(frequency_array, magnitude_array)
    axes[1].set_title("Frequency spectrum")
    axes[1].set_xlabel("Frequency")
    return fig, axes


def plot_image(
    image: Any, *, title: str | None = None, cmap: str | None = "gray"
) -> FigureResult:
    """Display a two-dimensional image or an [H,W,3/4] color image."""
    array = _array("image", image, ndim=(2, 3))
    if array.ndim == 3 and array.shape[-1] not in (3, 4):
        raise ValueError("A three-dimensional image must have 3 or 4 channels last.")
    fig, ax = plt.subplots(constrained_layout=True)
    ax.imshow(array, cmap=cmap if array.ndim == 2 else None)
    ax.axis("off")
    if title:
        ax.set_title(title)
    return fig, ax


def plot_image_montage(
    images: Sequence[Any],
    *,
    labels: Sequence[str] | None = None,
    ncols: int = 3,
    cmap: str | None = "gray",
) -> FigureResult:
    """Display a sequence of same-purpose 2-D or RGB images as a montage."""
    arrays = [_array(f"images[{index}]", value, ndim=(2, 3)) for index, value in enumerate(images)]
    if not arrays:
        raise ValueError("images must not be empty.")
    if not isinstance(ncols, int) or ncols < 1:
        raise ValueError("ncols must be a positive integer.")
    names = _labels(labels, len(arrays), "Image") if labels is not None else [""] * len(arrays)
    ncols = min(ncols, len(arrays))
    nrows = ceil(len(arrays) / ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, constrained_layout=True)
    for ax, array, name in zip(axes.ravel(), arrays, names, strict=False):
        if array.ndim == 3 and array.shape[-1] not in (3, 4):
            raise ValueError("Color montage images must have 3 or 4 channels last.")
        ax.imshow(array, cmap=cmap if array.ndim == 2 else None)
        ax.axis("off")
        if name:
            ax.set_title(name)
    for ax in axes.ravel()[len(arrays) :]:
        ax.set_visible(False)
    return fig, axes


def plot_image_projection(
    volume: Any, *, axis: int = 0, threshold: float | None = None
) -> FigureResult:
    """Plot a sum projection of a numeric volume shaped [D,H,W]."""
    array = _numeric("volume", volume, ndim=3)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    projected = np.sum(array > threshold, axis=axis) if threshold is not None else np.sum(array, axis=axis)
    fig, ax = plt.subplots(constrained_layout=True)
    ax.imshow(projected, cmap="gray")
    ax.set_title("Volume projection")
    ax.axis("off")
    return fig, ax


def plot_3d_volume_scatter(
    volume: Any, *, threshold: float = 0.0, max_points: int = 5000
) -> FigureResult:
    """Plot non-background voxels from a [D,H,W] volume as a 3-D scatter."""
    array = _numeric("volume", volume, ndim=3)
    if not isinstance(max_points, int) or max_points < 1:
        raise ValueError("max_points must be a positive integer.")
    coordinates = np.argwhere(array > threshold)
    if coordinates.size == 0:
        raise ValueError("No voxels exceed threshold.")
    if coordinates.shape[0] > max_points:
        step = ceil(coordinates.shape[0] / max_points)
        coordinates = coordinates[::step]
    values = array[tuple(coordinates.T)]
    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    points = ax.scatter(coordinates[:, 2], coordinates[:, 1], coordinates[:, 0], c=values, s=4)
    fig.colorbar(points, ax=ax, label="Voxel value")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    return fig, ax


def plot_venn_diagram(
    subset_sizes: Sequence[float], *, labels: Sequence[str] | None = None
) -> FigureResult:
    """Plot a two-set (3 regions) or three-set (7 regions) Venn diagram."""
    sizes = _numeric("subset_sizes", subset_sizes, ndim=1)
    if sizes.size not in (3, 7) or (sizes < 0).any():
        raise ValueError("subset_sizes must contain 3 or 7 non-negative region sizes.")
    count = 2 if sizes.size == 3 else 3
    names = _labels(labels, count, "Set")
    fig, ax = plt.subplots(constrained_layout=True)
    centers = [(0.42, 0.55), (0.58, 0.55)] if count == 2 else [(0.42, 0.60), (0.58, 0.60), (0.50, 0.43)]
    colors = ("tab:blue", "tab:orange", "tab:green")
    for center, color, name in zip(centers, colors, names, strict=True):
        ax.add_patch(Circle(center, 0.25, color=color, alpha=0.3, label=name))
    if count == 2:
        positions = [(0.30, 0.55), (0.70, 0.55), (0.50, 0.55)]
    else:
        positions = [(0.30, 0.64), (0.70, 0.64), (0.50, 0.72), (0.50, 0.30), (0.39, 0.47), (0.61, 0.47), (0.50, 0.52)]
    for position, value in zip(positions, sizes, strict=True):
        ax.text(*position, f"{value:g}", ha="center", va="center")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


__all__ = [
    "plot_3d_volume_scatter",
    "plot_activation_curves",
    "plot_bar",
    "plot_boxplot",
    "plot_cluster_elbow",
    "plot_cluster_scatter",
    "plot_cluster_silhouette",
    "plot_coefficient_path",
    "plot_correlation_heatmap",
    "plot_density",
    "plot_embedding",
    "plot_forecast",
    "plot_forecast_diagnostics",
    "plot_histogram",
    "plot_image",
    "plot_image_montage",
    "plot_image_projection",
    "plot_joint_distribution",
    "plot_line",
    "plot_pairwise_relationships",
    "plot_regularization_errorbar",
    "plot_relationship_scatter",
    "plot_scatter",
    "plot_signal_spectrum",
    "plot_swarm_facet",
    "plot_venn_diagram",
    "plot_violin",
]
