"""Plots for explicit, precomputed model-interpretation results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


FigureResult = tuple[Figure, Axes | np.ndarray]


def _numeric(name: str, values: Any, *, ndim: int) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if array.ndim != ndim or array.size == 0:
        raise ValueError(f"{name} must be non-empty with {ndim} dimensions.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _names(feature_names: Sequence[str], count: int) -> list[str]:
    names = [str(value) for value in feature_names]
    if len(names) != count:
        raise ValueError("feature_names must contain one name per value.")
    return names


def plot_feature_coefficients(
    feature_names: Sequence[str], coefficients: Any, *, title: str = "Feature coefficients"
) -> FigureResult:
    """Plot signed model coefficients [F] against explicit feature names [F]."""
    values = _numeric("coefficients", coefficients, ndim=1)
    names = _names(feature_names, values.size)
    order = np.argsort(values)
    fig, ax = plt.subplots(constrained_layout=True)
    ax.barh(np.arange(values.size), values[order])
    ax.set_yticks(np.arange(values.size), np.asarray(names)[order])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Coefficient")
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.25)
    return fig, ax


def plot_feature_importance(
    feature_names: Sequence[str], importance: Any, *, title: str = "Feature importance"
) -> FigureResult:
    """Plot non-negative, precomputed feature-importance values [F]."""
    values = _numeric("importance", importance, ndim=1)
    if np.any(values < 0):
        raise ValueError("importance values must be non-negative.")
    names = _names(feature_names, values.size)
    order = np.argsort(values)
    fig, ax = plt.subplots(constrained_layout=True)
    ax.barh(np.arange(values.size), values[order])
    ax.set_yticks(np.arange(values.size), np.asarray(names)[order])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.25)
    return fig, ax


def plot_occlusion_sensitivity(
    image: Any,
    sensitivity: Any,
    *,
    image_cmap: str = "gray",
    sensitivity_cmap: str = "jet",
) -> FigureResult:
    """Plot aligned, precomputed [H,W] image and occlusion-sensitivity arrays."""
    image_array = _numeric("image", image, ndim=2)
    sensitivity_array = _numeric("sensitivity", sensitivity, ndim=2)
    if image_array.shape != sensitivity_array.shape:
        raise ValueError("image and sensitivity must have equal [H,W] shapes.")
    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    image_artist = axes[0].imshow(image_array, cmap=image_cmap)
    sensitivity_artist = axes[1].imshow(sensitivity_array, cmap=sensitivity_cmap)
    axes[0].set_title("Image")
    axes[1].set_title("Occlusion sensitivity")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(image_artist, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(sensitivity_artist, ax=axes[1], fraction=0.046, pad=0.04)
    return fig, axes


__all__ = [
    "plot_feature_coefficients",
    "plot_feature_importance",
    "plot_occlusion_sensitivity",
]
