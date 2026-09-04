"""Survival figures for precomputed curves, metrics, and optimization histories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


FigureResult = tuple[Figure, Axes | np.ndarray]


def _numeric(name: str, values: Any, *, ndim: int | tuple[int, ...]) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    allowed = (ndim,) if isinstance(ndim, int) else ndim
    if array.ndim not in allowed or array.size == 0:
        raise ValueError(f"{name} must be non-empty with {allowed} dimensions.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _aligned(name: str, time: np.ndarray, values: Any) -> np.ndarray:
    array = _numeric(name, values, ndim=(1, 2))
    if array.shape[0] != time.size:
        raise ValueError(f"{name} must have len(time) rows.")
    return array


def _time(name: str, values: Any) -> np.ndarray:
    time = _numeric(name, values, ndim=1)
    if np.any(np.diff(time) < 0):
        raise ValueError(f"{name} must be sorted in non-decreasing order.")
    return time


def plot_kaplan_meier(
    curves: Mapping[str, tuple[Any, Any]], *, title: str = "Kaplan-Meier curves"
) -> FigureResult:
    """Plot validated KM step curves supplied as label -> ([T] time, [T] survival)."""
    if not isinstance(curves, Mapping) or not curves:
        raise ValueError("curves must be a non-empty mapping.")
    fig, ax = plt.subplots(constrained_layout=True)
    for label, pair in curves.items():
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("Each curve must be a (time, survival) tuple.")
        time = _time(f"curves[{label}].time", pair[0])
        survival = _numeric(f"curves[{label}].survival", pair[1], ndim=1)
        if survival.size != time.size or np.any((survival < 0) | (survival > 1)):
            raise ValueError("KM survival values must align with time and lie in [0,1].")
        ax.step(time, survival, where="post", label=str(label))
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc="best")
    return fig, ax


def plot_survival_curves(
    time: Any,
    survival: Any,
    *,
    labels: Sequence[str] | None = None,
    title: str = "Predicted survival curves",
) -> FigureResult:
    """Plot precomputed survival probabilities shaped [T] or [T,N]."""
    time_array = _time("time", time)
    matrix = _aligned("survival", time_array, survival)
    matrix = matrix[:, np.newaxis] if matrix.ndim == 1 else matrix
    if np.any((matrix < 0) | (matrix > 1)):
        raise ValueError("survival probabilities must lie in [0,1].")
    names = [f"Curve {index + 1}" for index in range(matrix.shape[1])] if labels is None else [str(value) for value in labels]
    if len(names) != matrix.shape[1]:
        raise ValueError("labels must contain one name per survival column.")
    fig, ax = plt.subplots(constrained_layout=True)
    for name, values in zip(names, matrix.T, strict=True):
        ax.plot(time_array, values, label=name if matrix.shape[1] > 1 or labels else None)
    if matrix.shape[1] > 1 or labels:
        ax.legend(loc="best")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    return fig, ax


def plot_survival_training_loss(
    epochs: Any, train_loss: Any, *, validation_loss: Any | None = None
) -> FigureResult:
    """Plot training and optional validation loss histories aligned on epochs [E]."""
    epoch_array = _time("epochs", epochs)
    train = _numeric("train_loss", train_loss, ndim=1)
    if train.size != epoch_array.size:
        raise ValueError("train_loss must have len(epochs) values.")
    validation = None if validation_loss is None else _numeric("validation_loss", validation_loss, ndim=1)
    if validation is not None and validation.size != epoch_array.size:
        raise ValueError("validation_loss must have len(epochs) values.")
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(epoch_array, train, label="Training loss")
    if validation is not None:
        ax.plot(epoch_array, validation, label="Validation loss")
        ax.legend(loc="best")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Survival training loss")
    return fig, ax


def plot_learning_rate_finder(
    learning_rates: Any, losses: Any, *, selected_rate: float | None = None
) -> FigureResult:
    """Plot precomputed learning-rate finder values [N] against losses [N]."""
    rates = _numeric("learning_rates", learning_rates, ndim=1)
    loss = _numeric("losses", losses, ndim=1)
    if rates.size != loss.size or np.any(rates <= 0):
        raise ValueError("Positive learning_rates and losses must have equal lengths.")
    fig, ax = plt.subplots(constrained_layout=True)
    ax.semilogx(rates, loss)
    if selected_rate is not None:
        if selected_rate <= 0:
            raise ValueError("selected_rate must be positive.")
        ax.axvline(float(selected_rate), linestyle="--", label="Selected rate")
        ax.legend(loc="best")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Loss")
    ax.set_title("Learning-rate finder")
    return fig, ax


def plot_learning_rate_schedule(steps: Any, learning_rates: Any) -> FigureResult:
    """Plot a positive, precomputed learning-rate schedule [N] over steps [N]."""
    step_array = _time("steps", steps)
    rates = _numeric("learning_rates", learning_rates, ndim=1)
    if rates.size != step_array.size or np.any(rates <= 0):
        raise ValueError("Positive learning_rates must align with steps.")
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(step_array, rates)
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning-rate schedule")
    ax.set_axisbelow(True)
    ax.grid(alpha=0.25)
    return fig, ax


def plot_brier_score(
    time: Any, scores: Any, *, labels: Sequence[str] | None = None
) -> FigureResult:
    """Plot one or more precomputed Brier-score series shaped [T] or [T,M]."""
    time_array = _time("time", time)
    matrix = _aligned("scores", time_array, scores)
    matrix = matrix[:, np.newaxis] if matrix.ndim == 1 else matrix
    if np.any((matrix < 0) | (matrix > 1)):
        raise ValueError("Brier scores must lie in [0,1].")
    names = [f"Model {index + 1}" for index in range(matrix.shape[1])] if labels is None else [str(value) for value in labels]
    if len(names) != matrix.shape[1]:
        raise ValueError("labels must contain one name per score column.")
    fig, ax = plt.subplots(constrained_layout=True)
    for name, values in zip(names, matrix.T, strict=True):
        ax.plot(time_array, values, label=name if matrix.shape[1] > 1 or labels else None)
    if matrix.shape[1] > 1 or labels:
        ax.legend(loc="best")
    ax.set_xlabel("Time")
    ax.set_ylabel("Brier score")
    ax.set_title("Time-dependent Brier score")
    return fig, ax


def plot_nbll(
    time: Any, scores: Any, *, labels: Sequence[str] | None = None
) -> FigureResult:
    """Plot precomputed negative binomial log-likelihood series [T] or [T,M]."""
    time_array = _time("time", time)
    matrix = _aligned("scores", time_array, scores)
    matrix = matrix[:, np.newaxis] if matrix.ndim == 1 else matrix
    names = [f"Model {index + 1}" for index in range(matrix.shape[1])] if labels is None else [str(value) for value in labels]
    if len(names) != matrix.shape[1]:
        raise ValueError("labels must contain one name per score column.")
    fig, ax = plt.subplots(constrained_layout=True)
    for name, values in zip(names, matrix.T, strict=True):
        ax.plot(time_array, values, label=name if matrix.shape[1] > 1 or labels else None)
    if matrix.shape[1] > 1 or labels:
        ax.legend(loc="best")
    ax.set_xlabel("Time")
    ax.set_ylabel("Negative binomial log-likelihood")
    ax.set_title("Time-dependent NBLL")
    return fig, ax


__all__ = [
    "plot_brier_score",
    "plot_kaplan_meier",
    "plot_learning_rate_finder",
    "plot_learning_rate_schedule",
    "plot_nbll",
    "plot_survival_curves",
    "plot_survival_training_loss",
]
