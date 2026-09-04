"""Classification figures for labels, probabilities, and training histories."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


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


def _class_names(classes: np.ndarray, names: Sequence[str] | None) -> list[str]:
    if names is None:
        return [str(value) for value in classes]
    result = [str(value) for value in names]
    if len(result) != classes.size:
        raise ValueError("class_names must contain one name per class.")
    return result


def plot_class_distribution_bar(
    classes: Sequence[Any], counts: Any, *, title: str = "Class distribution"
) -> FigureResult:
    """Plot class counts; classes and counts are parallel [K] vectors."""
    names = [str(value) for value in classes]
    if not names:
        raise ValueError("classes must not be empty.")
    values = _numeric("counts", counts, ndim=1)
    if values.size != len(names) or (values < 0).any():
        raise ValueError("counts must be non-negative with one value per class.")
    fig, ax = plt.subplots(constrained_layout=True)
    ax.bar(np.arange(len(names)), values)
    ax.set_xticks(np.arange(len(names)), names)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    return fig, ax


def plot_class_distribution_pie(
    classes: Sequence[Any], counts: Any, *, title: str = "Class distribution"
) -> FigureResult:
    """Plot class proportions; classes and non-negative counts are [K]."""
    names = [str(value) for value in classes]
    if not names:
        raise ValueError("classes must not be empty.")
    values = _numeric("counts", counts, ndim=1)
    if values.size != len(names) or (values < 0).any() or values.sum() <= 0:
        raise ValueError("counts must be non-negative, non-zero, and aligned with classes.")
    fig, ax = plt.subplots(constrained_layout=True)
    ax.pie(values, labels=names, autopct="%1.1f%%", startangle=90)
    ax.set_title(title)
    return fig, ax


def plot_class_sample_montage(
    images: Sequence[Any],
    labels: Sequence[Any],
    *,
    ncols: int = 3,
    cmap: str | None = "gray",
) -> FigureResult:
    """Display labeled classification samples; images and labels contain N items."""
    arrays = [_array(f"images[{index}]", image, ndim=(2, 3)) for index, image in enumerate(images)]
    if not arrays:
        raise ValueError("images must not be empty.")
    names = [str(value) for value in labels]
    if len(names) != len(arrays):
        raise ValueError("labels must contain one value per image.")
    if not isinstance(ncols, int) or ncols < 1:
        raise ValueError("ncols must be a positive integer.")
    ncols = min(ncols, len(arrays))
    nrows = ceil(len(arrays) / ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, constrained_layout=True)
    for ax, image, label in zip(axes.ravel(), arrays, names, strict=False):
        if image.ndim == 3 and image.shape[-1] not in (3, 4):
            raise ValueError("Color images must have 3 or 4 channels last.")
        ax.imshow(image, cmap=cmap if image.ndim == 2 else None)
        ax.set_title(label)
        ax.axis("off")
    for ax in axes.ravel()[len(arrays) :]:
        ax.set_visible(False)
    return fig, axes


def _confusion_from_labels(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> np.ndarray:
    matrix = np.zeros((classes.size, classes.size), dtype=int)
    class_to_index = {value: index for index, value in enumerate(classes.tolist())}
    try:
        for observed, predicted in zip(y_true.tolist(), y_pred.tolist(), strict=True):
            matrix[class_to_index[observed], class_to_index[predicted]] += 1
    except KeyError as exc:
        raise ValueError("y_true and y_pred contain a class not listed in classes.") from exc
    return matrix


def plot_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    *,
    classes: Sequence[Any] | None = None,
    class_names: Sequence[str] | None = None,
    normalize: bool = False,
) -> FigureResult:
    """Plot a confusion matrix from parallel observed and predicted labels [N]."""
    observed = _array("y_true", y_true, ndim=1)
    predicted = _array("y_pred", y_pred, ndim=1)
    if observed.size != predicted.size:
        raise ValueError("y_true and y_pred must have equal lengths.")
    resolved_classes = np.asarray(list(classes)) if classes is not None else np.unique(np.concatenate([observed, predicted]))
    if resolved_classes.size == 0:
        raise ValueError("classes must not be empty.")
    names = _class_names(resolved_classes, class_names)
    counts = _confusion_from_labels(observed, predicted, resolved_classes)
    display = counts.astype(float)
    if normalize:
        denominators = display.sum(axis=1, keepdims=True)
        display = np.divide(display, denominators, out=np.zeros_like(display), where=denominators != 0)
    fig, ax = plt.subplots(constrained_layout=True)
    image = ax.imshow(display, cmap="Blues")
    for row in range(display.shape[0]):
        for column in range(display.shape[1]):
            text = f"{display[row, column]:.2f}" if normalize else str(counts[row, column])
            ax.text(column, row, text, ha="center", va="center")
    ax.set_xticks(np.arange(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(names)), names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Observed class")
    ax.set_title("Confusion matrix")
    fig.colorbar(image, ax=ax)
    return fig, ax


def _binary_roc(observed: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    positives = int(observed.sum())
    negatives = observed.size - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC requires both positive and negative observations.")
    order = np.argsort(-score, kind="mergesort")
    truth = observed[order]
    ranked = score[order]
    distinct = np.r_[ranked[1:] != ranked[:-1], True]
    true_positive = np.cumsum(truth)[distinct]
    false_positive = np.cumsum(1 - truth)[distinct]
    tpr = np.r_[0.0, true_positive / positives]
    fpr = np.r_[0.0, false_positive / negatives]
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def _curve_classes(
    y_true: np.ndarray, scores: np.ndarray, classes: Sequence[Any] | None
) -> tuple[np.ndarray, np.ndarray]:
    if scores.ndim == 1:
        unique = np.unique(y_true)
        if unique.size != 2:
            raise ValueError("One-dimensional scores require exactly two observed classes.")
        return unique[-1:], scores[:, np.newaxis]
    resolved = np.asarray(list(classes)) if classes is not None else np.unique(y_true)
    if scores.shape[0] != y_true.size or scores.shape[1] != resolved.size:
        raise ValueError("scores must have shape [N, number of classes].")
    if resolved.size == 2:
        return resolved[-1:], scores[:, -1:]
    return resolved, scores


def plot_roc_curve(
    y_true: Any,
    scores: Any,
    *,
    classes: Sequence[Any] | None = None,
    class_names: Sequence[str] | None = None,
) -> FigureResult:
    """Plot binary or one-vs-rest ROC curves from labels [N] and scores [N] or [N,C]."""
    observed = _array("y_true", y_true, ndim=1)
    score_array = _numeric("scores", scores, ndim=(1, 2))
    if score_array.shape[0] != observed.size:
        raise ValueError("scores must contain one row per observed label.")
    curve_classes, curve_scores = _curve_classes(observed, score_array, classes)
    names = _class_names(curve_classes, class_names)
    fig, ax = plt.subplots(constrained_layout=True)
    for class_value, name, class_score in zip(curve_classes, names, curve_scores.T, strict=True):
        binary = (observed == class_value).astype(int)
        fpr, tpr, auc = _binary_roc(binary, class_score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Receiver operating characteristic")
    ax.legend(loc="lower right")
    return fig, ax


def _binary_precision_recall(observed: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positives = int(observed.sum())
    if positives == 0 or positives == observed.size:
        raise ValueError("Precision-recall requires both positive and negative observations.")
    order = np.argsort(-score, kind="mergesort")
    truth = observed[order]
    ranked = score[order]
    distinct = np.r_[ranked[1:] != ranked[:-1], True]
    true_positive = np.cumsum(truth)[distinct]
    false_positive = np.cumsum(1 - truth)[distinct]
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / positives
    return np.r_[0.0, recall], np.r_[1.0, precision]


def plot_precision_recall_curve(
    y_true: Any,
    scores: Any,
    *,
    classes: Sequence[Any] | None = None,
    class_names: Sequence[str] | None = None,
) -> FigureResult:
    """Plot binary or one-vs-rest precision-recall curves from [N] labels and scores."""
    observed = _array("y_true", y_true, ndim=1)
    score_array = _numeric("scores", scores, ndim=(1, 2))
    if score_array.shape[0] != observed.size:
        raise ValueError("scores must contain one row per observed label.")
    curve_classes, curve_scores = _curve_classes(observed, score_array, classes)
    names = _class_names(curve_classes, class_names)
    fig, ax = plt.subplots(constrained_layout=True)
    for class_value, name, class_score in zip(curve_classes, names, curve_scores.T, strict=True):
        recall, precision = _binary_precision_recall((observed == class_value).astype(int), class_score)
        ax.plot(recall, precision, label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")
    ax.legend(loc="lower left")
    return fig, ax


def plot_training_history(
    train_loss: Any,
    *,
    validation_loss: Any | None = None,
    validation_metric: Any | None = None,
    metric_name: str = "Validation metric",
    validation_epochs: Any | None = None,
) -> FigureResult:
    """Plot training loss [E] and optional validation loss/metric histories."""
    train = _numeric("train_loss", train_loss, ndim=1)
    epochs = np.arange(1, train.size + 1)
    val_loss = None if validation_loss is None else _numeric("validation_loss", validation_loss, ndim=1)
    if val_loss is not None and val_loss.size != train.size:
        raise ValueError("validation_loss must have len(train_loss) values.")
    metric = None if validation_metric is None else _numeric("validation_metric", validation_metric, ndim=1)
    panel_count = 2 if metric is not None else 1
    fig, axes = plt.subplots(1, panel_count, squeeze=False, constrained_layout=True)
    loss_ax = axes.ravel()[0]
    loss_ax.plot(epochs, train, label="Training loss")
    if val_loss is not None:
        loss_ax.plot(epochs, val_loss, label="Validation loss")
    if val_loss is not None:
        loss_ax.legend(loc="best")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.set_title("Training history")
    if metric is not None:
        metric_epochs = np.arange(1, metric.size + 1) if validation_epochs is None else _numeric("validation_epochs", validation_epochs, ndim=1)
        if metric_epochs.size != metric.size:
            raise ValueError("validation_epochs and validation_metric must have equal lengths.")
        metric_ax = axes.ravel()[1]
        metric_ax.plot(metric_epochs, metric)
        metric_ax.set_xlabel("Epoch")
        metric_ax.set_ylabel(metric_name)
        metric_ax.set_title(metric_name)
    return fig, axes if panel_count > 1 else loss_ax


__all__ = [
    "plot_class_distribution_bar",
    "plot_class_distribution_pie",
    "plot_class_sample_montage",
    "plot_confusion_matrix",
    "plot_precision_recall_curve",
    "plot_roc_curve",
    "plot_training_history",
]
