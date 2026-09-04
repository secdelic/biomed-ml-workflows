"""Segmentation figures for explicit images, masks, predictions, and histories."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


FigureResult = tuple[Figure, Axes | np.ndarray]


def _array(name: str, values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _volume_or_image(name: str, values: Any, *, sample: int = 0, channel: int = 0) -> np.ndarray:
    array = _array(name, values)
    if array.ndim == 5:
        if not 0 <= sample < array.shape[0] or not 0 <= channel < array.shape[1]:
            raise ValueError(f"{name} sample or channel is out of range.")
        array = array[sample, channel]
    elif array.ndim == 4:
        if not 0 <= channel < array.shape[0]:
            raise ValueError(f"{name} channel is out of range.")
        array = array[channel]
    if array.ndim not in (2, 3):
        raise ValueError(f"{name} must resolve to [H,W] or [D,H,W].")
    return array


def _slice(array: np.ndarray, *, axis: int, index: int | None) -> np.ndarray:
    if array.ndim == 2:
        return array
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    resolved = array.shape[axis] // 2 if index is None else int(index)
    if not 0 <= resolved < array.shape[axis]:
        raise ValueError("slice_index is out of range.")
    return np.take(array, resolved, axis=axis)


def _channels(name: str, values: Any, *, sample: int = 0) -> np.ndarray:
    array = _array(name, values)
    if array.ndim == 5:
        if not 0 <= sample < array.shape[0]:
            raise ValueError(f"{name} sample is out of range.")
        array = array[sample]
    if array.ndim == 3:
        array = array[np.newaxis, ...]
    if array.ndim != 4:
        raise ValueError(f"{name} must have shape [C,D,H,W] or [B,C,D,H,W].")
    return array


def plot_image_and_mask(
    image: Any,
    mask: Any,
    *,
    image_channel: int = 0,
    mask_channel: int = 0,
    slice_axis: int = 0,
    slice_index: int | None = None,
) -> FigureResult:
    """Plot aligned image and mask slices from 2-D, 3-D, [C,D,H,W], or [B,C,D,H,W] inputs."""
    image_slice = _slice(_volume_or_image("image", image, channel=image_channel), axis=slice_axis, index=slice_index)
    mask_slice = _slice(_volume_or_image("mask", mask, channel=mask_channel), axis=slice_axis, index=slice_index)
    if image_slice.shape != mask_slice.shape:
        raise ValueError("Resolved image and mask slices must have equal shapes.")
    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    axes[0].imshow(image_slice, cmap="gray")
    axes[0].set_title("Image")
    axes[1].imshow(mask_slice)
    axes[1].set_title("Segmentation")
    for ax in axes:
        ax.axis("off")
    return fig, axes


def _plot_channels(
    values: Any,
    *,
    panel_name: str,
    sample: int,
    slice_axis: int,
    slice_index: int | None,
    cmap: str | None,
) -> FigureResult:
    channel_array = _channels(panel_name.lower(), values, sample=sample)
    ncols = min(4, channel_array.shape[0])
    nrows = ceil(channel_array.shape[0] / ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, constrained_layout=True)
    for index, (ax, channel) in enumerate(zip(axes.ravel(), channel_array, strict=False)):
        ax.imshow(_slice(channel, axis=slice_axis, index=slice_index), cmap=cmap)
        ax.set_title(f"{panel_name} channel {index}")
        ax.axis("off")
    for ax in axes.ravel()[channel_array.shape[0] :]:
        ax.set_visible(False)
    return fig, axes


def plot_input_channels(
    image: Any, *, sample: int = 0, slice_axis: int = 0, slice_index: int | None = None
) -> FigureResult:
    """Plot every channel of an image tensor shaped [C,D,H,W] or [B,C,D,H,W]."""
    return _plot_channels(
        image,
        panel_name="Input",
        sample=sample,
        slice_axis=slice_axis,
        slice_index=slice_index,
        cmap="gray",
    )


def plot_ground_truth_channels(
    mask: Any, *, sample: int = 0, slice_axis: int = 0, slice_index: int | None = None
) -> FigureResult:
    """Plot every channel of a ground-truth tensor [C,D,H,W] or [B,C,D,H,W]."""
    return _plot_channels(
        mask,
        panel_name="Ground truth",
        sample=sample,
        slice_axis=slice_axis,
        slice_index=slice_index,
        cmap=None,
    )


def plot_prediction_channels(
    prediction: Any,
    *,
    sample: int = 0,
    slice_axis: int = 0,
    slice_index: int | None = None,
) -> FigureResult:
    """Plot every channel of a prediction tensor [C,D,H,W] or [B,C,D,H,W]."""
    return _plot_channels(
        prediction,
        panel_name="Prediction",
        sample=sample,
        slice_axis=slice_axis,
        slice_index=slice_index,
        cmap=None,
    )


def plot_image_mask_prediction(
    image: Any,
    mask: Any,
    prediction: Any,
    *,
    image_channel: int = 0,
    mask_channel: int = 0,
    prediction_channel: int = 0,
    slice_axis: int = 0,
    slice_index: int | None = None,
) -> FigureResult:
    """Plot aligned image, ground-truth, and prediction slices from explicit tensors."""
    panels = [
        _slice(_volume_or_image("image", image, channel=image_channel), axis=slice_axis, index=slice_index),
        _slice(_volume_or_image("mask", mask, channel=mask_channel), axis=slice_axis, index=slice_index),
        _slice(
            _volume_or_image("prediction", prediction, channel=prediction_channel),
            axis=slice_axis,
            index=slice_index,
        ),
    ]
    if len({panel.shape for panel in panels}) != 1:
        raise ValueError("Resolved image, mask, and prediction slices must have equal shapes.")
    fig, axes = plt.subplots(1, 3, constrained_layout=True)
    titles = ("Image", "Ground truth", "Prediction")
    for ax, panel, title in zip(axes, panels, titles, strict=True):
        ax.imshow(panel, cmap="gray" if title == "Image" else None)
        ax.set_title(title)
        ax.axis("off")
    return fig, axes


def plot_masking_comparison(
    image: Any, threshold_mask: Any, processed_mask: Any
) -> FigureResult:
    """Compare aligned [H,W] images for original, threshold, and processed masking."""
    panels = [_array("image", image), _array("threshold_mask", threshold_mask), _array("processed_mask", processed_mask)]
    if any(panel.ndim != 2 for panel in panels) or len({panel.shape for panel in panels}) != 1:
        raise ValueError("All masking comparison inputs must be aligned [H,W] arrays.")
    fig, axes = plt.subplots(1, 3, constrained_layout=True)
    for ax, panel, title in zip(axes, panels, ("Image", "Threshold mask", "Processed mask"), strict=True):
        ax.imshow(panel, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    return fig, axes


def plot_loss_and_mean_dice(
    epochs: Any,
    train_loss: Any,
    mean_dice: Any,
    *,
    validation_loss: Any | None = None,
) -> FigureResult:
    """Plot aligned [E] loss and mean-Dice histories without recomputing metrics."""
    epoch_array = _array("epochs", epochs).astype(float, copy=False)
    train = _array("train_loss", train_loss).astype(float, copy=False)
    dice = _array("mean_dice", mean_dice).astype(float, copy=False)
    if any(value.ndim != 1 for value in (epoch_array, train, dice)):
        raise ValueError("epochs, train_loss, and mean_dice must be one-dimensional.")
    if not (epoch_array.size == train.size == dice.size):
        raise ValueError("epochs, train_loss, and mean_dice must have equal lengths.")
    val = None if validation_loss is None else _array("validation_loss", validation_loss).astype(float, copy=False)
    if val is not None and (val.ndim != 1 or val.size != epoch_array.size):
        raise ValueError("validation_loss must be [E] and align with epochs.")
    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    axes[0].plot(epoch_array, train, label="Training loss")
    if val is not None:
        axes[0].plot(epoch_array, val, label="Validation loss")
        axes[0].legend(loc="best")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss history")
    axes[1].plot(epoch_array, dice)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean Dice")
    axes[1].set_title("Mean Dice history")
    return fig, axes


def plot_classwise_dice(
    epochs: Any, dice_by_class: Any, *, class_names: Sequence[str] | None = None
) -> FigureResult:
    """Plot precomputed classwise Dice histories shaped [C,E] against epochs [E]."""
    epoch_array = _array("epochs", epochs).astype(float, copy=False)
    matrix = _array("dice_by_class", dice_by_class).astype(float, copy=False)
    if epoch_array.ndim != 1 or matrix.ndim != 2 or matrix.shape[1] != epoch_array.size:
        raise ValueError("epochs must be [E] and dice_by_class must be [C,E].")
    names = [f"Class {index}" for index in range(matrix.shape[0])] if class_names is None else [str(value) for value in class_names]
    if len(names) != matrix.shape[0]:
        raise ValueError("class_names must contain one name per Dice row.")
    fig, ax = plt.subplots(constrained_layout=True)
    for name, values in zip(names, matrix, strict=True):
        ax.plot(epoch_array, values, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.set_title("Classwise Dice history")
    ax.legend(loc="best")
    return fig, ax


__all__ = [
    "plot_classwise_dice",
    "plot_ground_truth_channels",
    "plot_image_and_mask",
    "plot_image_mask_prediction",
    "plot_input_channels",
    "plot_loss_and_mean_dice",
    "plot_masking_comparison",
    "plot_prediction_channels",
]
