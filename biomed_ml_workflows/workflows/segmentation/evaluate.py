"""Generic held-out prediction, sliding-window inference, and Dice for SegResNet."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
from monai.inferers import sliding_window_inference
from monai.networks.utils import one_hot
from torch import nn
from torch.utils.data import DataLoader

from .train import SegmentationLabelContract, prepare_segmentation_target


@dataclass(frozen=True)
class SlidingWindowConfig:
    roi_size: tuple[int, int, int]
    overlap: float = 0.25
    sw_batch_size: int = 1

    def __post_init__(self) -> None:
        if len(self.roi_size) != 3 or any(value < 1 for value in self.roi_size):
            raise ValueError("roi_size must contain three positive integers.")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("overlap must satisfy 0 <= overlap < 1.")
        if self.sw_batch_size < 1:
            raise ValueError("sw_batch_size must be positive.")


@dataclass(frozen=True)
class DiceChannelResult:
    channel_index: int
    dice: float | None
    status: str
    valid_sample_count: int
    total_sample_count: int


@dataclass(frozen=True)
class SegmentationDiceResult:
    mean_dice: float | None
    channels: tuple[DiceChannelResult, ...]
    empty_target_policy: str = "UNDEFINED_WHEN_TARGET_AND_PREDICTION_ARE_BOTH_EMPTY"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SegmentationEvaluationResult:
    logits: torch.Tensor
    probabilities: torch.Tensor
    predictions: torch.Tensor
    labels: torch.Tensor
    sample_ids: tuple[str, ...] | None
    dice: SegmentationDiceResult
    label_encoding: str
    inference_mode: str
    score_interpretation: str = "SEGMENTATION_SCORES_NOT_CLINICAL_UTILITY_OR_EXTERNAL_VALIDITY"


def infer_segmentation_logits(
    model: nn.Module,
    images: torch.Tensor,
    *,
    device: str | torch.device,
    sliding_window: SlidingWindowConfig | None = None,
    use_amp: bool = False,
) -> torch.Tensor:
    """Return logits by direct or explicit sliding-window inference."""

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if use_amp and resolved_device.type != "cuda":
        raise ValueError("AMP inference is supported only on the explicit CUDA path.")
    if images.ndim != 5:
        raise ValueError("Images must have shape [B, C, D, H, W].")
    model.to(resolved_device)
    model.eval()
    resolved_images = images.to(device=resolved_device, dtype=torch.float32)
    with torch.inference_mode(), torch.amp.autocast(
        device_type=resolved_device.type,
        enabled=use_amp,
    ):
        if sliding_window is None:
            output = model(resolved_images)
        else:
            output = sliding_window_inference(
                inputs=resolved_images,
                roi_size=sliding_window.roi_size,
                sw_batch_size=sliding_window.sw_batch_size,
                predictor=model,
                overlap=sliding_window.overlap,
            )
    if not isinstance(output, torch.Tensor) or output.ndim != 5:
        raise ValueError("Segmentation predictor must return one [B, C, D, H, W] tensor.")
    return output


def _probabilities_and_predictions(
    logits: torch.Tensor,
    contract: SegmentationLabelContract,
) -> tuple[torch.Tensor, torch.Tensor]:
    if contract.encoding == "INTEGER_CLASS_MAP":
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1, keepdim=True)
    else:
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= contract.prediction_threshold).to(dtype=torch.float32)
    return probabilities, predictions


def segmentation_dice(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    contract: SegmentationLabelContract,
) -> SegmentationDiceResult:
    """Compute explicit class/channel Dice with undefined empty channels retained."""

    if contract.encoding == "INTEGER_CLASS_MAP":
        prediction_channels = one_hot(predictions.to(dtype=torch.long), contract.out_channels)
        label_channels = one_hot(labels.to(dtype=torch.long), contract.out_channels)
    else:
        prediction_channels = predictions.to(dtype=torch.float32)
        label_channels = labels.to(dtype=torch.float32)
    channel_indices = list(range(contract.out_channels))
    if not contract.include_background:
        channel_indices = channel_indices[1:]
    channel_results: list[DiceChannelResult] = []
    defined_values: list[float] = []
    spatial_dims = tuple(range(2, prediction_channels.ndim))
    for channel_index in channel_indices:
        prediction = prediction_channels[:, channel_index]
        target = label_channels[:, channel_index]
        intersection = (prediction * target).sum(dim=tuple(axis - 1 for axis in spatial_dims))
        denominator = prediction.sum(dim=tuple(axis - 1 for axis in spatial_dims)) + target.sum(
            dim=tuple(axis - 1 for axis in spatial_dims)
        )
        valid = denominator > 0
        valid_count = int(valid.sum().item())
        total_count = int(denominator.numel())
        if valid_count == 0:
            channel_results.append(
                DiceChannelResult(
                    channel_index=channel_index,
                    dice=None,
                    status="UNDEFINED_EMPTY_TARGET_AND_PREDICTION",
                    valid_sample_count=0,
                    total_sample_count=total_count,
                )
            )
        else:
            values = (2.0 * intersection[valid]) / denominator[valid]
            value = float(values.mean().item())
            defined_values.append(value)
            channel_results.append(
                DiceChannelResult(
                    channel_index=channel_index,
                    dice=value,
                    status="COMPUTED",
                    valid_sample_count=valid_count,
                    total_sample_count=total_count,
                )
            )
    return SegmentationDiceResult(
        mean_dice=(sum(defined_values) / len(defined_values)) if defined_values else None,
        channels=tuple(channel_results),
    )


def _unpack_batch(
    batch: Any,
) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...] | None]:
    if isinstance(batch, Mapping):
        if "image" not in batch or "label" not in batch:
            raise KeyError("Mapping batches must contain 'image' and 'label'.")
        raw_ids = batch.get("sample_id")
        if raw_ids is None:
            sample_ids = None
        elif isinstance(raw_ids, str):
            sample_ids = (raw_ids,)
        else:
            sample_ids = tuple(str(value) for value in raw_ids)
        return torch.as_tensor(batch["image"]), torch.as_tensor(batch["label"]), sample_ids
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return torch.as_tensor(batch[0]), torch.as_tensor(batch[1]), None
    raise TypeError("A batch must be a mapping or an image/label sequence.")


def predict_segmenter(
    model: nn.Module,
    loader: DataLoader,
    *,
    label_contract: SegmentationLabelContract,
    device: str | torch.device,
    sliding_window: SlidingWindowConfig | None = None,
    use_amp: bool = False,
) -> SegmentationEvaluationResult:
    """Return reusable held-out logits, scores, masks, labels, IDs, and Dice."""

    logits_parts: list[torch.Tensor] = []
    probability_parts: list[torch.Tensor] = []
    prediction_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    identifiers: list[str] = []
    identifiers_present: bool | None = None
    for batch in loader:
        images, raw_labels, sample_ids = _unpack_batch(batch)
        logits_device = infer_segmentation_logits(
            model,
            images,
            device=device,
            sliding_window=sliding_window,
            use_amp=use_amp,
        )
        labels_device = prepare_segmentation_target(
            raw_labels.to(logits_device.device), logits_device, label_contract
        )
        probabilities_device, predictions_device = _probabilities_and_predictions(
            logits_device, label_contract
        )
        present = sample_ids is not None
        if identifiers_present is None:
            identifiers_present = present
        elif identifiers_present != present:
            raise ValueError("sample_id must be present for all batches or none.")
        if sample_ids is not None:
            if len(sample_ids) != images.shape[0]:
                raise ValueError("sample_id count must match batch size.")
            identifiers.extend(sample_ids)
        logits_parts.append(logits_device.detach().cpu())
        probability_parts.append(probabilities_device.detach().cpu())
        prediction_parts.append(predictions_device.detach().cpu())
        label_parts.append(labels_device.detach().cpu())
    if not logits_parts:
        raise ValueError("DataLoader produced zero samples.")
    if identifiers_present and len(set(identifiers)) != len(identifiers):
        raise ValueError("sample_id values must be unique for sample-level mapping.")
    logits = torch.cat(logits_parts)
    probabilities = torch.cat(probability_parts)
    predictions = torch.cat(prediction_parts)
    labels = torch.cat(label_parts)
    return SegmentationEvaluationResult(
        logits=logits,
        probabilities=probabilities,
        predictions=predictions,
        labels=labels,
        sample_ids=tuple(identifiers) if identifiers_present else None,
        dice=segmentation_dice(predictions, labels, label_contract),
        label_encoding=label_contract.encoding,
        inference_mode="SLIDING_WINDOW" if sliding_window is not None else "DIRECT",
    )


def evaluate_segmenter(
    model: nn.Module,
    test_loader: DataLoader,
    *,
    label_contract: SegmentationLabelContract,
    device: str | torch.device,
    sliding_window: SlidingWindowConfig | None = None,
    use_amp: bool = False,
) -> SegmentationEvaluationResult:
    """Evaluate only after validation-selected model state has been reloaded."""

    return predict_segmenter(
        model,
        test_loader,
        label_contract=label_contract,
        device=device,
        sliding_window=sliding_window,
        use_amp=use_amp,
    )
