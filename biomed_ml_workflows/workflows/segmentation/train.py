"""Generic SegResNet label, patch-boundary, reproducibility, and training contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeAlias

import torch
from monai.losses import DiceLoss
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..classification.train import (
    ReproducibilityRecord,
    SplitResult,
    configure_reproducibility,
    make_dataloader,
    split_samples,
)


LabelEncoding: TypeAlias = Literal["INTEGER_CLASS_MAP", "MULTICHANNEL"]


@dataclass(frozen=True)
class SegmentationLabelContract:
    """Explicit relationship between targets, model channels, loss, and Dice."""

    encoding: LabelEncoding
    out_channels: int
    include_background: bool = True
    prediction_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.encoding not in {"INTEGER_CLASS_MAP", "MULTICHANNEL"}:
            raise ValueError("encoding must be INTEGER_CLASS_MAP or MULTICHANNEL.")
        if self.encoding == "INTEGER_CLASS_MAP" and self.out_channels < 2:
            raise ValueError("Integer class maps require at least two output channels.")
        if self.encoding == "MULTICHANNEL" and self.out_channels < 1:
            raise ValueError("Multi-channel targets require at least one output channel.")
        if not self.include_background and self.out_channels == 1:
            raise ValueError("Excluding channel 0 requires more than one output channel.")
        if not 0.0 < self.prediction_threshold < 1.0:
            raise ValueError("prediction_threshold must be strictly between 0 and 1.")


@dataclass(frozen=True)
class PatchPartitionAudit:
    patch_count: int
    source_volume_count: int
    partition_patch_counts: dict[str, int]
    order: str = "PARTITION_BEFORE_PATCH_GENERATION"
    cross_partition_sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SegmentationEpochMetrics:
    loss: float
    sample_count: int
    batch_count: int


@dataclass(frozen=True)
class SegmentationEpochRecord:
    epoch: int
    train: SegmentationEpochMetrics
    validation: SegmentationEpochMetrics


@dataclass(frozen=True)
class SegmentationTrainingConfig:
    epochs: int = 1
    learning_rate: float = 1e-3
    early_stopping_patience: int | None = None
    min_delta: float = 0.0
    use_amp: bool = False

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive or None.")
        if self.min_delta < 0.0:
            raise ValueError("min_delta must be non-negative.")


@dataclass(frozen=True)
class SegmentationTrainingResult:
    history: tuple[SegmentationEpochRecord, ...]
    best_epoch: int
    best_validation_loss: float
    label_encoding: str
    loss_name: str
    amp_enabled: bool
    selection_partition: str = "validation"
    selection_metric: str = "validation_loss"
    checkpoint_storage: str = "IN_MEMORY_STATE_DICT"
    best_checkpoint_reloaded: bool = True
    test_data_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_patch_partitioning(
    split: SplitResult,
    patch_source_volume_ids: Sequence[str],
    patch_partitions: Sequence[str],
) -> PatchPartitionAudit:
    """Verify that patches remain in their source volume's prior partition.

    This function audits the required order ``volume -> partition -> patches``.
    It rejects unknown source volumes and any patch assigned to a different
    partition than the one established by ``split_samples``.
    """

    if len(patch_source_volume_ids) != len(patch_partitions):
        raise ValueError("Patch source IDs and partition names must have equal lengths.")
    expected = {
        **{sample_id: "train" for sample_id in split.train_ids},
        **{sample_id: "validation" for sample_id in split.validation_ids},
        **{sample_id: "test" for sample_id in split.test_ids},
    }
    permitted = {"train", "validation", "test"}
    observed_by_source: dict[str, set[str]] = {}
    counts = {name: 0 for name in permitted}
    for source_id, partition in zip(patch_source_volume_ids, patch_partitions, strict=True):
        source = str(source_id)
        if source not in expected:
            raise ValueError(f"Patch source volume {source!r} is absent from the prior split.")
        if partition not in permitted:
            raise ValueError(f"Unknown patch partition {partition!r}.")
        observed_by_source.setdefault(source, set()).add(partition)
        counts[partition] += 1
        if partition != expected[source]:
            raise ValueError(
                f"Patch from source volume {source!r} was assigned to {partition!r} "
                f"instead of its prior partition {expected[source]!r}."
            )
    crossed = tuple(sorted(source for source, values in observed_by_source.items() if len(values) > 1))
    if crossed:
        raise ValueError(f"Source volumes cross patch partitions: {crossed!r}.")
    return PatchPartitionAudit(
        patch_count=len(patch_source_volume_ids),
        source_volume_count=len(observed_by_source),
        partition_patch_counts={name: counts[name] for name in ("train", "validation", "test")},
    )


def prepare_segmentation_target(
    target: torch.Tensor,
    logits: torch.Tensor,
    contract: SegmentationLabelContract,
) -> torch.Tensor:
    """Validate and normalize a target without changing its semantic encoding."""

    if logits.ndim != 5 or logits.shape[1] != contract.out_channels:
        raise ValueError("Model logits must have shape [B, out_channels, D, H, W].")
    if contract.encoding == "INTEGER_CLASS_MAP":
        if target.ndim == 4:
            target = target.unsqueeze(1)
        if target.ndim != 5 or target.shape[1] != 1:
            raise ValueError("Integer class-map targets must have shape [B, 1, D, H, W].")
        if target.shape[0] != logits.shape[0] or target.shape[2:] != logits.shape[2:]:
            raise ValueError("Target batch and spatial dimensions must match model logits.")
        if target.is_floating_point() and not torch.equal(target, target.round()):
            raise ValueError("Integer class-map targets contain non-integer values.")
        normalized = target.to(dtype=torch.long)
        if normalized.numel() and (
            int(normalized.min().item()) < 0
            or int(normalized.max().item()) >= contract.out_channels
        ):
            raise ValueError("Integer class-map labels fall outside the declared output channels.")
        return normalized
    if target.ndim != 5 or target.shape != logits.shape:
        raise ValueError("Multi-channel targets must match [B, out_channels, D, H, W].")
    normalized = target.to(dtype=torch.float32)
    if normalized.numel() and (
        float(normalized.min().item()) < 0.0 or float(normalized.max().item()) > 1.0
    ):
        raise ValueError("Multi-channel targets must contain values in [0, 1].")
    return normalized


def build_default_segmentation_loss(
    contract: SegmentationLabelContract,
) -> nn.Module:
    """Return a documented encoding-compatible default loss.

    Integer maps use softmax Dice with one-hot conversion. Multi-channel
    targets use independent sigmoid Dice. This small generic default remains
    compatible with the strict deterministic CPU/CUDA policy used by the
    pilot; callers must explicitly inject a different loss when their
    scientific objective requires one.
    """

    if contract.encoding == "INTEGER_CLASS_MAP":
        return DiceLoss(
            include_background=contract.include_background,
            to_onehot_y=True,
            softmax=True,
        )
    return DiceLoss(
        include_background=contract.include_background,
        to_onehot_y=False,
        sigmoid=True,
    )


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, Mapping):
        if "image" not in batch or "label" not in batch:
            raise KeyError("Mapping batches must contain 'image' and 'label'.")
        return torch.as_tensor(batch["image"]), torch.as_tensor(batch["label"])
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return torch.as_tensor(batch[0]), torch.as_tensor(batch[1])
    raise TypeError("A batch must be a mapping or an image/label sequence.")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    contract: SegmentationLabelContract,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: Optimizer | None,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> SegmentationEpochMetrics:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    batch_count = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            images, raw_target = _unpack_batch(batch)
            images = images.to(device=device, dtype=torch.float32)
            raw_target = raw_target.to(device=device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                target = prepare_segmentation_target(raw_target, logits, contract)
                loss = loss_fn(logits, target)
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError("Segmentation loss must be a finite scalar.")
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            batch_size = images.shape[0]
            total_loss += float(loss.detach().item()) * batch_size
            total_samples += batch_size
            batch_count += 1
    if total_samples == 0:
        raise ValueError("DataLoader produced zero samples.")
    return SegmentationEpochMetrics(
        loss=total_loss / total_samples,
        sample_count=total_samples,
        batch_count=batch_count,
    )


def fit_segmenter(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    label_contract: SegmentationLabelContract,
    device: str | torch.device,
    config: SegmentationTrainingConfig | None = None,
    optimizer: Optimizer | None = None,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> SegmentationTrainingResult:
    """Train and reload an in-memory checkpoint selected on validation loss only.

    This API deliberately accepts no test loader. Held-out test evaluation is a
    separate call after validation-based selection has completed.
    """

    resolved_config = config or SegmentationTrainingConfig()
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if resolved_config.use_amp and resolved_device.type != "cuda":
        raise ValueError("AMP is supported only for the explicit CUDA execution path.")
    model.to(resolved_device)
    resolved_optimizer = optimizer or torch.optim.Adam(
        model.parameters(), lr=resolved_config.learning_rate
    )
    resolved_loss = loss_fn or build_default_segmentation_loss(label_contract)
    scaler = torch.amp.GradScaler("cuda", enabled=resolved_config.use_amp)
    history: list[SegmentationEpochRecord] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, resolved_config.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=resolved_device,
            contract=label_contract,
            loss_fn=resolved_loss,
            optimizer=resolved_optimizer,
            scaler=scaler,
            use_amp=resolved_config.use_amp,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            device=resolved_device,
            contract=label_contract,
            loss_fn=resolved_loss,
            optimizer=None,
            scaler=scaler,
            use_amp=resolved_config.use_amp,
        )
        history.append(
            SegmentationEpochRecord(
                epoch=epoch,
                train=train_metrics,
                validation=validation_metrics,
            )
        )
        if validation_metrics.loss < best_validation_loss - resolved_config.min_delta:
            best_validation_loss = validation_metrics.loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if (
            resolved_config.early_stopping_patience is not None
            and epochs_without_improvement >= resolved_config.early_stopping_patience
        ):
            break

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected.")
    model.load_state_dict(best_state)
    model.to(resolved_device)
    return SegmentationTrainingResult(
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        label_encoding=label_contract.encoding,
        loss_name=type(resolved_loss).__name__,
        amp_enabled=resolved_config.use_amp,
    )
