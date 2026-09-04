"""Leakage-aware splitting, determinism, and minimal supervised training."""

from __future__ import annotations

import os
import platform
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from math import floor, isclose
from typing import Any

import monai
import numpy as np
import sklearn
import torch
from monai.data import DataLoader
from monai.utils import set_determinism
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SplitResult:
    """Inspectible split indices, identifiers, groups, and achieved fractions."""

    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    train_group_ids: tuple[str, ...]
    validation_group_ids: tuple[str, ...]
    test_group_ids: tuple[str, ...]
    seed: int
    stratified: bool
    grouping_applied: bool
    requested_fractions: dict[str, float]
    observed_sample_fractions: dict[str, float]
    observed_unit_fractions: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation suitable for persistence."""

        return asdict(self)


@dataclass(frozen=True)
class ReproducibilityRecord:
    python: str
    torch: str
    torchvision: str
    monai: str
    numpy: str
    scikit_learn: str
    cuda_runtime: str
    device: str
    device_name: str
    seed: int
    deterministic_algorithms: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    scope: str = "SAME_SOFTWARE_AND_HARDWARE_ONLY"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    sample_count: int


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train: EpochMetrics
    validation: EpochMetrics


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 1
    learning_rate: float = 1e-3
    early_stopping_patience: int | None = None
    min_delta: float = 0.0

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
class TrainingResult:
    history: tuple[EpochRecord, ...]
    best_epoch: int
    best_validation_loss: float
    selection_partition: str = "validation"
    checkpoint_storage: str = "IN_MEMORY_STATE_DICT"
    best_checkpoint_reloaded: bool = True
    test_data_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _partition_counts(n_units: int, fractions: Sequence[float]) -> tuple[int, int, int]:
    if n_units < 3:
        raise ValueError("At least three split units are required.")
    raw = [n_units * value for value in fractions]
    counts = [max(1, floor(value)) for value in raw]
    while sum(counts) < n_units:
        index = max(range(3), key=lambda i: (raw[i] - counts[i], fractions[i], -i))
        counts[index] += 1
    while sum(counts) > n_units:
        candidates = [i for i, count in enumerate(counts) if count > 1]
        if not candidates:
            raise ValueError("Requested fractions cannot produce three non-empty partitions.")
        index = max(candidates, key=lambda i: (counts[i] - raw[i], -fractions[i], i))
        counts[index] -= 1
    return counts[0], counts[1], counts[2]


def split_samples(
    sample_ids: Sequence[str],
    labels: Sequence[int | str],
    *,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    group_ids: Sequence[str] | None = None,
    stratify: bool = True,
    require_groups: bool = False,
) -> SplitResult:
    """Split samples deterministically, using groups as indivisible units.

    When groups are supplied, requested proportions apply to groups. Unequal
    group sizes can therefore yield different sample-level fractions. A
    sample-level split must never be described as patient-level separation.
    """

    n_samples = len(sample_ids)
    if n_samples != len(labels) or (group_ids is not None and len(group_ids) != n_samples):
        raise ValueError("sample_ids, labels, and group_ids must have equal lengths.")
    if n_samples == 0:
        raise ValueError("At least one sample is required.")
    if len(set(sample_ids)) != n_samples:
        raise ValueError("sample_ids must be unique.")
    if require_groups and group_ids is None:
        raise ValueError("This study requires group IDs; sample-level splitting is not allowed.")
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(not np.isfinite(value) or value <= 0.0 for value in fractions):
        raise ValueError("All split fractions must be finite and positive.")
    if not isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Split fractions must sum to 1.0.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    if group_ids is None:
        unit_names = [str(index) for index in range(n_samples)]
        unit_labels = list(labels)
        unit_to_indices = {str(index): [index] for index in range(n_samples)}
    else:
        unit_to_indices: dict[str, list[int]] = {}
        for index, group in enumerate(group_ids):
            unit_to_indices.setdefault(str(group), []).append(index)
        unit_names = sorted(unit_to_indices)
        unit_labels = []
        for group in unit_names:
            observed = {labels[index] for index in unit_to_indices[group]}
            if stratify and len(observed) != 1:
                raise ValueError(
                    f"Group {group!r} contains multiple labels; a stratification label cannot be inferred."
                )
            unit_labels.append(next(iter(observed)) if len(observed) == 1 else "UNSTRATIFIED")

    train_count, validation_count, test_count = _partition_counts(len(unit_names), fractions)
    stratification_labels = unit_labels if stratify else None
    try:
        train_units, holdout_units = train_test_split(
            unit_names,
            train_size=train_count,
            test_size=validation_count + test_count,
            random_state=seed,
            shuffle=True,
            stratify=stratification_labels,
        )
        label_by_unit = dict(zip(unit_names, unit_labels, strict=True))
        holdout_labels = [label_by_unit[unit] for unit in holdout_units] if stratify else None
        validation_units, test_units = train_test_split(
            holdout_units,
            train_size=validation_count,
            test_size=test_count,
            random_state=seed + 1,
            shuffle=True,
            stratify=holdout_labels,
        )
    except ValueError as error:
        raise ValueError(
            "The requested stratified split is not technically valid for the available class/unit counts."
        ) from error

    def indices_for(units: Sequence[str]) -> tuple[int, ...]:
        return tuple(sorted(index for unit in units for index in unit_to_indices[unit]))

    train_indices = indices_for(train_units)
    validation_indices = indices_for(validation_units)
    test_indices = indices_for(test_units)
    index_sets = [set(train_indices), set(validation_indices), set(test_indices)]
    if any(index_sets[i] & index_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Internal split error: sample overlap detected.")
    if set().union(*index_sets) != set(range(n_samples)):
        raise RuntimeError("Internal split error: samples are missing from the partition union.")
    unit_sets = [set(train_units), set(validation_units), set(test_units)]
    if any(unit_sets[i] & unit_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Internal split error: group/unit overlap detected.")

    sample_counts = (len(train_indices), len(validation_indices), len(test_indices))
    unit_counts = (len(train_units), len(validation_units), len(test_units))
    names = ("train", "validation", "test")
    return SplitResult(
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        train_ids=tuple(sample_ids[index] for index in train_indices),
        validation_ids=tuple(sample_ids[index] for index in validation_indices),
        test_ids=tuple(sample_ids[index] for index in test_indices),
        train_group_ids=tuple(sorted(train_units)) if group_ids is not None else (),
        validation_group_ids=tuple(sorted(validation_units)) if group_ids is not None else (),
        test_group_ids=tuple(sorted(test_units)) if group_ids is not None else (),
        seed=seed,
        stratified=stratify,
        grouping_applied=group_ids is not None,
        requested_fractions=dict(zip(names, fractions, strict=True)),
        observed_sample_fractions={
            name: count / n_samples for name, count in zip(names, sample_counts, strict=True)
        },
        observed_unit_fractions={
            name: count / len(unit_names) for name, count in zip(names, unit_counts, strict=True)
        },
    )


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "NOT_INSTALLED"


def configure_reproducibility(
    *,
    seed: int,
    device: str | torch.device,
    deterministic_algorithms: bool = True,
) -> ReproducibilityRecord:
    """Apply one explicit same-environment reproducibility policy."""

    if not 0 <= seed <= np.iinfo(np.uint32).max:
        raise ValueError("seed must fit in an unsigned 32-bit integer.")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if deterministic_algorithms and resolved_device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed, use_deterministic_algorithms=deterministic_algorithms)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic_algorithms
    device_name = (
        torch.cuda.get_device_name(resolved_device)
        if resolved_device.type == "cuda"
        else platform.processor() or "CPU"
    )
    return ReproducibilityRecord(
        python=platform.python_version(),
        torch=torch.__version__,
        torchvision=_package_version("torchvision"),
        monai=monai.__version__,
        numpy=np.__version__,
        scikit_learn=sklearn.__version__,
        cuda_runtime=torch.version.cuda or "NOT_AVAILABLE",
        device=str(resolved_device),
        device_name=device_name,
        seed=seed,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
    )


def make_dataloader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    """Create a MONAI DataLoader with an explicit sampling generator."""

    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, Mapping):
        if "image" not in batch or "label" not in batch:
            raise KeyError("Mapping batches must contain 'image' and 'label'.")
        images, labels = batch["image"], batch["label"]
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        images, labels = batch[0], batch[1]
    else:
        raise TypeError("A batch must be a mapping or an image/label sequence.")
    return torch.as_tensor(images), torch.as_tensor(labels)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: Optimizer | None,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            images, labels = _unpack_batch(batch)
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long).reshape(-1)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if logits.ndim != 2 or logits.shape[0] != labels.shape[0]:
                raise ValueError("Model output must have shape [batch, classes].")
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = labels.numel()
            total_loss += float(loss.detach().item()) * batch_size
            total_correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
            total_samples += batch_size
    if total_samples == 0:
        raise ValueError("DataLoader produced zero samples.")
    return EpochMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        sample_count=total_samples,
    )


def fit_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    device: str | torch.device,
    config: TrainingConfig | None = None,
    optimizer: Optimizer | None = None,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> TrainingResult:
    """Fit on training data and select/reload a validation-only checkpoint.

    No test loader is accepted by this API. Test evaluation must occur only
    after this function returns with the validation-selected state reloaded.
    """

    resolved_config = config or TrainingConfig()
    resolved_device = torch.device(device)
    model.to(resolved_device)
    resolved_optimizer = optimizer or torch.optim.Adam(
        model.parameters(), lr=resolved_config.learning_rate
    )
    resolved_loss = loss_fn or nn.CrossEntropyLoss()
    history: list[EpochRecord] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, resolved_config.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=resolved_device,
            loss_fn=resolved_loss,
            optimizer=resolved_optimizer,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            device=resolved_device,
            loss_fn=resolved_loss,
            optimizer=None,
        )
        history.append(EpochRecord(epoch=epoch, train=train_metrics, validation=validation_metrics))
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
    return TrainingResult(
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
    )
