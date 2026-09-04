"""Leakage-aware CoxPH data boundaries and minimal training."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
import pycox
import torch
import torchtuples
from pycox.models import CoxPH
from pycox.models.loss import CoxPHLoss

from ..classification.train import (
    SplitResult,
    configure_reproducibility,
    split_samples,
)


@dataclass(frozen=True)
class SurvivalDataset:
    """Validated baseline covariates and right-censored single-event outcomes."""

    features: np.ndarray
    durations: np.ndarray
    events: np.ndarray
    partition: str


@dataclass(frozen=True)
class PreprocessedPartitions:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray | None
    fit_partition: str = "train"
    validation_fit_used: bool = False
    test_fit_used: bool = False


@dataclass(frozen=True)
class CoxPHReproducibilityRecord:
    python: str
    torch: str
    torchvision: str
    monai: str
    numpy: str
    scikit_learn: str
    pycox: str
    torchtuples: str
    pandas: str
    cuda_runtime: str
    device: str
    device_name: str
    seed: int
    deterministic_algorithms: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    scope: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoxPHTrainingConfig:
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer: Literal["adam", "sgd"] = "adam"
    early_stopping_patience: int | None = None
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if self.optimizer not in {"adam", "sgd"}:
            raise ValueError("optimizer must be 'adam' or 'sgd'.")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive or None.")
        if self.min_delta < 0.0:
            raise ValueError("min_delta must be non-negative.")


@dataclass(frozen=True)
class CoxPHEpochRecord:
    epoch: int
    train_loss: float
    validation_loss: float


@dataclass(frozen=True)
class CoxPHTrainingResult:
    history: tuple[CoxPHEpochRecord, ...]
    best_epoch: int
    best_validation_loss: float
    optimizer: str
    full_training_partition_per_update: bool = True
    selection_partition: str = "validation"
    checkpoint_storage: str = "IN_MEMORY_STATE_DICT"
    best_checkpoint_reloaded: bool = True
    test_data_accepted_by_training_api: bool = False
    test_data_used: bool = False
    baseline_hazard_partition: str = "train"
    baseline_hazard_uses_training_outcomes_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _numpy_array(values: Any) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _features_array(features: Any, *, partition: str) -> np.ndarray:
    array = np.asarray(_numpy_array(features), dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{partition} features must have shape [samples, features].")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{partition} features must be non-empty.")
    if not np.isfinite(array).all():
        raise ValueError(
            f"{partition} features contain missing or infinite values; this workflow does not impute."
        )
    return np.ascontiguousarray(array)


def _outcomes_array(
    durations: Any,
    events: Any,
    *,
    partition: str,
    require_event: bool,
) -> tuple[np.ndarray, np.ndarray]:
    duration_array = np.asarray(_numpy_array(durations), dtype=np.float32)
    event_input = _numpy_array(events)
    if duration_array.ndim != 1 or event_input.ndim != 1:
        raise ValueError(f"{partition} durations and events must be one-dimensional.")
    if not np.isfinite(duration_array).all() or np.any(duration_array < 0.0):
        raise ValueError(f"{partition} durations must be finite and non-negative.")
    if event_input.dtype.kind not in "biuf" or not np.isfinite(event_input).all():
        raise ValueError(f"{partition} events must be finite numeric values exactly in {{0, 1}}.")
    if not np.isin(event_input, [0, 1]).all():
        raise ValueError(
            f"{partition} events must be exactly 1 for observed events or 0 for right censoring."
        )
    event_array = event_input.astype(np.float32, copy=False)
    if require_event and int(event_array.sum()) < 1:
        raise ValueError(f"{partition} must contain at least one observed event.")
    return np.ascontiguousarray(duration_array), np.ascontiguousarray(event_array)


def validate_survival_dataset(
    features: Any,
    durations: Any,
    events: Any,
    *,
    partition: str,
    require_event: bool = True,
) -> SurvivalDataset:
    """Fail fast on shape, missingness, duration, and event-encoding errors."""

    feature_array = _features_array(features, partition=partition)
    duration_array, event_array = _outcomes_array(
        durations, events, partition=partition, require_event=require_event
    )
    if not (feature_array.shape[0] == duration_array.shape[0] == event_array.shape[0]):
        raise ValueError(f"{partition} feature and outcome sample counts must match.")
    return SurvivalDataset(feature_array, duration_array, event_array, partition)


def split_survival_samples(
    sample_ids: Sequence[str],
    events: Sequence[int],
    *,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    group_ids: Sequence[str] | None = None,
    stratify: bool = True,
    require_groups: bool = False,
) -> SplitResult:
    """Reuse the frozen canonical split utility, stratifying by event when valid."""

    return split_samples(
        sample_ids,
        events,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
        group_ids=group_ids,
        stratify=stratify,
        require_groups=require_groups,
    )


def fit_train_only_preprocessor(
    preprocessor: Any,
    train_features: Any,
    validation_features: Any,
    test_features: Any | None = None,
) -> PreprocessedPartitions:
    """Fit a caller-supplied transformer on training covariates only."""

    if not callable(getattr(preprocessor, "fit", None)) or not callable(
        getattr(preprocessor, "transform", None)
    ):
        raise TypeError("preprocessor must provide fit and transform methods.")
    train = _features_array(train_features, partition="train")
    validation = _features_array(validation_features, partition="validation")
    test = None if test_features is None else _features_array(test_features, partition="test")
    if train.shape[1] != validation.shape[1] or (test is not None and train.shape[1] != test.shape[1]):
        raise ValueError("All partitions must have the same input feature count.")
    preprocessor.fit(train)
    transformed_train = _features_array(preprocessor.transform(train), partition="train transformed")
    transformed_validation = _features_array(
        preprocessor.transform(validation), partition="validation transformed"
    )
    transformed_test = (
        None
        if test is None
        else _features_array(preprocessor.transform(test), partition="test transformed")
    )
    return PreprocessedPartitions(transformed_train, transformed_validation, transformed_test)


def configure_coxph_reproducibility(
    *, seed: int, device: str | torch.device, deterministic_algorithms: bool = True
) -> CoxPHReproducibilityRecord:
    """Apply the frozen seed policy and add survival-stack metadata."""

    base = configure_reproducibility(
        seed=seed, device=device, deterministic_algorithms=deterministic_algorithms
    )
    return CoxPHReproducibilityRecord(
        **base.to_dict(),
        pycox=pycox.__version__,
        torchtuples=torchtuples.__version__,
        pandas=pd.__version__,
    )


def _tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def fit_coxph(
    model: CoxPH,
    train_features: Any,
    train_durations: Any,
    train_events: Any,
    validation_features: Any,
    validation_durations: Any,
    validation_events: Any,
    *,
    config: CoxPHTrainingConfig | None = None,
) -> CoxPHTrainingResult:
    """Train on one full risk set and select/reload by validation loss only.

    The API deliberately accepts no test partition. After selection, baseline
    hazards are computed with explicit training covariates and outcomes only.
    """

    resolved = config or CoxPHTrainingConfig()
    train = validate_survival_dataset(
        train_features, train_durations, train_events, partition="train"
    )
    validation = validate_survival_dataset(
        validation_features,
        validation_durations,
        validation_events,
        partition="validation",
    )
    if train.features.shape[1] != validation.features.shape[1]:
        raise ValueError("Training and validation feature counts must match.")
    device = next(model.net.parameters()).device
    optimizer_class = torch.optim.Adam if resolved.optimizer == "adam" else torch.optim.SGD
    optimizer = optimizer_class(
        model.net.parameters(),
        lr=resolved.learning_rate,
        weight_decay=resolved.weight_decay,
    )
    loss_function = CoxPHLoss()
    train_x = _tensor(train.features, device)
    train_d = _tensor(train.durations, device)
    train_e = _tensor(train.events, device)
    validation_x = _tensor(validation.features, device)
    validation_d = _tensor(validation.durations, device)
    validation_e = _tensor(validation.events, device)
    history: list[CoxPHEpochRecord] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, resolved.epochs + 1):
        model.net.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss_tensor = loss_function(model.net(train_x), train_d, train_e)
        if not torch.isfinite(train_loss_tensor):
            raise RuntimeError("Training loss became non-finite.")
        train_loss_tensor.backward()
        optimizer.step()

        model.net.eval()
        with torch.inference_mode():
            validation_loss_tensor = loss_function(
                model.net(validation_x), validation_d, validation_e
            )
        if not torch.isfinite(validation_loss_tensor):
            raise RuntimeError("Validation loss became non-finite.")
        train_loss = float(train_loss_tensor.detach().item())
        validation_loss = float(validation_loss_tensor.detach().item())
        history.append(CoxPHEpochRecord(epoch, train_loss, validation_loss))
        if validation_loss < best_validation_loss - resolved.min_delta:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(
                {name: value.detach().cpu() for name, value in model.net.state_dict().items()}
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if (
            resolved.early_stopping_patience is not None
            and epochs_without_improvement >= resolved.early_stopping_patience
        ):
            break

    if best_state is None:
        raise RuntimeError("No validation-selected model state was produced.")
    model.net.load_state_dict(best_state)
    model.net.to(device)
    model.net.eval()
    model.compute_baseline_hazards(
        input=train.features,
        target=(train.durations, train.events),
        set_hazards=True,
        eval_=True,
    )
    if model.baseline_hazards_ is None or len(model.baseline_hazards_) == 0:
        raise RuntimeError("Training baseline-hazard estimation produced no values.")
    return CoxPHTrainingResult(
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        optimizer=resolved.optimizer,
    )
