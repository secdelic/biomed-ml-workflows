"""Generic held-out evaluation for DenseNet121 classification models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    auroc: float | None
    auroc_status: str


@dataclass(frozen=True)
class EvaluationResult:
    logits: torch.Tensor
    probabilities: torch.Tensor
    predicted_classes: torch.Tensor
    labels: torch.Tensor
    sample_ids: tuple[str, ...] | None
    metrics: ClassificationMetrics
    probability_interpretation: str = "UNCALIBRATED_MODEL_SCORES_NOT_CLINICAL_PROBABILITIES"


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...] | None]:
    if isinstance(batch, Mapping):
        if "image" not in batch or "label" not in batch:
            raise KeyError("Mapping batches must contain 'image' and 'label'.")
        identifiers = batch.get("sample_id")
        if identifiers is None:
            sample_ids = None
        elif isinstance(identifiers, str):
            sample_ids = (identifiers,)
        else:
            sample_ids = tuple(str(value) for value in identifiers)
        return torch.as_tensor(batch["image"]), torch.as_tensor(batch["label"]), sample_ids
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return torch.as_tensor(batch[0]), torch.as_tensor(batch[1]), None
    raise TypeError("A batch must be a mapping or an image/label sequence.")


def classification_metrics(
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    predicted_classes: torch.Tensor,
) -> ClassificationMetrics:
    """Compute accuracy and AUROC only when AUROC is mathematically defined."""

    labels_np = labels.detach().cpu().numpy().astype(int, copy=False)
    probabilities_np = probabilities.detach().cpu().numpy()
    predictions_np = predicted_classes.detach().cpu().numpy().astype(int, copy=False)
    accuracy = float(accuracy_score(labels_np, predictions_np))
    class_count = probabilities_np.shape[1]
    observed = np.unique(labels_np)
    if observed.size != class_count or not np.array_equal(observed, np.arange(class_count)):
        return ClassificationMetrics(
            accuracy=accuracy,
            auroc=None,
            auroc_status="UNDEFINED_REQUIRED_CLASS_ABSENT",
        )
    if class_count == 2:
        auroc = float(roc_auc_score(labels_np, probabilities_np[:, 1]))
    else:
        auroc = float(
            roc_auc_score(
                labels_np,
                probabilities_np,
                multi_class="ovr",
                average="macro",
                labels=np.arange(class_count),
            )
        )
    return ClassificationMetrics(accuracy=accuracy, auroc=auroc, auroc_status="COMPUTED")


def predict_classifier(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str | torch.device,
) -> EvaluationResult:
    """Return held-out logits, softmax scores, predictions, labels, IDs, metrics."""

    resolved_device = torch.device(device)
    model.to(resolved_device)
    model.eval()
    logits_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    identifiers: list[str] = []
    identifiers_present: bool | None = None
    with torch.inference_mode():
        for batch in loader:
            images, labels, sample_ids = _unpack_batch(batch)
            images = images.to(device=resolved_device, dtype=torch.float32)
            labels = labels.to(dtype=torch.long).reshape(-1).cpu()
            logits = model(images).detach().cpu()
            if logits.ndim != 2 or logits.shape[0] != labels.shape[0] or logits.shape[1] < 2:
                raise ValueError("Model output must have shape [batch, classes] with at least 2 classes.")
            present = sample_ids is not None
            if identifiers_present is None:
                identifiers_present = present
            elif identifiers_present != present:
                raise ValueError("sample_id must be present for all batches or none.")
            if sample_ids is not None:
                if len(sample_ids) != labels.numel():
                    raise ValueError("sample_id count must match batch size.")
                identifiers.extend(sample_ids)
            logits_parts.append(logits)
            label_parts.append(labels)
    if not logits_parts:
        raise ValueError("DataLoader produced zero samples.")
    logits = torch.cat(logits_parts)
    labels = torch.cat(label_parts)
    probabilities = torch.softmax(logits, dim=1)
    predicted_classes = probabilities.argmax(dim=1)
    if identifiers_present and len(set(identifiers)) != len(identifiers):
        raise ValueError("sample_id values must be unique for sample-level mapping.")
    metrics = classification_metrics(labels, probabilities, predicted_classes)
    return EvaluationResult(
        logits=logits,
        probabilities=probabilities,
        predicted_classes=predicted_classes,
        labels=labels,
        sample_ids=tuple(identifiers) if identifiers_present else None,
        metrics=metrics,
    )


def evaluate_classifier(
    model: nn.Module,
    test_loader: DataLoader,
    *,
    device: str | torch.device,
) -> EvaluationResult:
    """Evaluate once on a held-out partition after validation-only selection."""

    return predict_classifier(model, test_loader, device=device)
