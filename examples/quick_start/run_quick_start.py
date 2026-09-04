"""Run all three workflows with fixed synthetic inputs.

Outputs are technical demonstrations only and are not scientific performance
evidence. No network, external dataset, or pretrained weight is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from biomed_ml_workflows.figures.classification import plot_roc_curve
from biomed_ml_workflows.figures.segmentation import plot_image_mask_prediction
from biomed_ml_workflows.figures.survival import plot_survival_curves
from biomed_ml_workflows.methods.classification import build_densenet121
from biomed_ml_workflows.methods.segmentation import build_segresnet
from biomed_ml_workflows.methods.survival import build_coxph_model
from biomed_ml_workflows.workflows.classification import (
    TrainingConfig,
    configure_reproducibility,
    evaluate_classifier,
    fit_classifier,
    split_samples,
)
from biomed_ml_workflows.workflows.segmentation import (
    SegmentationLabelContract,
    SegmentationTrainingConfig,
    SlidingWindowConfig,
    evaluate_segmenter,
    fit_segmenter,
)
from biomed_ml_workflows.workflows.survival import (
    CoxPHTrainingConfig,
    configure_coxph_reproducibility,
    evaluate_coxph,
    fit_coxph,
    fit_train_only_preprocessor,
    split_survival_samples,
)


def _prepare_output(root: Path) -> dict[str, Path]:
    paths = {name: root / name for name in ("metrics", "predictions", "figures", "logs")}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _classification(seed: int, paths: dict[str, Path]) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    images = rng.normal(size=(16, 1, 32, 32)).astype(np.float32)
    labels = np.asarray([0, 1] * 8, dtype=np.int64)
    sample_ids = [f"synthetic-image-{index:02d}" for index in range(16)]
    split = split_samples(
        sample_ids,
        labels.tolist(),
        group_ids=[f"synthetic-group-{index:02d}" for index in range(16)],
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=seed,
    )
    tensor_images = torch.from_numpy(images)
    tensor_labels = torch.from_numpy(labels)
    loader = lambda indices, batch: DataLoader(
        TensorDataset(tensor_images[list(indices)], tensor_labels[list(indices)]),
        batch_size=batch,
        shuffle=False,
    )
    model = build_densenet121(spatial_dims=2, in_channels=1, out_channels=2, device="cpu")
    training = fit_classifier(
        model,
        loader(split.train_indices, 8),
        loader(split.validation_indices, 4),
        device="cpu",
        config=TrainingConfig(epochs=1, learning_rate=1e-3),
    )
    evaluation = evaluate_classifier(model, loader(split.test_indices, 4), device="cpu")
    observed = tensor_labels[list(split.test_indices)].numpy()
    probabilities = evaluation.probabilities.numpy()
    np.savez(paths["predictions"] / "classification.npz", labels=observed, probabilities=probabilities)
    fig, _ = plot_roc_curve(observed, probabilities[:, 1], class_names=["Positive"])
    fig.savefig(paths["figures"] / "classification_roc.png")
    result = {
        "accuracy": evaluation.metrics.accuracy,
        "auroc": evaluation.metrics.auroc,
        "auroc_status": evaluation.metrics.auroc_status,
        "best_epoch": training.best_epoch,
        "test_count": int(observed.size),
    }
    (paths["metrics"] / "classification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _segmentation(seed: int, paths: dict[str, Path]) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 1)
    masks = np.zeros((8, 8, 8, 8), dtype=np.int64)
    for index in range(8):
        offset = index % 2
        masks[index, 2 + offset : 6 + offset, 2:6, 2:6] = 1
    images = masks[:, None].astype(np.float32)
    images += rng.normal(0, 0.05, size=images.shape).astype(np.float32)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(images[:4]), torch.from_numpy(masks[:4])), batch_size=2)
    validation_loader = DataLoader(TensorDataset(torch.from_numpy(images[4:6]), torch.from_numpy(masks[4:6])), batch_size=2)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(images[6:]), torch.from_numpy(masks[6:])), batch_size=2)
    model = build_segresnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=8,
        blocks_down=(1, 1),
        blocks_up=(1,),
    )
    contract = SegmentationLabelContract(encoding="INTEGER_CLASS_MAP", out_channels=2)
    training = fit_segmenter(
        model,
        train_loader,
        validation_loader,
        label_contract=contract,
        device="cpu",
        config=SegmentationTrainingConfig(epochs=1, learning_rate=1e-3),
    )
    evaluation = evaluate_segmenter(
        model,
        test_loader,
        label_contract=contract,
        device="cpu",
        sliding_window=SlidingWindowConfig(roi_size=(8, 8, 8), overlap=0.25),
    )
    predictions = evaluation.predictions.numpy()
    np.save(paths["predictions"] / "segmentation.npy", predictions)
    fig, _ = plot_image_mask_prediction(
        images[6:7],
        masks[6:7, None],
        predictions[:1],
        slice_index=4,
    )
    fig.savefig(paths["figures"] / "segmentation_panel.png")
    result = {
        "best_epoch": training.best_epoch,
        "prediction_shape": list(predictions.shape),
        "dice": evaluation.dice.to_dict(),
    }
    (paths["metrics"] / "segmentation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _survival(seed: int, paths: dict[str, Path]) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 2)
    group_count = 24
    features = rng.normal(size=(group_count, 4)).astype(np.float32)
    log_risk = 0.7 * features[:, 0] - 0.4 * features[:, 1]
    event_time = rng.exponential(scale=9.0 * np.exp(-log_risk))
    censor_time = rng.exponential(scale=13.0, size=group_count)
    durations = np.minimum(event_time, censor_time).astype(np.float32)
    events = (event_time <= censor_time).astype(np.int64)
    events[:6] = 0
    events[6:12] = 1
    ids = [f"synthetic-survival-{index:02d}" for index in range(group_count)]
    split = split_survival_samples(
        ids,
        events.tolist(),
        group_ids=[f"synthetic-group-{index:02d}" for index in range(group_count)],
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=seed,
        stratify=True,
        require_groups=True,
    )
    train_idx = np.asarray(split.train_indices)
    validation_idx = np.asarray(split.validation_indices)
    test_idx = np.asarray(split.test_indices)
    scaled = fit_train_only_preprocessor(
        StandardScaler(), features[train_idx], features[validation_idx], features[test_idx]
    )
    assert scaled.test is not None
    model = build_coxph_model(in_features=4, hidden_dims=(8,), dropout=0.0, batch_norm=False, device="cpu")
    training = fit_coxph(
        model,
        scaled.train,
        durations[train_idx],
        events[train_idx],
        scaled.validation,
        durations[validation_idx],
        events[validation_idx],
        config=CoxPHTrainingConfig(epochs=4, learning_rate=5e-3),
    )
    evaluation = evaluate_coxph(
        model,
        scaled.test,
        durations[test_idx],
        events[test_idx],
        sample_ids=[ids[index] for index in test_idx],
    )
    np.savez(
        paths["predictions"] / "survival.npz",
        log_risk=evaluation.risk.log_risk,
        relative_risk=evaluation.risk.relative_risk,
        times=evaluation.survival.times,
        survival=evaluation.survival.survival_probabilities,
    )
    fig, _ = plot_survival_curves(
        evaluation.survival.times,
        evaluation.survival.survival_probabilities[:3].T,
    )
    fig.savefig(paths["figures"] / "survival_curves.png")
    result = {
        "best_epoch": training.best_epoch,
        "concordance": evaluation.concordance.value,
        "comparable_pairs": evaluation.concordance.comparable_pairs,
        "test_count": len(evaluation.risk.sample_ids),
    }
    (paths["metrics"] / "survival.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run(output_root: Path, seed: int = 20260904) -> dict[str, Any]:
    """Execute the fixed synthetic quick start and return a summary."""
    paths = _prepare_output(output_root)
    classification_environment = configure_reproducibility(seed=seed, device="cpu")
    survival_environment = configure_coxph_reproducibility(seed=seed, device="cpu")
    summary = {
        "status": "SYNTHETIC_TECHNICAL_DEMONSTRATION_ONLY",
        "classification": _classification(seed, paths),
        "segmentation": _segmentation(seed, paths),
        "survival": _survival(seed, paths),
        "environment": {
            "classification": classification_environment.to_dict(),
            "survival": survival_environment.to_dict(),
        },
        "scientific_claim": "NONE",
    }
    (paths["logs"] / "quick_start.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "output")
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    result = run(args.output, args.seed)
    print(json.dumps({"status": result["status"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
