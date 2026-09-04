"""Generate the complete synthetic figure capability pack.

Every output is a SYNTHETIC DEMONSTRATION and is NOT SCIENTIFIC PERFORMANCE
EVIDENCE. The script is deterministic, offline, and performs no model fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from biomed_ml_workflows.figures import classification as classification_figures
from biomed_ml_workflows.figures import interpretation as interpretation_figures
from biomed_ml_workflows.figures import segmentation as segmentation_figures
from biomed_ml_workflows.figures import statistical as statistical_figures
from biomed_ml_workflows.figures import survival as survival_figures


OUTPUT_ROOT = REPO_ROOT / "output" / "figure_gallery"


@dataclass(frozen=True)
class DemoCase:
    family: str
    function: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    @property
    def name(self) -> str:
        return self.function.__name__

    @property
    def relative_output(self) -> Path:
        return Path(self.family) / f"{self.family}_{self.name}.png"


def build_demo_cases() -> list[DemoCase]:
    """Return one deterministic, synthetic execution case per public function."""
    rng = np.random.default_rng(20260904)
    x = np.linspace(0, 10, 40)
    categories = ["A", "B", "C"]
    data = rng.normal(size=(60, 3))
    groups = np.where(np.arange(60) % 2 == 0, "Group A", "Group B")
    image_axis = np.linspace(-1, 1, 32)
    xx, yy = np.meshgrid(image_axis, image_axis)
    image = np.exp(-3 * (xx**2 + yy**2))
    mask = ((xx**2 + yy**2) < 0.35).astype(float)
    prediction = ((xx - 0.08) ** 2 + (yy + 0.05) ** 2 < 0.35).astype(float)
    volume_axis = np.linspace(-1, 1, 18)
    zz, vy, vx = np.meshgrid(volume_axis, volume_axis, volume_axis, indexing="ij")
    volume = np.exp(-4 * (vx**2 + vy**2 + zz**2))
    channels = np.stack([volume, volume * (vx > 0), volume * (vy > 0), volume * (zz > 0)])
    label_channels = np.stack([(volume > 0.35), (volume > 0.55), (volume > 0.75)]).astype(float)
    prediction_channels = np.roll(label_channels, shift=1, axis=2)
    epochs = np.arange(1, 9)
    train_loss = np.exp(-epochs / 4) + 0.08
    validation_loss = np.exp(-epochs / 4.5) + 0.12
    mean_dice = 0.35 + 0.55 * (1 - np.exp(-epochs / 3))
    time = np.linspace(0, 24, 30)
    survival_matrix = np.column_stack(
        [np.exp(-time / 24), np.exp(-time / 16), np.exp(-time / 10)]
    )
    labels = np.asarray([0, 1] * 20)
    scores = np.clip(np.linspace(0.05, 0.95, 40) + rng.normal(0, 0.12, 40), 0, 1)
    sample_images = [np.roll(image, shift=index * 2, axis=1) for index in range(6)]
    embedding = np.column_stack([rng.normal(size=50), rng.normal(size=50)])

    cases = [
        DemoCase("statistical", statistical_figures.plot_line, (x, np.vstack([np.sin(x), np.cos(x)])), {"labels": ["Sine", "Cosine"], "xlabel": "x"}),
        DemoCase("statistical", statistical_figures.plot_scatter, (data[:, 0], data[:, 1]), {"xlabel": "Variable 1", "ylabel": "Variable 2"}),
        DemoCase("statistical", statistical_figures.plot_histogram, (data,), {"labels": ["A", "B", "C"]}),
        DemoCase("statistical", statistical_figures.plot_density, (np.vstack([data[:, 0], data[:, 1]]),), {"labels": ["A", "B"]}),
        DemoCase("statistical", statistical_figures.plot_bar, (categories, np.asarray([[12, 18, 9], [8, 14, 11]])), {"series_labels": ["Cohort 1", "Cohort 2"]}),
        DemoCase("statistical", statistical_figures.plot_boxplot, ([data[:, 0], data[:, 1], data[:, 2]],), {"labels": categories}),
        DemoCase("statistical", statistical_figures.plot_violin, ([data[:, 0], data[:, 1], data[:, 2]],), {"labels": categories}),
        DemoCase("statistical", statistical_figures.plot_correlation_heatmap, (data,), {"labels": ["A", "B", "C"]}),
        DemoCase("statistical", statistical_figures.plot_joint_distribution, (data[:, 0], data[:, 1]), {}),
        DemoCase("statistical", statistical_figures.plot_pairwise_relationships, (data,), {"labels": ["A", "B", "C"]}),
        DemoCase("statistical", statistical_figures.plot_relationship_scatter, (data[:, 0], data[:, 1]), {"groups": groups}),
        DemoCase("statistical", statistical_figures.plot_swarm_facet, (data[:, 0], np.resize(categories, 60)), {"facets": groups}),
        DemoCase("statistical", statistical_figures.plot_regularization_errorbar, (np.logspace(-3, 1, 8), np.linspace(0.65, 0.82, 8), np.full(8, 0.025)), {"selected_parameter": 0.1}),
        DemoCase("statistical", statistical_figures.plot_coefficient_path, (np.logspace(-3, 1, 8), np.vstack([np.linspace(1, 0, 8), np.linspace(-0.7, 0, 8), np.sin(np.linspace(0, np.pi, 8))])), {"feature_names": ["A", "B", "C"], "selected_parameter": 0.1}),
        DemoCase("statistical", statistical_figures.plot_activation_curves, (np.linspace(-5, 5, 100), {"ReLU": np.maximum(np.linspace(-5, 5, 100), 0), "Sigmoid": 1 / (1 + np.exp(-np.linspace(-5, 5, 100))), "Tanh": np.tanh(np.linspace(-5, 5, 100)), "Leaky ReLU": np.where(np.linspace(-5, 5, 100) >= 0, np.linspace(-5, 5, 100), 0.01 * np.linspace(-5, 5, 100))}), {}),
        DemoCase("statistical", statistical_figures.plot_cluster_elbow, (np.arange(1, 9), np.asarray([120, 82, 58, 44, 37, 33, 30, 28])), {}),
        DemoCase("statistical", statistical_figures.plot_cluster_silhouette, (np.linspace(0.1, 0.8, 40), np.repeat([0, 1], 20)), {"average_score": 0.45}),
        DemoCase("statistical", statistical_figures.plot_cluster_scatter, (embedding[:, 0], embedding[:, 1], np.repeat([0, 1], 25)), {"centers": [[-0.5, 0.0], [0.5, 0.0]]}),
        DemoCase("statistical", statistical_figures.plot_embedding, (embedding,), {"labels": np.repeat(["A", "B"], 25), "method_name": "Precomputed embedding"}),
        DemoCase("statistical", statistical_figures.plot_forecast, (x, np.sin(x), np.sin(x) * 0.95), {"lower": np.sin(x) * 0.95 - 0.2, "upper": np.sin(x) * 0.95 + 0.2}),
        DemoCase("statistical", statistical_figures.plot_forecast_diagnostics, (rng.normal(0, 0.2, 50),), {"fitted": np.linspace(1, 2, 50)}),
        DemoCase("statistical", statistical_figures.plot_signal_spectrum, (x, np.sin(2 * np.pi * x), np.linspace(0, 5, 30), np.exp(-((np.linspace(0, 5, 30) - 1) ** 2))), {}),
        DemoCase("statistical", statistical_figures.plot_image, (image,), {"title": "Synthetic image"}),
        DemoCase("statistical", statistical_figures.plot_image_montage, (sample_images,), {"labels": [f"Sample {i + 1}" for i in range(6)]}),
        DemoCase("statistical", statistical_figures.plot_image_projection, (volume,), {"threshold": 0.2}),
        DemoCase("statistical", statistical_figures.plot_3d_volume_scatter, (volume,), {"threshold": 0.35}),
        DemoCase("statistical", statistical_figures.plot_venn_diagram, ([14, 10, 6, 8, 4, 3, 2],), {"labels": ["A", "B", "C"]}),
        DemoCase("classification", classification_figures.plot_class_distribution_bar, (["Negative", "Positive"], [24, 16]), {}),
        DemoCase("classification", classification_figures.plot_class_distribution_pie, (["Negative", "Positive"], [24, 16]), {}),
        DemoCase("classification", classification_figures.plot_class_sample_montage, (sample_images, ["A", "B", "A", "B", "A", "B"]), {}),
        DemoCase("classification", classification_figures.plot_confusion_matrix, (labels, (scores >= 0.5).astype(int)), {"class_names": ["Negative", "Positive"]}),
        DemoCase("classification", classification_figures.plot_roc_curve, (labels, scores), {"class_names": ["Positive"]}),
        DemoCase("classification", classification_figures.plot_precision_recall_curve, (labels, scores), {"class_names": ["Positive"]}),
        DemoCase("classification", classification_figures.plot_training_history, (train_loss,), {"validation_loss": validation_loss, "validation_metric": mean_dice, "metric_name": "Validation AUROC", "validation_epochs": epochs}),
        DemoCase("segmentation", segmentation_figures.plot_image_and_mask, (channels[np.newaxis], label_channels[np.newaxis]), {"slice_index": 9}),
        DemoCase("segmentation", segmentation_figures.plot_input_channels, (channels[np.newaxis],), {"slice_index": 9}),
        DemoCase("segmentation", segmentation_figures.plot_ground_truth_channels, (label_channels[np.newaxis],), {"slice_index": 9}),
        DemoCase("segmentation", segmentation_figures.plot_prediction_channels, (prediction_channels[np.newaxis],), {"slice_index": 9}),
        DemoCase("segmentation", segmentation_figures.plot_image_mask_prediction, (channels[np.newaxis], label_channels[np.newaxis], prediction_channels[np.newaxis]), {"slice_index": 9}),
        DemoCase("segmentation", segmentation_figures.plot_masking_comparison, (image, mask, prediction), {}),
        DemoCase("segmentation", segmentation_figures.plot_loss_and_mean_dice, (epochs, train_loss, mean_dice), {"validation_loss": validation_loss}),
        DemoCase("segmentation", segmentation_figures.plot_classwise_dice, (epochs, np.vstack([mean_dice, mean_dice - 0.08, mean_dice - 0.15])), {"class_names": ["Region A", "Region B", "Region C"]}),
        DemoCase("survival", survival_figures.plot_kaplan_meier, ({"Group A": (time, survival_matrix[:, 0]), "Group B": (time, survival_matrix[:, 1])},), {}),
        DemoCase("survival", survival_figures.plot_survival_curves, (time, survival_matrix), {"labels": ["Patient A", "Patient B", "Patient C"]}),
        DemoCase("survival", survival_figures.plot_survival_training_loss, (epochs, train_loss), {"validation_loss": validation_loss}),
        DemoCase("survival", survival_figures.plot_learning_rate_finder, (np.logspace(-5, -1, 30), np.linspace(1.4, 0.5, 30) + np.linspace(0, 0.3, 30) ** 2), {"selected_rate": 1e-3}),
        DemoCase("survival", survival_figures.plot_learning_rate_schedule, (np.arange(1, 31), np.geomspace(1e-3, 1e-5, 30)), {}),
        DemoCase("survival", survival_figures.plot_brier_score, (time, np.column_stack([0.12 + 0.08 * time / time.max(), 0.15 + 0.06 * time / time.max()])), {"labels": ["Model A", "Model B"]}),
        DemoCase("survival", survival_figures.plot_nbll, (time, np.column_stack([0.45 + 0.1 * time / time.max(), 0.5 + 0.08 * time / time.max()])), {"labels": ["Model A", "Model B"]}),
        DemoCase("interpretation", interpretation_figures.plot_feature_coefficients, (["Age", "Marker A", "Marker B", "Volume"], [-0.3, 0.7, 0.25, -0.1]), {}),
        DemoCase("interpretation", interpretation_figures.plot_feature_importance, (["Age", "Marker A", "Marker B", "Volume"], [0.15, 0.42, 0.28, 0.15]), {}),
        DemoCase("interpretation", interpretation_figures.plot_occlusion_sensitivity, (image, np.exp(-10 * ((xx - 0.2) ** 2 + (yy + 0.1) ** 2))), {}),
    ]
    names = [case.name for case in cases]
    if len(names) != len(set(names)):
        raise RuntimeError("Demo registry contains duplicate public function names.")
    return cases


def generate_pack() -> list[Path]:
    """Execute every case, save one PNG, and verify each saved image opens."""
    outputs: list[Path] = []
    for case in build_demo_cases():
        destination = OUTPUT_ROOT / case.relative_output
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = case.function(*case.args, **case.kwargs)
        if not isinstance(result, tuple) or not result or not hasattr(result[0], "savefig"):
            raise RuntimeError(f"{case.name} did not return a Figure first.")
        figure = result[0]
        figure.savefig(destination)
        opened = plt.imread(destination)
        if opened.size == 0 or opened.ndim not in (2, 3):
            raise RuntimeError(f"Generated figure is unreadable: {destination}")
        plt.close(figure)
        outputs.append(destination)
    return outputs


if __name__ == "__main__":
    generated = generate_pack()
    print(f"GENERATED_FIGURES={len(generated)}")
    for path in generated:
        print(path.relative_to(REPO_ROOT).as_posix())
