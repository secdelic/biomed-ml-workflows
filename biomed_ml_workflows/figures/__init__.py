"""Public scientific figure functions for BioMed ML Workflows."""

from .classification import (
    plot_class_distribution_bar,
    plot_class_distribution_pie,
    plot_class_sample_montage,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_training_history,
)
from .interpretation import (
    plot_feature_coefficients,
    plot_feature_importance,
    plot_occlusion_sensitivity,
)
from .segmentation import (
    plot_classwise_dice,
    plot_ground_truth_channels,
    plot_image_and_mask,
    plot_image_mask_prediction,
    plot_input_channels,
    plot_loss_and_mean_dice,
    plot_masking_comparison,
    plot_prediction_channels,
)
from .statistical import (
    plot_3d_volume_scatter,
    plot_activation_curves,
    plot_bar,
    plot_boxplot,
    plot_cluster_elbow,
    plot_cluster_scatter,
    plot_cluster_silhouette,
    plot_coefficient_path,
    plot_correlation_heatmap,
    plot_density,
    plot_embedding,
    plot_forecast,
    plot_forecast_diagnostics,
    plot_histogram,
    plot_image,
    plot_image_montage,
    plot_image_projection,
    plot_joint_distribution,
    plot_line,
    plot_pairwise_relationships,
    plot_regularization_errorbar,
    plot_relationship_scatter,
    plot_scatter,
    plot_signal_spectrum,
    plot_swarm_facet,
    plot_venn_diagram,
    plot_violin,
)
from .survival import (
    plot_brier_score,
    plot_kaplan_meier,
    plot_learning_rate_finder,
    plot_learning_rate_schedule,
    plot_nbll,
    plot_survival_curves,
    plot_survival_training_loss,
)


__all__ = [name for name in globals() if name.startswith("plot_")]
