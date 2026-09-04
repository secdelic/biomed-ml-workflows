"""SegResNet segmentation workflow."""

from ..classification import (
    ReproducibilityRecord,
    SplitResult,
    configure_reproducibility,
    make_dataloader,
    split_samples,
)
from .evaluate import (
    DiceChannelResult,
    SegmentationDiceResult,
    SegmentationEvaluationResult,
    SlidingWindowConfig,
    evaluate_segmenter,
    infer_segmentation_logits,
    predict_segmenter,
    segmentation_dice,
)
from .train import (
    PatchPartitionAudit,
    SegmentationEpochMetrics,
    SegmentationEpochRecord,
    SegmentationLabelContract,
    SegmentationTrainingConfig,
    SegmentationTrainingResult,
    build_default_segmentation_loss,
    fit_segmenter,
    prepare_segmentation_target,
    validate_patch_partitioning,
)
from .transforms import SegmentationTransformConfig, build_eval_transforms, build_train_transforms

__all__ = [
    "DiceChannelResult",
    "PatchPartitionAudit",
    "ReproducibilityRecord",
    "SegmentationDiceResult",
    "SegmentationEpochMetrics",
    "SegmentationEpochRecord",
    "SegmentationEvaluationResult",
    "SegmentationLabelContract",
    "SegmentationTrainingConfig",
    "SegmentationTrainingResult",
    "SegmentationTransformConfig",
    "SlidingWindowConfig",
    "SplitResult",
    "build_default_segmentation_loss",
    "build_eval_transforms",
    "build_train_transforms",
    "configure_reproducibility",
    "evaluate_segmenter",
    "fit_segmenter",
    "infer_segmentation_logits",
    "make_dataloader",
    "predict_segmenter",
    "prepare_segmentation_target",
    "segmentation_dice",
    "split_samples",
    "validate_patch_partitioning",
]

