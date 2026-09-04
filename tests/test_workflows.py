from __future__ import annotations

import inspect

import numpy as np
import torch

from biomed_ml_workflows.methods.classification import build_densenet121
from biomed_ml_workflows.methods.segmentation import build_segresnet
from biomed_ml_workflows.methods.survival import build_coxph_model, build_coxph_network
from biomed_ml_workflows.workflows.classification import fit_classifier, split_samples
from biomed_ml_workflows.workflows.segmentation import (
    SegmentationLabelContract,
    fit_segmenter,
    prepare_segmentation_target,
    segmentation_dice,
    validate_patch_partitioning,
)
from biomed_ml_workflows.workflows.survival import (
    fit_coxph,
    fit_train_only_preprocessor,
    split_survival_samples,
    validate_survival_dataset,
)


class FitRecorder:
    def __init__(self) -> None:
        self.fit_values: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "FitRecorder":
        self.fit_values = values.copy()
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return values + 1


def test_model_constructors_and_cpu_forward_shapes() -> None:
    classifier = build_densenet121(spatial_dims=2, in_channels=1, out_channels=3)
    classifier.eval()
    with torch.inference_mode():
        classification = classifier(torch.randn(1, 1, 32, 32))
    assert classification.shape == (1, 3)
    assert torch.isfinite(classification).all()

    segmenter = build_segresnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=8,
        blocks_down=(1, 1),
        blocks_up=(1,),
    )
    segmenter.eval()
    with torch.inference_mode():
        segmentation = segmenter(torch.randn(1, 1, 8, 8, 8))
    assert segmentation.shape == (1, 2, 8, 8, 8)
    assert torch.isfinite(segmentation).all()

    network = build_coxph_network(in_features=4, hidden_dims=(8,))
    survival = network(torch.randn(5, 4))
    assert survival.shape == (5, 1)
    assert torch.isfinite(survival).all()
    assert next(build_coxph_model(in_features=4).net.parameters()).device.type == "cpu"


def test_group_splits_are_deterministic_complete_and_disjoint() -> None:
    ids = tuple(f"sample-{index:02d}" for index in range(24))
    groups = tuple(f"subject-{index // 2:02d}" for index in range(24))
    labels = tuple((index // 2) % 2 for index in range(24))
    kwargs = dict(
        group_ids=groups,
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=19,
        require_groups=True,
    )
    first = split_samples(ids, labels, **kwargs)
    assert first == split_samples(ids, labels, **kwargs)
    sample_sets = [set(first.train_ids), set(first.validation_ids), set(first.test_ids)]
    group_sets = [set(first.train_group_ids), set(first.validation_group_ids), set(first.test_group_ids)]
    assert set(ids) == set().union(*sample_sets)
    for index in range(3):
        for other in range(index + 1, 3):
            assert sample_sets[index].isdisjoint(sample_sets[other])
            assert group_sets[index].isdisjoint(group_sets[other])


def test_patch_generation_is_checked_after_volume_partitioning() -> None:
    ids = tuple(f"volume-{index:02d}" for index in range(20))
    split = split_samples(
        ids,
        tuple(index % 2 for index in range(20)),
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=13,
    )
    sources: list[str] = []
    partitions: list[str] = []
    for partition, identifiers in (
        ("train", split.train_ids),
        ("validation", split.validation_ids),
        ("test", split.test_ids),
    ):
        for identifier in identifiers:
            sources.extend((identifier, identifier))
            partitions.extend((partition, partition))
    result = validate_patch_partitioning(split, sources, partitions)
    assert result.order == "PARTITION_BEFORE_PATCH_GENERATION"
    assert result.patch_count == 40
    assert result.cross_partition_sources == ()


def test_segmentation_label_contracts_remain_distinct() -> None:
    logits = torch.randn(2, 3, 8, 8, 8)
    integer_contract = SegmentationLabelContract(encoding="INTEGER_CLASS_MAP", out_channels=3)
    integer_target = torch.randint(0, 3, (2, 1, 8, 8, 8))
    prepared = prepare_segmentation_target(integer_target, logits, integer_contract)
    assert prepared.dtype == torch.long
    result = segmentation_dice(integer_target, integer_target, integer_contract)
    assert result.mean_dice == 1

    multichannel_contract = SegmentationLabelContract(
        encoding="MULTICHANNEL", out_channels=3, include_background=True
    )
    multichannel = torch.randint(0, 2, logits.shape).float()
    assert prepare_segmentation_target(multichannel, logits, multichannel_contract).shape == logits.shape
    try:
        prepare_segmentation_target(multichannel[:, :1], logits, multichannel_contract)
    except ValueError:
        pass
    else:
        raise AssertionError("incompatible multichannel labels were accepted")


def test_survival_validation_group_split_and_train_only_preprocessing() -> None:
    features = np.ones((24, 2), dtype=np.float32)
    durations = np.arange(1, 25, dtype=np.float32)
    events = np.asarray([0, 0, 1, 1] * 6)
    validated = validate_survival_dataset(features, durations, events, partition="train")
    assert validated.features.shape == (24, 2)

    ids = [f"sample-{index}" for index in range(24)]
    groups = [f"subject-{index // 2}" for index in range(24)]
    grouped_events = [([0, 1] * 6)[index // 2] for index in range(24)]
    split = split_survival_samples(
        ids,
        grouped_events,
        group_ids=groups,
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=13,
        stratify=True,
        require_groups=True,
    )
    sets = [set(split.train_group_ids), set(split.validation_group_ids), set(split.test_group_ids)]
    assert sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2]) and sets[1].isdisjoint(sets[2])

    train = np.asarray([[1, 2], [3, 4]], dtype=np.float32)
    validation = np.asarray([[100, 200]], dtype=np.float32)
    test = np.asarray([[300, 400]], dtype=np.float32)
    recorder = FitRecorder()
    transformed = fit_train_only_preprocessor(recorder, train, validation, test)
    np.testing.assert_array_equal(recorder.fit_values, train)
    np.testing.assert_array_equal(transformed.validation, validation + 1)
    np.testing.assert_array_equal(transformed.test, test + 1)
    assert transformed.fit_partition == "train"
    assert not transformed.validation_fit_used and not transformed.test_fit_used


def test_training_interfaces_do_not_accept_test_data() -> None:
    for function in (fit_classifier, fit_segmenter, fit_coxph):
        parameters = inspect.signature(function).parameters
        assert "test" not in parameters
        assert "test_loader" not in parameters
        assert "test_features" not in parameters
