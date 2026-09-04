"""Explicit paired 3-D segmentation transform boundaries for SegResNet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    NormalizeIntensityd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Resized,
)


ChannelDimension: TypeAlias = int | str
LabelEncoding: TypeAlias = Literal["INTEGER_CLASS_MAP", "MULTICHANNEL"]


@dataclass(frozen=True)
class SegmentationTransformConfig:
    """Preprocessing and train-only augmentation with explicit label encoding.

    ``spatial_size`` and ``train_crop_size`` default to ``None`` because
    resampling and patch size are study-specific scientific decisions. The
    caller must explicitly describe whether a raw label is an integer class map
    or a multi-channel target; the transform never infers that representation.
    """

    label_encoding: LabelEncoding
    image_key: str = "image"
    label_key: str = "label"
    image_channel_dim: ChannelDimension = 0
    label_channel_dim: ChannelDimension = "no_channel"
    spatial_size: tuple[int, int, int] | None = None
    train_crop_size: tuple[int, int, int] | None = None
    normalize_intensity: bool = False
    normalize_nonzero: bool = False
    normalize_channel_wise: bool = True
    rotation_probability: float = 0.25
    rotation_axes: tuple[int, int] = (0, 1)
    flip_probability: float = 0.25
    flip_axes: tuple[int, ...] = (0, 1, 2)
    intensity_scale_probability: float = 0.20
    intensity_scale_factor: float = 0.10
    intensity_shift_probability: float = 0.20
    intensity_shift_offset: float = 0.10

    def __post_init__(self) -> None:
        if self.label_encoding not in {"INTEGER_CLASS_MAP", "MULTICHANNEL"}:
            raise ValueError("label_encoding must be INTEGER_CLASS_MAP or MULTICHANNEL.")
        if not self.image_key or not self.label_key or self.image_key == self.label_key:
            raise ValueError("image_key and label_key must be non-empty and distinct.")
        for name, size in (
            ("spatial_size", self.spatial_size),
            ("train_crop_size", self.train_crop_size),
        ):
            if size is not None and (len(size) != 3 or any(value < 1 for value in size)):
                raise ValueError(f"{name} must be None or three positive integers.")
        if len(self.rotation_axes) != 2 or any(axis not in (0, 1, 2) for axis in self.rotation_axes):
            raise ValueError("rotation_axes must identify two 3-D spatial axes.")
        if len(set(self.rotation_axes)) != 2:
            raise ValueError("rotation_axes must be distinct.")
        if not self.flip_axes or any(axis not in (0, 1, 2) for axis in self.flip_axes):
            raise ValueError("flip_axes must contain valid 3-D spatial axes.")
        for name, probability in (
            ("rotation_probability", self.rotation_probability),
            ("flip_probability", self.flip_probability),
            ("intensity_scale_probability", self.intensity_scale_probability),
            ("intensity_shift_probability", self.intensity_shift_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.intensity_scale_factor < 0.0 or self.intensity_shift_offset < 0.0:
            raise ValueError("Intensity augmentation magnitudes must be non-negative.")

    @property
    def label_dtype(self) -> torch.dtype:
        return torch.int64 if self.label_encoding == "INTEGER_CLASS_MAP" else torch.float32


def _deterministic_prefix(config: SegmentationTransformConfig) -> list[object]:
    transforms: list[object] = [
        EnsureChannelFirstd(keys=config.image_key, channel_dim=config.image_channel_dim),
        EnsureChannelFirstd(keys=config.label_key, channel_dim=config.label_channel_dim),
    ]
    if config.spatial_size is not None:
        transforms.append(
            Resized(
                keys=(config.image_key, config.label_key),
                spatial_size=config.spatial_size,
                mode=("trilinear", "nearest"),
                align_corners=(False, None),
            )
        )
    if config.normalize_intensity:
        transforms.append(
            NormalizeIntensityd(
                keys=config.image_key,
                nonzero=config.normalize_nonzero,
                channel_wise=config.normalize_channel_wise,
            )
        )
    return transforms


def _final_type_transform(config: SegmentationTransformConfig) -> EnsureTyped:
    return EnsureTyped(
        keys=(config.image_key, config.label_key),
        dtype=(torch.float32, config.label_dtype),
        track_meta=False,
    )


def build_eval_transforms(config: SegmentationTransformConfig) -> Compose:
    """Build deterministic validation/test transforms with paired geometry."""

    return Compose([*_deterministic_prefix(config), _final_type_transform(config)])


def build_train_transforms(
    config: SegmentationTransformConfig,
    *,
    seed: int,
) -> Compose:
    """Build explicitly seeded train-only augmentation.

    All random spatial operations receive both image and label keys. Random
    intensity operations receive only the image key, so target values cannot be
    changed by intensity augmentation.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative.")
    paired_keys = (config.image_key, config.label_key)
    transforms = _deterministic_prefix(config)
    if config.train_crop_size is not None:
        transforms.append(
            RandSpatialCropd(
                keys=paired_keys,
                roi_size=config.train_crop_size,
                random_center=True,
                random_size=False,
            )
        )
    transforms.append(
        RandRotate90d(
            keys=paired_keys,
            prob=config.rotation_probability,
            max_k=3,
            spatial_axes=config.rotation_axes,
        )
    )
    transforms.extend(
        RandFlipd(keys=paired_keys, prob=config.flip_probability, spatial_axis=axis)
        for axis in config.flip_axes
    )
    transforms.extend(
        [
            RandScaleIntensityd(
                keys=config.image_key,
                factors=config.intensity_scale_factor,
                prob=config.intensity_scale_probability,
            ),
            RandShiftIntensityd(
                keys=config.image_key,
                offsets=config.intensity_shift_offset,
                prob=config.intensity_shift_probability,
            ),
            _final_type_transform(config),
        ]
    )
    pipeline = Compose(transforms)
    pipeline.set_random_state(seed=seed)
    return pipeline
