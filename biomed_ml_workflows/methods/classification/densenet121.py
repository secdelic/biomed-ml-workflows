"""Offline MONAI DenseNet121 construction and 2-D transforms."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import TypeAlias

import torch
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    RandFlipd,
    RandRotated,
    RandZoomd,
    Resized,
    ScaleIntensityd,
)


ChannelDimension: TypeAlias = int | str


@dataclass(frozen=True)
class DenseNet121Config:
    """Explicit, offline construction parameters for the classification model."""

    spatial_dims: int
    in_channels: int
    out_channels: int

    def __post_init__(self) -> None:
        if self.spatial_dims != 2:
            raise ValueError("DenseNet121 classification is 2-D; spatial_dims must be 2.")
        if self.in_channels < 1:
            raise ValueError("in_channels must be a positive integer.")
        if self.out_channels < 2:
            raise ValueError("out_channels must be at least 2 for classification.")


@dataclass(frozen=True)
class ImageTransformConfig:
    """Explicit preprocessing and train-only augmentation parameters.

    ``channel_dim='no_channel'`` is appropriate for a raw ``H x W`` array;
    use ``channel_dim=0`` for data already arranged as ``C x H x W``.
    Per-image min/max scaling has no fitted cross-split state, but its
    scientific suitability remains modality- and study-specific.
    """

    spatial_size: tuple[int, int] = (32, 32)
    image_key: str = "image"
    channel_dim: ChannelDimension = "no_channel"
    scale_intensity: bool = True
    channel_wise_scaling: bool = False
    rotation_range_radians: float = pi / 12
    rotation_probability: float = 0.25
    flip_probability: float = 0.25
    flip_spatial_axis: int = 0
    zoom_probability: float = 0.20
    min_zoom: float = 0.90
    max_zoom: float = 1.10

    def __post_init__(self) -> None:
        if len(self.spatial_size) != 2 or any(size < 32 for size in self.spatial_size):
            raise ValueError("spatial_size must contain two values, each at least 32.")
        if not self.image_key:
            raise ValueError("image_key must be non-empty.")
        for name, probability in (
            ("rotation_probability", self.rotation_probability),
            ("flip_probability", self.flip_probability),
            ("zoom_probability", self.zoom_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.rotation_range_radians < 0.0:
            raise ValueError("rotation_range_radians must be non-negative.")
        if not 0.0 < self.min_zoom <= self.max_zoom:
            raise ValueError("zoom bounds must satisfy 0 < min_zoom <= max_zoom.")


def build_densenet121(
    *,
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    device: str | torch.device | None = None,
) -> DenseNet121:
    """Construct an untrained MONAI DenseNet121 without network access.

    The model is created with ``pretrained=False`` so construction never
    downloads weights. Device placement is optional and never hard-coded.
    """

    config = DenseNet121Config(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    model = DenseNet121(
        spatial_dims=config.spatial_dims,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        pretrained=False,
    )
    return model if device is None else model.to(torch.device(device))


def _deterministic_transforms(config: ImageTransformConfig) -> list[object]:
    transforms: list[object] = [
        EnsureChannelFirstd(
            keys=config.image_key,
            channel_dim=config.channel_dim,
        )
    ]
    if config.scale_intensity:
        transforms.append(
            ScaleIntensityd(
                keys=config.image_key,
                channel_wise=config.channel_wise_scaling,
            )
        )
    transforms.extend(
        [
            Resized(
                keys=config.image_key,
                spatial_size=config.spatial_size,
                mode="bilinear",
                align_corners=False,
            ),
            EnsureTyped(
                keys=config.image_key,
                dtype=torch.float32,
                track_meta=False,
            ),
        ]
    )
    return transforms


def build_eval_transforms(config: ImageTransformConfig | None = None) -> Compose:
    """Build deterministic validation/test transforms with no augmentation."""

    resolved = config or ImageTransformConfig()
    return Compose(_deterministic_transforms(resolved))


def build_train_transforms(
    config: ImageTransformConfig | None = None,
    *,
    seed: int,
) -> Compose:
    """Build preprocessing plus random augmentation for training only.

    The returned ``Compose`` receives an explicit random state because MONAI
    ``Randomizable`` objects created after a global determinism call otherwise
    need their own seed.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative.")
    resolved = config or ImageTransformConfig()
    transforms = _deterministic_transforms(resolved)
    final_type = transforms.pop()
    transforms.extend(
        [
            RandRotated(
                keys=resolved.image_key,
                range_x=resolved.rotation_range_radians,
                prob=resolved.rotation_probability,
                keep_size=True,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            ),
            RandFlipd(
                keys=resolved.image_key,
                prob=resolved.flip_probability,
                spatial_axis=resolved.flip_spatial_axis,
            ),
            RandZoomd(
                keys=resolved.image_key,
                prob=resolved.zoom_probability,
                min_zoom=resolved.min_zoom,
                max_zoom=resolved.max_zoom,
                mode="bilinear",
                # Zoom keep-size padding uses pad modes, where MONAI maps
                # NumPy "edge" to PyTorch "replicate" boundary padding.
                padding_mode="edge",
                align_corners=False,
                keep_size=True,
            ),
            final_type,
        ]
    )
    pipeline = Compose(transforms)
    pipeline.set_random_state(seed=seed)
    return pipeline
