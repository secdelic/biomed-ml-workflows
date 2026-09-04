"""Independently written offline MONAI SegResNet construction.

This generic model module contains no dataset-specific modality, label, crop,
path, checkpoint, download, postprocessing, or evaluation assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from monai.networks.nets import SegResNet


UpsampleMode: TypeAlias = Literal["deconv", "nontrainable", "pixelshuffle"]


@dataclass(frozen=True)
class SegResNetConfig:
    """Explicit architectural parameters for a generic 3-D SegResNet."""

    spatial_dims: int
    in_channels: int
    out_channels: int
    init_filters: int = 8
    dropout_prob: float | None = None
    blocks_down: tuple[int, ...] = (1, 2, 2, 4)
    blocks_up: tuple[int, ...] = (1, 1, 1)
    upsample_mode: UpsampleMode = "deconv"

    def __post_init__(self) -> None:
        if self.spatial_dims != 3:
            raise ValueError("SegResNet segmentation is 3-D; spatial_dims must be 3.")
        if self.in_channels < 1:
            raise ValueError("in_channels must be a positive integer.")
        if self.out_channels < 1:
            raise ValueError("out_channels must be a positive integer.")
        if self.init_filters < 8 or self.init_filters % 8 != 0:
            raise ValueError(
                "init_filters must be a positive multiple of 8 for the fixed eight-group normalization."
            )
        if self.dropout_prob is not None and not 0.0 <= self.dropout_prob < 1.0:
            raise ValueError("dropout_prob must be None or satisfy 0 <= value < 1.")
        if len(self.blocks_down) < 2 or any(value < 1 for value in self.blocks_down):
            raise ValueError("blocks_down must contain at least two positive integers.")
        if len(self.blocks_up) != len(self.blocks_down) - 1:
            raise ValueError("blocks_up length must equal len(blocks_down) - 1.")
        if any(value < 1 for value in self.blocks_up):
            raise ValueError("blocks_up values must be positive integers.")
        if self.upsample_mode not in {"deconv", "nontrainable", "pixelshuffle"}:
            raise ValueError("upsample_mode must be deconv, nontrainable, or pixelshuffle.")

    @property
    def spatial_divisibility(self) -> int:
        """Required divisibility arising from encoder downsampling levels."""

        return 2 ** (len(self.blocks_down) - 1)


def build_segresnet(
    *,
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    init_filters: int = 8,
    dropout_prob: float | None = None,
    blocks_down: tuple[int, ...] = (1, 2, 2, 4),
    blocks_up: tuple[int, ...] = (1, 1, 1),
    upsample_mode: UpsampleMode = "deconv",
) -> SegResNet:
    """Construct an untrained MONAI SegResNet without device or data coupling.

    Device placement is intentionally the caller's responsibility. Input
    tensors use ``[B, C, D, H, W]`` and each spatial extent should be divisible
    by ``2 ** (len(blocks_down) - 1)``.
    """

    config = SegResNetConfig(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=init_filters,
        dropout_prob=dropout_prob,
        blocks_down=blocks_down,
        blocks_up=blocks_up,
        upsample_mode=upsample_mode,
    )
    return SegResNet(
        spatial_dims=config.spatial_dims,
        init_filters=config.init_filters,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        dropout_prob=config.dropout_prob,
        blocks_down=config.blocks_down,
        blocks_up=config.blocks_up,
        upsample_mode=config.upsample_mode,
    )
