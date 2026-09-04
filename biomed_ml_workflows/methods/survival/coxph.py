"""Independently written pycox CoxPH model construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from pycox.models import CoxPH
from torch import nn


ActivationName = Literal["relu", "tanh", "elu"]


@dataclass(frozen=True)
class CoxPHNetworkConfig:
    """Explicit configuration for a baseline-covariate CoxPH log-risk network."""

    in_features: int
    hidden_dims: tuple[int, ...] = (16,)
    activation: ActivationName = "relu"
    dropout: float = 0.0
    batch_norm: bool = False
    output_bias: bool = False

    def __post_init__(self) -> None:
        if self.in_features < 1:
            raise ValueError("in_features must be a positive integer.")
        if any(width < 1 for width in self.hidden_dims):
            raise ValueError("Every hidden dimension must be positive.")
        if self.activation not in {"relu", "tanh", "elu"}:
            raise ValueError("activation must be one of: relu, tanh, elu.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")


def _activation(name: ActivationName) -> nn.Module:
    return {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}[name]()


class CoxPHNetwork(nn.Module):
    """Feed-forward network producing one unconstrained log-risk score."""

    def __init__(self, config: CoxPHNetworkConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = config.in_features
        for width in config.hidden_dims:
            layers.append(nn.Linear(previous, width))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(_activation(config.activation))
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            previous = width
        layers.append(nn.Linear(previous, 1, bias=config.output_bias))
        self.network = nn.Sequential(*layers)
        self.config = config

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("CoxPH features must have shape [samples, features].")
        if features.shape[1] != self.config.in_features:
            raise ValueError(
                f"Expected {self.config.in_features} features, got {features.shape[1]}."
            )
        return self.network(features)


def build_coxph_network(
    *,
    in_features: int,
    hidden_dims: tuple[int, ...] = (16,),
    activation: ActivationName = "relu",
    dropout: float = 0.0,
    batch_norm: bool = False,
    output_bias: bool = False,
    device: str | torch.device | None = None,
) -> CoxPHNetwork:
    """Construct an untrained log-risk network without data or network access."""

    model = CoxPHNetwork(
        CoxPHNetworkConfig(
            in_features=in_features,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
            output_bias=output_bias,
        )
    )
    return model if device is None else model.to(torch.device(device))


def build_coxph_model(
    *,
    in_features: int,
    hidden_dims: tuple[int, ...] = (16,),
    activation: ActivationName = "relu",
    dropout: float = 0.0,
    batch_norm: bool = False,
    output_bias: bool = False,
    device: str | torch.device | None = None,
) -> CoxPH:
    """Wrap the log-risk network in pycox 0.3.0's CoxPH model.

    No optimizer, data, pretrained weights, or device is selected implicitly.
    """

    network = build_coxph_network(
        in_features=in_features,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout=dropout,
        batch_norm=batch_norm,
        output_bias=output_bias,
        device=device,
    )
    # pycox auto-selects CUDA when ``device=None``. Passing the network's actual
    # placement keeps the public default CPU-safe and avoids an implicit device
    # policy while still honoring an explicitly requested device.
    resolved_device = next(network.parameters()).device
    return CoxPH(network, device=resolved_device)
