"""Conditioner networks for the per-edge term types.

Architectures replicate the original Keras/TF implementation's defaults
(``tram_models.py`` in https://github.com/tensorchiefs/tram-dag)
so that fitted models are directly comparable:

- ``LinearShift``        — Linear(n, 1, bias=False)            (term "ls")
- ``ComplexShift``       — 64-128-64 ReLU MLP -> 1, no bias    (term "cs",
                            original ``ComplexShiftDefaultTabular``)
- ``ComplexIntercept``   — 8-8 ReLU MLP -> n_params, no bias   (term "ci",
                            original ``ComplexInterceptDefaultTabular``)
- ``SimpleIntercept``    — free parameter vector (no parent dependence)

Parent features follow the original implementation's encoding: continuous parents enter raw
(one column), ordinal parents are one-hot encoded (``levels`` columns).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SimpleIntercept(nn.Module):
    """Free (data-independent) transform parameters, broadcast over the batch."""

    def __init__(self, n_params: int):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(n_params))

    def forward(self, n: int) -> Tensor:
        return self.theta.unsqueeze(0).expand(n, -1)


class ComplexIntercept(nn.Module):
    """Transform parameters as a function of the (joint) ci-parent features.

    ``residual=True`` adds a trainable linear+constant base ``base_bias +
    base_lin(x)`` to the MLP output — used by ``seed_from_classical`` to anchor
    the module at a classical (constant intercept + linear-shift) solution while
    the MLP learns the deviation. Default ``False`` reproduces the original module
    exactly (no base params)."""

    def __init__(self, n_features: int, n_params: int, residual: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 8), nn.ReLU(),
            nn.Linear(8, 8), nn.ReLU(),
            nn.Linear(8, n_params, bias=False),
        )
        if residual:
            self.base_bias = nn.Parameter(torch.zeros(n_params))
            self.base_lin = nn.Linear(n_features, n_params, bias=False)
        else:
            self.base_bias = None
            self.base_lin = None

    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)
        if self.base_bias is not None:
            out = out + self.base_bias + self.base_lin(x)
        return out


class LinearShift(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.fc = nn.Linear(n_features, 1, bias=False)

    @property
    def weight(self) -> Tensor:
        return self.fc.weight.squeeze(0)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x).squeeze(-1)


class ComplexShift(nn.Module):
    """``residual=True`` adds a trainable linear base ``base(x)`` to the MLP output
    (seeded to a classical ``ls`` weight by ``seed_from_classical``; the MLP starts
    near-zero and learns the nonlinear deviation). Default ``False`` reproduces the
    original module exactly."""

    def __init__(self, n_features: int, residual: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1, bias=False),
        )
        self.base = nn.Linear(n_features, 1, bias=False) if residual else None

    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x).squeeze(-1)
        if self.base is not None:
            out = out + self.base(x).squeeze(-1)
        return out
