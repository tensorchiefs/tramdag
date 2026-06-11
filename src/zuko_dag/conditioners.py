"""Conditioner networks for the per-edge term types.

Architectures replicate the tramdag defaults (``tramdag/models/tram_models.py``)
so that fitted models are directly comparable:

- ``LinearShift``        — Linear(n, 1, bias=False)            (term "ls")
- ``ComplexShift``       — 64-128-64 ReLU MLP -> 1, no bias    (term "cs",
                            tramdag ``ComplexShiftDefaultTabular``)
- ``ComplexIntercept``   — 8-8 ReLU MLP -> n_params, no bias   (term "ci",
                            tramdag ``ComplexInterceptDefaultTabular``)
- ``SimpleIntercept``    — free parameter vector (no parent dependence)

Parent features follow tramdag's encoding: continuous parents enter raw
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
    """Transform parameters as a function of the (joint) ci-parent features."""

    def __init__(self, n_features: int, n_params: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 8), nn.ReLU(),
            nn.Linear(8, 8), nn.ReLU(),
            nn.Linear(8, n_params, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


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
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)
