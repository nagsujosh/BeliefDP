"""Prediction heads for phase, progress, and boundary."""
from __future__ import annotations

import torch
import torch.nn as nn


class PhaseHead(nn.Module):
    """Predicts phase logits with optional temperature scaling.

    Args:
        d_model: Input hidden dimension.
        num_phases: Number of phase classes K.
        dropout: Dropout rate.
        temperature: Initial temperature for logit scaling.
        learn_temperature: Whether temperature is a learnable parameter.
    """

    def __init__(
        self,
        d_model: int,
        num_phases: int,
        dropout: float = 0.1,
        temperature: float = 1.0,
        learn_temperature: bool = True,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_phases),
        )
        if learn_temperature:
            # Store log(T) as parameter to ensure T > 0
            self.log_temperature = nn.Parameter(
                torch.tensor(float(temperature)).log()
            )
        else:
            self.register_buffer(
                "log_temperature", torch.tensor(float(temperature)).log()
            )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Predict phase logits.

        Args:
            h: (..., d_model) hidden states.

        Returns:
            (..., K) raw logits divided by temperature.
        """
        logits = self.mlp(h)
        return logits / self.temperature


class ProgressHead(nn.Module):
    """Predicts progress scalar in [0, 1]."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Predict progress.

        Args:
            h: (..., d_model) hidden states.

        Returns:
            (...,) scalar progress in [0, 1].
        """
        return torch.sigmoid(self.mlp(h)).squeeze(-1)


class BoundaryHead(nn.Module):
    """Predicts boundary probability in [0, 1]."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Predict boundary probability.

        Args:
            h: (..., d_model) hidden states.

        Returns:
            (...,) scalar boundary probability in [0, 1].
        """
        return torch.sigmoid(self.mlp(h)).squeeze(-1)
