"""Emission head for Neural HMM: maps hidden states to per-phase log-likelihoods.

The emission head produces log p(o_t | z_t = k) for each phase k,
which serves as the observation likelihood in the HMM forward algorithm.

Unlike the PhaseHead which produces logits for direct classification,
the emission head produces log-probabilities normalized over phases,
suitable for Bayesian belief updates.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmissionHead(nn.Module):
    """Produces emission log-probabilities for the Neural HMM.

    Architecture:
        h_t → Linear(D, D) → ReLU → Dropout → Linear(D, K) → log_softmax
    
    Output semantics:
        log p(o_t | z_t = k) — how likely the observation is under each phase.
        These are passed to the NeuralHMM forward algorithm.

    Also exposes raw logits (pre-softmax) for direct phase supervision
    when training the emission head with cross-entropy.

    Args:
        d_model: Hidden state dimension from backbone.
        num_phases: Number of discrete phases K.
        dropout: Dropout rate in MLP.
        temperature: Softmax temperature for emission sharpness.
            Higher T → softer emissions → HMM transitions dominate.
            Lower T → sharper emissions → more responsive to observations.
        learn_temperature: If True, temperature is a learnable parameter.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_phases: int = 11,
        dropout: float = 0.1,
        temperature: float = 1.0,
        learn_temperature: bool = True,
    ):
        super().__init__()
        self.K = num_phases

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_phases),
        )

        if learn_temperature:
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

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute emission log-probabilities and raw logits.

        Args:
            h: (..., d_model) hidden states from backbone.

        Returns:
            Dict with:
                emission_logits: (..., K) raw logits (for direct CE supervision).
                emission_log_probs: (..., K) log p(o_t | z_t = k) (for HMM).
        """
        logits = self.mlp(h)                                    # (..., K)
        scaled_logits = logits / self.temperature               # (..., K)
        log_probs = F.log_softmax(scaled_logits, dim=-1)        # (..., K)

        return {
            "emission_logits": scaled_logits,   # For direct phase CE loss
            "emission_log_probs": log_probs,    # For HMM forward algorithm
        }
