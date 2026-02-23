"""Structured phase inference model: Transformer + Neural HMM.

Jointly trained structured temporal model:

    R3M embeddings → Causal Transformer → hidden states h_t
                                              ↓
                    ┌─────────────────────────┼─────────────────────────┐
                    ↓                         ↓                         ↓
              Emission head             Progress head             Boundary head
            log p(o_t | z_t)               u_t ∈ [0,1]           b̂_t ∈ [0,1]
                    ↓                                                 ↓
                    └──────────── Neural HMM ◄────────────────────────┘
                                      ↓
                              Phase belief α_t ∈ Δ^{K-1}

The Neural HMM is fully differentiable: boundary predictions modulate
transition probabilities, and beliefs are supervised with cross-entropy.
The model exposes both `z_logits` (emission logits) and `beliefs`
for structured supervision and evaluation.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from online_phase.models.backbone_transformer import TransformerBackbone
from online_phase.models.backbone_gru import GRUBackbone
from online_phase.models.emission_head import EmissionHead
from online_phase.models.heads import ProgressHead, BoundaryHead
from online_phase.models.neural_hmm import NeuralHMM


class PhaseModelStructured(nn.Module):
    """End-to-end structured model: Transformer + Neural HMM.

    Key properties:
    1. Uses EmissionHead (produces log-likelihoods).
    2. Includes a differentiable NeuralHMM that produces belief distributions.
    3. Boundary head output feeds into HMM transition probabilities.
    4. Outputs both emission logits (raw) and HMM beliefs (structured).

    Training uses losses on both:
    - Emission logits → cross-entropy (teaches the head to discriminate phases)
    - HMM beliefs → cross-entropy + NLL (teaches temporally coherent predictions)
    - Progress → ranking + Huber + monotonicity
    - Boundary → BCE + sparsity

    Args:
        config: Configuration dict with model + HMM hyperparameters.
    """

    def __init__(self, config: dict):
        super().__init__()

        backbone_type = config.get("backbone_type", "transformer")
        input_dim = config.get("input_dim", 2048)
        d_model = config.get("d_model", 256)
        num_phases = config.get("num_phases", 11)
        dropout = config.get("dropout", 0.1)

        # --- Backbone (shared) ---
        if backbone_type == "transformer":
            self.backbone = TransformerBackbone(
                input_dim=input_dim,
                d_model=d_model,
                nhead=config.get("nhead", 4),
                num_layers=config.get("num_layers", 4),
                dim_feedforward=config.get("dim_feedforward", 512),
                dropout=dropout,
                max_seq_len=config.get("max_seq_len", 512),
            )
        elif backbone_type == "gru":
            self.backbone = GRUBackbone(
                input_dim=input_dim,
                d_model=d_model,
                num_layers=config.get("num_layers", 2),
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

        # --- Prediction heads ---
        self.emission_head = EmissionHead(
            d_model=d_model,
            num_phases=num_phases,
            dropout=dropout,
            temperature=config.get("temperature", 1.0),
            learn_temperature=config.get("learn_temperature", True),
        )
        self.progress_head = ProgressHead(d_model=d_model, dropout=dropout)
        self.boundary_head = BoundaryHead(d_model=d_model, dropout=dropout)

        # --- Neural HMM inference layer ---
        self.hmm = NeuralHMM(
            K=num_phases,
            eta_min=config.get("hmm_eta_min", 0.02),
            eta_max=config.get("hmm_eta_max", 0.35),
            rho=config.get("hmm_rho", 0.001),
            use_duration_prior=config.get("use_duration_prior", False),
            duration_model=config.get("duration_model", "gaussian"),
            duration_strength=config.get("duration_strength", 1.0),
            min_std=config.get("duration_min_std", 5.0),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass: backbone → heads → HMM → beliefs.

        Args:
            x: (B, W, input_dim) embedding window.
            attention_mask: (B, W) float mask (1=real, 0=padding).

        Returns:
            Dict with:
                z_logits:           (B, W, K) emission logits.
                emission_log_probs: (B, W, K) log p(o_t | z_t) (for HMM input).
                beliefs:            (B, W, K) HMM posterior beliefs α_t.
                log_beliefs:        (B, W, K) log α_t (for CE loss).
                log_normalizers:    (B, W) log Z_t (for NLL loss).
                progress:           (B, W) progress u_t ∈ [0, 1].
                boundary:           (B, W) boundary b̂_t ∈ [0, 1].
                hidden:             (B, W, d_model) backbone hidden states.
        """
        # --- Backbone ---
        hidden = self.backbone(x, attention_mask)  # (B, W, d_model)

        # --- Prediction heads (all timesteps) ---
        emission_out = self.emission_head(hidden)
        emission_logits = emission_out["emission_logits"]        # (B, W, K)
        emission_log_probs = emission_out["emission_log_probs"]  # (B, W, K)

        progress = self.progress_head(hidden)  # (B, W)
        boundary = self.boundary_head(hidden)  # (B, W)

        # --- Neural HMM forward pass ---
        # Boundary probs modulate transition matrices
        hmm_out = self.hmm(
            emission_log_probs=emission_log_probs,
            boundary_probs=boundary,
            mask=attention_mask,
        )

        return {
            # Emission head output
            "z_logits": emission_logits,
            "emission_log_probs": emission_log_probs,
            # HMM beliefs (structured temporal output)
            "beliefs": hmm_out["beliefs"],
            "log_beliefs": hmm_out["log_beliefs"],
            "log_normalizers": hmm_out["log_normalizers"],
            # Other heads
            "progress": progress,
            "boundary": boundary,
            # Internal
            "hidden": hidden,
        }
