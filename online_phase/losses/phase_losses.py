"""Phase classification loss: confidence-weighted CE with label smoothing + focal."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseLoss(nn.Module):
    """Confidence-weighted cross-entropy with adaptive label smoothing.

    Label smoothing adapts to per-frame confidence: low-confidence frames
    (far from anchors) get more smoothing. Optional focal loss factor
    for handling class imbalance.

    Args:
        num_phases: Number of phase classes K.
        label_smoothing_max: Maximum label smoothing epsilon.
        confidence_alpha: Base weight floor: w_t = alpha + (1-alpha)*c_t.
        focal_gamma: Focal loss exponent (0 = disabled).
    """

    def __init__(
        self,
        num_phases: int = 11,
        label_smoothing_max: float = 0.1,
        confidence_alpha: float = 0.3,
        focal_gamma: float = 0.0,
    ):
        super().__init__()
        self.K = num_phases
        self.eps_max = label_smoothing_max
        self.alpha = confidence_alpha
        self.focal_gamma = focal_gamma

    def forward(
        self,
        z_logits: torch.Tensor,
        z_target: torch.Tensor,
        confidence: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute phase loss over full sequence.

        Args:
            z_logits: (B, W, K) raw logits.
            z_target: (B, W) int64 phase labels.
            confidence: (B, W) confidence weights.
            mask: (B, W) attention mask (1=real, 0=padding).

        Returns:
            Scalar loss.
        """
        B, W, K = z_logits.shape

        # Adaptive label smoothing: eps_t = eps_max * (1 - c_t)
        eps = self.eps_max * (1.0 - confidence)  # (B, W)

        # Build soft targets: (1 - eps) * onehot + eps/K
        onehot = F.one_hot(z_target, K).float()  # (B, W, K)
        soft_target = (1.0 - eps.unsqueeze(-1)) * onehot + eps.unsqueeze(-1) / K

        # Log-softmax of logits
        log_probs = F.log_softmax(z_logits, dim=-1)  # (B, W, K)

        # Cross-entropy with soft targets: -sum(y * log_q)
        ce = -(soft_target * log_probs).sum(dim=-1)  # (B, W)

        # Optional focal factor
        if self.focal_gamma > 0:
            probs = torch.softmax(z_logits, dim=-1)  # (B, W, K)
            p_correct = probs.gather(-1, z_target.unsqueeze(-1)).squeeze(-1)  # (B, W)
            focal = (1.0 - p_correct) ** self.focal_gamma
            ce = ce * focal

        # Confidence weighting: w_t = alpha + (1-alpha)*c_t
        weights = self.alpha + (1.0 - self.alpha) * confidence  # (B, W)

        # Apply mask and weights
        loss = (ce * weights * mask).sum() / mask.sum().clamp(min=1)
        return loss
