"""Consistency losses: phase-change ↔ boundary alignment + left-to-right penalty."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyLoss(nn.Module):
    """Consistency between phase, boundary, and temporal ordering predictions.

    L_link: Phase-change magnitude should correlate with boundary probability.
    L_ltr: Expected phase index should not decrease over time.

    Args:
        lambda_link: Weight for link loss.
        lambda_ltr: Weight for left-to-right loss.
    """

    def __init__(self, lambda_link: float = 1.0, lambda_ltr: float = 0.5):
        super().__init__()
        self.lambda_link = lambda_link
        self.lambda_ltr = lambda_ltr

    def forward(
        self,
        z_logits: torch.Tensor,
        boundary: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute consistency losses.

        Args:
            z_logits: (B, W, K) phase logits.
            boundary: (B, W) predicted boundary probabilities.
            mask: (B, W) attention mask.

        Returns:
            (total_loss, {"link": ..., "ltr": ...})
        """
        B, W, K = z_logits.shape

        if W < 2:
            zero = torch.tensor(0.0, device=z_logits.device)
            return zero, {"link": zero, "ltr": zero}

        q = torch.softmax(z_logits, dim=-1)  # (B, W, K)

        # Consecutive-frame mask (both frames valid)
        pair_mask = mask[:, :-1] * mask[:, 1:]  # (B, W-1)

        # --- L_link: phase-change magnitude <-> boundary probability ---
        # Total variation between consecutive phase distributions
        s_t = 0.5 * (q[:, 1:] - q[:, :-1]).abs().sum(dim=-1)  # (B, W-1)
        s_t = s_t.clamp(0, 1)

        # BCE between boundary prediction and phase-change magnitude
        b_t = boundary[:, 1:].clamp(1e-7, 1 - 1e-7)  # (B, W-1)
        l_link = -(s_t * torch.log(b_t) + (1 - s_t) * torch.log(1 - b_t))
        l_link = (l_link * pair_mask).sum() / pair_mask.sum().clamp(min=1)

        # --- L_ltr: left-to-right phase expectation ---
        # Expected phase index
        phase_indices = torch.arange(K, dtype=q.dtype, device=q.device)
        E_t = (q * phase_indices).sum(dim=-1)  # (B, W)

        # Penalize E_{t-1} > E_t (backward movement)
        backward = torch.clamp(E_t[:, :-1] - E_t[:, 1:], min=0) ** 2  # (B, W-1)
        l_ltr = (backward * pair_mask).sum() / pair_mask.sum().clamp(min=1)

        total = self.lambda_link * l_link + self.lambda_ltr * l_ltr
        return total, {"link": l_link, "ltr": l_ltr}
