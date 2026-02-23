"""Boundary prediction loss: soft BCE with class imbalance correction."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryLoss(nn.Module):
    """Binary cross-entropy for boundary prediction with soft targets.

    Uses pos_weight to correct for boundary sparsity (boundaries are rare,
    ~10-15% of frames are near boundaries). Without this, the boundary head
    collapses to always predicting 0.

    Includes an optional sparsity penalty L_sparse = mean(b̂_t) that
    discourages the model from predicting high boundary probability
    everywhere (helps precision when recall is already high).

    Args:
        pos_weight: Weight for positive class (#neg/#pos). ~5-6 for this dataset.
        focal_gamma: Focal BCE exponent (0 = disabled).
        sparsity_weight: Weight for L_sparse = mean(b̂_t). 0 = disabled.
    """

    def __init__(
        self,
        pos_weight: float = 6.0,
        focal_gamma: float = 0.0,
        sparsity_weight: float = 0.0,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.focal_gamma = focal_gamma
        self.sparsity_weight = sparsity_weight

    def forward(
        self,
        b_pred: torch.Tensor,
        b_target: torch.Tensor,
        confidence: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute boundary loss.

        Args:
            b_pred: (B, W) predicted boundary probability.
            b_target: (B, W) soft boundary targets.
            confidence: (B, W) confidence weights.
            mask: (B, W) attention mask.

        Returns:
            Scalar loss.
        """
        # Clamp predictions to avoid log(0)
        eps = 1e-7
        b_pred = b_pred.clamp(eps, 1 - eps)

        if self.focal_gamma > 0:
            loss = self._focal_bce(b_pred, b_target)
        else:
            # Weighted BCE: higher weight for positive (boundary) frames
            # BCE = -[w_pos * y * log(p) + (1-y) * log(1-p)]
            pos_term = self.pos_weight * b_target * torch.log(b_pred)
            neg_term = (1 - b_target) * torch.log(1 - b_pred)
            loss = -(pos_term + neg_term)

        # Weight by confidence and mask
        weights = 0.3 + 0.7 * confidence
        loss = (loss * weights * mask).sum() / mask.sum().clamp(min=1)

        # Sparsity penalty: penalize high b̂ everywhere → improves precision
        if self.sparsity_weight > 0:
            l_sparse = (b_pred * mask).sum() / mask.sum().clamp(min=1)
            loss = loss + self.sparsity_weight * l_sparse

        return loss

    def _focal_bce(
        self, b_pred: torch.Tensor, b_target: torch.Tensor
    ) -> torch.Tensor:
        """Focal binary cross-entropy."""
        bce = -(b_target * torch.log(b_pred) + (1 - b_target) * torch.log(1 - b_pred))
        # p_t = predicted probability for the true class
        p_t = b_target * b_pred + (1 - b_target) * (1 - b_pred)
        focal = (1 - p_t) ** self.focal_gamma
        # Apply pos_weight to positive class
        class_weight = b_target * self.pos_weight + (1 - b_target)
        return focal * class_weight * bce
