"""Progress prediction losses: Huber + monotonicity + ranking."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgressLoss(nn.Module):
    """Combined progress loss with regression, monotonicity, and ranking terms.

    Args:
        lambda_hub: Weight for Huber regression loss.
        lambda_mono: Weight for monotonicity penalty.
        lambda_rank: Weight for ranking loss.
        num_rank_pairs: Number of random pairs to sample per window for ranking.
    """

    def __init__(
        self,
        lambda_hub: float = 1.0,
        lambda_mono: float = 0.2,
        lambda_rank: float = 0.2,
        num_rank_pairs: int = 32,
    ):
        super().__init__()
        self.lambda_hub = lambda_hub
        self.lambda_mono = lambda_mono
        self.lambda_rank = lambda_rank
        self.num_rank_pairs = num_rank_pairs

    def forward(
        self,
        p_pred: torch.Tensor,
        p_target: torch.Tensor,
        confidence: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute combined progress loss.

        Args:
            p_pred: (B, W) predicted progress.
            p_target: (B, W) target progress.
            confidence: (B, W) confidence weights.
            mask: (B, W) attention mask.

        Returns:
            (total_loss, {"huber": ..., "mono": ..., "rank": ...})
        """
        B, W = p_pred.shape

        # --- Huber regression loss (confidence-weighted) ---
        huber = F.smooth_l1_loss(p_pred, p_target, reduction="none")  # (B, W)
        weights = 0.3 + 0.7 * confidence
        l_hub = (huber * weights * mask).sum() / mask.sum().clamp(min=1)

        # --- Monotonicity penalty ---
        # Penalize p_{t-1} > p_t for consecutive valid frames
        if W > 1:
            diff = p_pred[:, :-1] - p_pred[:, 1:]  # (B, W-1), positive = violation
            mono_penalty = torch.clamp(diff, min=0) ** 2
            mono_mask = mask[:, :-1] * mask[:, 1:]
            l_mono = (mono_penalty * mono_mask).sum() / mono_mask.sum().clamp(min=1)
        else:
            l_mono = torch.tensor(0.0, device=p_pred.device)

        # --- Ranking loss ---
        # Sample pairs (i, j) with i < j within each window; enforce p_i < p_j
        l_rank = self._ranking_loss(p_pred, mask)

        total = (
            self.lambda_hub * l_hub
            + self.lambda_mono * l_mono
            + self.lambda_rank * l_rank
        )

        return total, {"huber": l_hub, "mono": l_mono, "rank": l_rank}

    def _ranking_loss(
        self, p_pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Pairwise ranking loss: sample pairs with i < j, enforce p_i < p_j.

        Biased toward large temporal separations for stronger signal.
        """
        B, W = p_pred.shape
        if W < 2:
            return torch.tensor(0.0, device=p_pred.device)

        device = p_pred.device
        total_loss = torch.tensor(0.0, device=device)
        count = 0

        for b in range(B):
            # Find valid positions
            valid = mask[b].nonzero(as_tuple=True)[0]
            n_valid = len(valid)
            if n_valid < 2:
                continue

            # Sample pairs: bias toward large separation
            n_pairs = min(self.num_rank_pairs, n_valid * (n_valid - 1) // 2)
            i_idx = torch.randint(0, n_valid - 1, (n_pairs,), device=device)
            # Bias j toward end of sequence for larger separation
            j_offset = torch.randint(1, n_valid, (n_pairs,), device=device)
            j_idx = torch.clamp(i_idx + j_offset, max=n_valid - 1)

            # Ensure i < j
            swap = i_idx >= j_idx
            i_idx[swap], j_idx[swap] = j_idx[swap].clone(), i_idx[swap].clone()
            # Remove degenerate pairs
            keep = i_idx < j_idx
            if keep.sum() == 0:
                continue
            i_idx, j_idx = i_idx[keep], j_idx[keep]

            p_i = p_pred[b, valid[i_idx]]
            p_j = p_pred[b, valid[j_idx]]

            # Ranking loss: log(1 + exp(-(p_j - p_i)))
            pair_loss = torch.log1p(torch.exp(-(p_j - p_i)))
            total_loss = total_loss + pair_loss.mean()
            count += 1

        return total_loss / max(count, 1)
