"""GRU backbone for temporal phase inference."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GRUBackbone(nn.Module):
    """Bidirectional=False GRU over a sliding window of embeddings.

    Args:
        input_dim: Dimension of input embeddings.
        d_model: GRU hidden dimension.
        num_layers: Number of GRU layers.
        dropout: Dropout rate (between layers).
    """

    def __init__(
        self,
        input_dim: int = 2048,
        d_model: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, W, input_dim) embedding window.
            attention_mask: (B, W) float, 1.0 for real frames, 0.0 for padding.

        Returns:
            (B, W, d_model) hidden states.
        """
        B, W, _ = x.shape
        h = self.input_proj(x)  # (B, W, d_model)

        if attention_mask is not None:
            # Compute actual lengths from mask
            lengths = attention_mask.sum(dim=1).long().clamp(min=1)  # (B,)
            # Pack for efficient RNN processing
            packed = pack_padded_sequence(
                h, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.gru(packed)
            h, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=W)
        else:
            h, _ = self.gru(h)

        return h  # (B, W, d_model)
