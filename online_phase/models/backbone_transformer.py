"""Causal Transformer encoder backbone for temporal phase inference."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sine/cosine positional encoding (not learned)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. x: (B, L, D)."""
        return x + self.pe[:, : x.size(1)]


class TransformerBackbone(nn.Module):
    """Causal Transformer encoder over a sliding window of embeddings.

    Args:
        input_dim: Dimension of input embeddings (e.g. 2048 for R3M).
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of TransformerEncoder layers.
        dim_feedforward: Feedforward hidden dimension.
        dropout: Dropout rate.
        max_seq_len: Maximum sequence length for positional encoding.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        # Dropout after input projection — prevents memorizing exact embeddings
        self.input_dropout = nn.Dropout(dropout)

        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # Disable fast path: merge_masks()
                                         # converts bool mask → float, then
                                         # _transformer_encoder_layer_fwd warns
                                         # "Converting mask without torch.bool
                                         # dtype to bool" (attention.cpp:150).
        )

        self._causal_mask_cache: dict[tuple[int, str], torch.Tensor] = {}

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or create a float causal attention mask.

        Uses float -inf mask (upper triangle = -inf, rest = 0.0).
        Only used during training; eval mode uses is_causal=True instead.
        """
        key = (seq_len, str(device))
        if key not in self._causal_mask_cache:
            mask = torch.zeros(seq_len, seq_len, device=device)
            mask.masked_fill_(
                torch.triu(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                    diagonal=1,
                ),
                float("-inf"),
            )
            self._causal_mask_cache[key] = mask
        return self._causal_mask_cache[key]

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

        h = self.input_proj(x)              # (B, W, d_model)
        h = self.input_dropout(h)           # Regularize input representations
        h = self.pos_encoding(h)            # (B, W, d_model)

        # Causal masking strategy (avoids PyTorch 2.0.x "Converting mask
        # without torch.bool" warning from attention.cpp:150):
        #   - Training: standard path uses float -inf mask (matches src.dtype,
        #     _canonical_mask leaves it unchanged → no conversion warning).
        #   - Eval: fast path's merge_masks() always triggers the C++ warning
        #     regardless of mask dtype. Passing mask=None + is_causal=True
        #     lets the fast path generate its own causal mask internally.
        if self.training:
            causal_mask = self._get_causal_mask(W, x.device)
            h = self.transformer_encoder(h, mask=causal_mask)
        else:
            h = self.transformer_encoder(h, mask=None, is_causal=True)

        return h  # (B, W, d_model)
