"""Input stems that project encoder outputs into transformer
embedding space.

Each stem converts a specific encoder's output tensor into
a ``(batch, seq_len, d_model)`` float tensor that the
shared transformer body can process.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.config.encoder import PIECE_PLANE, SQUARE_TOKEN
from src.config.model import TRANSFORMER


class PiecePlaneStem(nn.Module):
    """Project ``(B, 19, 8, 8)`` planes into transformer
    tokens.

    The spatial grid is reshaped to 64 square-tokens of
    ``NUM_CHANNELS`` features each, then linearly projected
    to ``d_model``.  A learnable positional embedding is
    added (one per square).

    Args:
        d_model (int): Embedding dimension of the
            transformer body.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(
            PIECE_PLANE.num_channels,
            d_model,
        )
        self.pos_embed = nn.Parameter(
            torch.randn(
                1,
                PIECE_PLANE.board_size**2,
                d_model,
            )
            * TRANSFORMER.pos_embed_init_scale,
        )

    @property
    def seq_len(self) -> int:
        """Number of tokens produced by this stem."""
        return PIECE_PLANE.board_size**2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map piece-plane tensor to transformer input.

        Args:
            x (torch.Tensor): Shape ``(B, C, 8, 8)``.

        Returns:
            torch.Tensor of shape ``(B, 64, d_model)``.
        """
        b = x.shape[0]
        n_sq = PIECE_PLANE.board_size**2
        # (B, C, 8, 8) → (B, C, 64) → (B, 64, C).
        x = x.reshape(
            b,
            PIECE_PLANE.num_channels,
            n_sq,
        ).permute(
            0,
            2,
            1,
        )
        x = self.proj(x) + self.pos_embed
        return x


class SquareTokenStem(nn.Module):
    """Embed ``(B, 72)`` integer tokens into transformer
    space.

    Uses a single ``nn.Embedding`` table (vocab 128) and
    learned positional embeddings.

    Args:
        d_model (int): Embedding dimension of the
            transformer body.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.tok_embed = nn.Embedding(
            SQUARE_TOKEN.vocab_size,
            d_model,
        )
        self.pos_embed = nn.Parameter(
            torch.randn(
                1,
                SQUARE_TOKEN.seq_len,
                d_model,
            )
            * TRANSFORMER.pos_embed_init_scale,
        )

    @property
    def seq_len(self) -> int:
        """Number of tokens produced by this stem."""
        return SQUARE_TOKEN.seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map token-id tensor to transformer input.

        Args:
            x (torch.Tensor): Shape ``(B, 72)``, int64.

        Returns:
            torch.Tensor of shape ``(B, 72, d_model)``.
        """
        return self.tok_embed(x) + self.pos_embed
