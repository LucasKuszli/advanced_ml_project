"""Input stems that project encoder outputs into transformer
embedding space.

Each stem converts a specific encoder's output tensor into
a ``(batch, seq_len, d_model)`` float tensor that the
shared transformer body can process.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config.encoder import DYNAMIC_PIECE_PLANE, FULL_PIECE_PLANE, PIECE_PLANE
from src.config.model import TRANSFORMER


class PiecePlaneStem(nn.Module):
    """Project ``(B, C, 8, 8)`` planes into transformer
    tokens.

    The spatial grid is reshaped to 64 square-tokens of
    *in_channels* features each, then linearly projected
    to ``d_model``.  A learnable positional embedding is
    added (one per square).

    Args:
        d_model (int): Embedding dimension of the
            transformer body.
        in_channels (int): Number of input channels
            (default=19, matching ``PiecePlaneLayout``).
    """

    def __init__(
        self,
        d_model: int,
        in_channels: int = PIECE_PLANE.num_channels,
    ) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._board_size = PIECE_PLANE.board_size
        self.proj = nn.Linear(in_channels, d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(
                1,
                self._board_size**2,
                d_model,
            )
            * TRANSFORMER.pos_embed_init_scale,
        )

    @property
    def seq_len(self) -> int:
        """Number of tokens produced by this stem."""
        return self._board_size**2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map piece-plane tensor to transformer input.

        Args:
            x (torch.Tensor): Shape ``(B, C, 8, 8)``.

        Returns:
            torch.Tensor of shape ``(B, 64, d_model)``.
        """
        b = x.shape[0]
        n_sq = self._board_size**2
        # (B, C, 8, 8) → (B, C, 64) → (B, 64, C).
        x = x.reshape(b, self._in_channels, n_sq).permute(0, 2, 1)
        x = self.proj(x) + self.pos_embed
        return x


class DynamicPiecePlaneStem(PiecePlaneStem):
    """Project ``(B, 31, 8, 8)`` planes into transformer tokens.

    Identical architecture to ``PiecePlaneStem`` but expects
    31 input channels (19 base + 12 dynamic).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__(d_model, in_channels=DYNAMIC_PIECE_PLANE.num_channels)


class FullPiecePlaneStem(PiecePlaneStem):
    """Project ``(B, 37, 8, 8)`` planes into transformer tokens.

    Identical architecture to ``PiecePlaneStem`` but expects
    37 input channels (19 base + 12 dynamic + 6 structural).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__(d_model, in_channels=FULL_PIECE_PLANE.num_channels)
