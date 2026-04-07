"""Chess evaluation transformer.

A small encoder-only transformer that predicts Stockfish
win-probability from any board encoder's output.  The
architecture is identical regardless of encoder — only the
input stem differs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config.model import TRANSFORMER
from src.model.input_stem import (
    DynamicPiecePlaneStem,
    FullPiecePlaneStem,
    PiecePlaneStem,
)


class ChessTransformer(nn.Module):
    """Encoder-only transformer for state-value prediction.

    Args:
        stem (nn.Module): An input stem (``PiecePlaneStem``,
            ``DynamicPiecePlaneStem``, or
            ``FullPiecePlaneStem``) that maps encoder
            output to ``(B, S, d_model)``.
        d_model (int): Embedding dimension (default=256).
        n_heads (int): Number of attention heads
            (default=8).
        n_layers (int): Number of transformer encoder
            layers (default=6).
        d_ff (int): Feed-forward hidden dimension.  When
            ``None`` defaults to ``4 * d_model``.
        dropout (float): Dropout rate (default=0.1).
    """

    def __init__(
        self,
        stem: nn.Module,
        d_model: int = TRANSFORMER.d_model,
        n_heads: int = TRANSFORMER.n_heads,
        n_layers: int = TRANSFORMER.n_layers,
        d_ff: int | None = None,
        dropout: float = TRANSFORMER.dropout,
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = TRANSFORMER.ff_factor * d_model

        self.stem = stem

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=TRANSFORMER.activation,
            batch_first=True,
            norm_first=TRANSFORMER.norm_first,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict win-probability from encoded board.

        Args:
            x (torch.Tensor): Raw encoder output — either
                ``(B, 19, 8, 8)`` float for piece-plane or
                ``(B, 72)`` int64 for square-token.

        Returns:
            torch.Tensor of shape ``(B,)`` with predicted
            win-probabilities in ``[0, 1]``.
        """
        x = self.stem(x)  # (B, S, d_model).
        x = self.encoder(x)  # (B, S, d_model).
        x = self.norm(x)
        # Mean-pool over sequence dimension.
        x = x.mean(dim=1)  # (B, d_model).
        x = self.head(x).squeeze(-1)  # (B,).
        return x.sigmoid()


def build_model(
    encoder_name: str,
    d_model: int = TRANSFORMER.d_model,
    n_heads: int = TRANSFORMER.n_heads,
    n_layers: int = TRANSFORMER.n_layers,
    d_ff: int | None = None,
    dropout: float = TRANSFORMER.dropout,
) -> ChessTransformer:
    """Construct a ``ChessTransformer`` for a given encoder.

    Args:
        encoder_name (str): One of ``"piece_plane"``,
            ``"dynamic_piece_plane"``, or
            ``"full_piece_plane"``.
        d_model (int): Embedding dimension (default=256).
        n_heads (int): Attention heads (default=8).
        n_layers (int): Encoder layers (default=6).
        d_ff (int | None): FF hidden dim (default=4*d).
        dropout (float): Dropout rate (default=0.1).

    Returns:
        ChessTransformer.
    """
    stems = {
        "piece_plane": PiecePlaneStem,
        "dynamic_piece_plane": DynamicPiecePlaneStem,
        "full_piece_plane": FullPiecePlaneStem,
    }
    if encoder_name not in stems:
        raise ValueError(f"Unknown encoder {encoder_name!r}. Choose from {list(stems)}.")
    stem = stems[encoder_name](d_model)
    return ChessTransformer(
        stem=stem,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
    )
