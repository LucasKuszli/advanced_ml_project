"""Enriched piece-plane board encoders.

Extends the base ``PiecePlaneEncoder`` with dynamic move-aware
and structural chess planes computed via python-chess.

Two encoders are provided for ablation:

``DynamicPiecePlaneEncoder`` — 31 channels
    19 base + 12 dynamic (attacks, defense, reachability,
    pins, check).

``FullPiecePlaneEncoder`` — 37 channels
    19 base + 12 dynamic + 6 structural (doubled, isolated,
    passed pawns).

Channel map
-----------
::

    0-11   Piece planes  (P N B R Q K p n b r q k).
       12  Side to move  (1 = white).
    13-16  Castling       (K Q k q rights).
       17  En passant     (target square).
       18  Halfmove clock (normalised to [0, 1]).
    ---- dynamic (move-aware) ----
       19  White attack map.
       20  Black attack map.
       21  White defense map.
       22  Black defense map.
       23  White attack count per square.
       24  Black attack count per square.
       25  White reachability (legal / pseudo-legal destinations).
       26  Black reachability (legal / pseudo-legal destinations).
       27  White pinned pieces.
       28  Black pinned pieces.
       29  White king in check (uniform).
       30  Black king in check (uniform).
    ---- structural (pawn) ----
       31  White doubled pawns.
       32  Black doubled pawns.
       33  White isolated pawns.
       34  Black isolated pawns.
       35  White passed pawns.
       36  Black passed pawns.
"""

from __future__ import annotations

import chess
import torch

from src.config.encoder import DYNAMIC_PIECE_PLANE, FULL_PIECE_PLANE
from src.encoders.base import BoardEncoder
from src.encoders.features import (
    compute_dynamic_features,
    compute_structural_features,
)
from src.encoders.piece_plane import PiecePlaneEncoder


class DynamicPiecePlaneEncoder(BoardEncoder):
    """Encode a board as ``(31, 8, 8)`` planes.

    Channels 0-18 are identical to ``PiecePlaneEncoder``.
    Channels 19-30 are dynamic move-aware planes computed
    from the current board state (see module docstring).
    """

    def __init__(self) -> None:
        self._base = PiecePlaneEncoder()

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (
            DYNAMIC_PIECE_PLANE.num_channels,
            DYNAMIC_PIECE_PLANE.board_size,
            DYNAMIC_PIECE_PLANE.board_size,
        )

    def encode(self, fen: str) -> torch.Tensor:
        """Convert a FEN string to a ``(31, 8, 8)`` tensor.

        Args:
            fen: Full FEN string (6 fields).

        Returns:
            ``torch.Tensor`` of shape ``(31, 8, 8)``.
        """
        base = self._base.encode(fen)
        board = chess.Board(fen)
        dynamic = compute_dynamic_features(board)
        return torch.cat([base, dynamic], dim=0)


class FullPiecePlaneEncoder(BoardEncoder):
    """Encode a board as ``(37, 8, 8)`` planes.

    Channels 0-18: raw piece / state planes.
    Channels 19-30: dynamic move-aware planes.
    Channels 31-36: structural pawn planes.

    See module docstring for the full channel map.
    """

    def __init__(self) -> None:
        self._base = PiecePlaneEncoder()

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (
            FULL_PIECE_PLANE.num_channels,
            FULL_PIECE_PLANE.board_size,
            FULL_PIECE_PLANE.board_size,
        )

    def encode(self, fen: str) -> torch.Tensor:
        """Convert a FEN string to a ``(37, 8, 8)`` tensor.

        Args:
            fen: Full FEN string (6 fields).

        Returns:
            ``torch.Tensor`` of shape ``(37, 8, 8)``.
        """
        base = self._base.encode(fen)
        board = chess.Board(fen)
        dynamic = compute_dynamic_features(board)
        structural = compute_structural_features(board)
        return torch.cat([base, dynamic, structural], dim=0)
