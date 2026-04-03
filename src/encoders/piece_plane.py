"""Piece-plane board encoder.

Encodes a FEN string into a ``(C, 8, 8)`` float tensor where
each channel is a binary plane for one feature.

Channel layout (19 total):
    0-11  Piece planes  (P N B R Q K p n b r q k).
       12  Side to move  (1 if white, 0 if black).
    13-16  Castling       (K Q k q rights).
       17  En passant     (1 on the target square).
       18  Halfmove clock (normalised to [0, 1]).
"""

from __future__ import annotations

import torch

from src.chess.base import FILES
from src.config.encoder import PIECE_PLANE
from src.encoders.base import BoardEncoder

# Piece symbol -> channel index.
_PIECE_CHANNEL: dict[str, int] = {
    piece: channel for channel, piece in enumerate(PIECE_PLANE.piece_order)
}

# Re-export for backwards compatibility.
NUM_CHANNELS = PIECE_PLANE.num_channels


def _square_to_rc(square: str) -> tuple[int, int]:
    """Map algebraic square to (row, col) for an 8×8 grid.

    Row 0 = rank 8 (top), row 7 = rank 1 (bottom).

    Args:
        square (str): Algebraic notation, e.g. ``"e4"``.

    Returns:
        tuple[int, int].
    """
    col = FILES.index(square[0])
    row = 8 - int(square[1])
    return row, col


class PiecePlaneEncoder(BoardEncoder):
    """Encode a board as ``(19, 8, 8)`` binary-ish planes.

    See module docstring for the channel layout.
    """

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (
            PIECE_PLANE.num_channels,
            PIECE_PLANE.board_size,
            PIECE_PLANE.board_size,
        )

    def encode(self, fen: str) -> torch.Tensor:
        """Convert a FEN string to a (19, 8, 8) tensor.

        Args:
            fen (str): Full FEN string (6 fields).

        Returns:
            torch.Tensor of shape ``(19, 8, 8)``.
        """
        parts = fen.split()
        placement = parts[0]
        active = parts[1]
        castling = parts[2]
        ep = parts[3]
        halfmove = int(parts[4])

        board = torch.zeros(
            PIECE_PLANE.num_channels,
            PIECE_PLANE.board_size,
            PIECE_PLANE.board_size,
            dtype=torch.float32,
        )

        # Piece planes.
        row = 0
        col = 0
        for ch in placement:
            if ch == "/":
                row += 1
                col = 0
            elif ch.isdigit():
                col += int(ch)
            else:
                board[_PIECE_CHANNEL[ch], row, col] = 1.0
                col += 1

        # Side to move.
        if active == "w":
            board[PIECE_PLANE.n_piece] = 1.0

        # Castling rights.
        for i, flag in enumerate("KQkq"):
            if flag in castling:
                board[PIECE_PLANE.n_piece + 1 + i] = 1.0

        # En passant target square.
        if ep != "-":
            r, c = _square_to_rc(ep)
            board[PIECE_PLANE.n_piece + 5, r, c] = 1.0

        # Halfmove clock (normalised).
        board[PIECE_PLANE.n_piece + 6] = min(
            halfmove / PIECE_PLANE.max_halfmove,
            1.0,
        )

        return board
