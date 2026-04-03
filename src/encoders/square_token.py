"""Square-token board encoder.

Encodes a FEN string into an integer token sequence ready
for an embedding layer.

Token layout (72 tokens):
    0      [CLS] — aggregation token.
    1-64   One token per square (a8, b8, …, h1) encoding
           the piece on that square (13 classes: empty +
           12 piece types).
    65     Side to move (0 = black, 1 = white).
    66-69  Castling rights (0 or 1 each for K Q k q).
    70     En-passant file index (0-7) or 8 if none.
    71     Halfmove clock (0-100, clamped).
"""

from __future__ import annotations

import torch

from src.chess.base import FILES
from src.config.encoder import SQUARE_TOKEN
from src.encoders.base import BoardEncoder

# Vocabulary: 0 = empty, 1-12 = piece types.
_PIECE_TOKEN: dict[str, int] = {p: i + 1 for i, p in enumerate(SQUARE_TOKEN.piece_order)}

# Re-export for backwards compatibility.
SEQ_LEN = SQUARE_TOKEN.seq_len
VOCAB_SIZE = SQUARE_TOKEN.vocab_size


class SquareTokenEncoder(BoardEncoder):
    """Encode a board as a length-72 int64 token sequence.

    See module docstring for the token layout.
    """

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (SQUARE_TOKEN.seq_len,)

    def encode(self, fen: str) -> torch.Tensor:
        """Convert a FEN string to a (72,) int64 tensor.

        Args:
            fen (str): Full FEN string (6 fields).

        Returns:
            torch.Tensor of shape ``(72,)`` and dtype
            ``int64``.
        """
        parts = fen.split()
        placement = parts[0]
        active = parts[1]
        castling = parts[2]
        ep = parts[3]
        halfmove = int(parts[4])

        tokens = [SQUARE_TOKEN.cls_id]

        # Square tokens (a8 → h1, reading order).
        for ch in placement:
            if ch == "/":
                continue
            elif ch.isdigit():
                tokens.extend(
                    [SQUARE_TOKEN.empty_token] * int(ch),
                )
            else:
                tokens.append(_PIECE_TOKEN[ch])

        # Side to move (offset into shared vocab).
        tokens.append(
            SQUARE_TOKEN.side_offset + (1 if active == "w" else 0),
        )

        # Castling rights (4 binary tokens).
        for flag in "KQkq":
            tokens.append(
                SQUARE_TOKEN.castle_offset + (1 if flag in castling else 0),
            )

        # En-passant file (0-7) or 8 for none.
        if ep == "-":
            tokens.append(SQUARE_TOKEN.ep_offset + 8)
        else:
            tokens.append(
                SQUARE_TOKEN.ep_offset + FILES.index(ep[0]),
            )

        # Halfmove clock (clamped to 0-100).
        tokens.append(
            SQUARE_TOKEN.halfmove_offset + min(halfmove, SQUARE_TOKEN.max_halfmove),
        )

        return torch.tensor(tokens, dtype=torch.int64)
