"""Configuration for board encoders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PiecePlaneLayout:
    """Channel layout constants for the piece-plane encoder.

    Channel map (19 total):
        0-11   Piece planes  (P N B R Q K p n b r q k).
           12  Side to move  (1 = white).
        13-16  Castling       (K Q k q rights).
           17  En passant     (target square).
           18  Halfmove clock (normalised to [0, 1]).
    """

    # Ordered piece symbols defining channel indices.
    piece_order: str = "PNBRQKpnbrqk"

    # Number of piece-type channels.
    n_piece: int = 12

    # Metadata channels: side, castling×4, EP, halfmove.
    n_meta: int = 7

    # Total channels (n_piece + n_meta).
    num_channels: int = 19

    # Max halfmove clock before clamping (50-move rule).
    max_halfmove: int = 100

    # Board spatial dimension.
    board_size: int = 8


@dataclass(frozen=True)
class SquareTokenLayout:
    """Token layout constants for the square-token encoder.

    Token map (72 tokens):
        0      [CLS] aggregation token.
        1-64   One token per square (a8 … h1).
        65     Side to move (0 = black, 1 = white).
        66-69  Castling rights (K Q k q).
        70     En-passant file (0-7) or 8 = none.
        71     Halfmove clock (0-100, clamped).

    Vocabulary (128 IDs):
        0-12   Empty + 12 piece types.
        13-14  Side to move.
        15-16  Castling flag.
        17-25  En-passant file (8 files + none).
        26-126 Halfmove clock (0-100).
        127    [CLS] token.
    """

    # Ordered piece symbols for token IDs 1-12.
    piece_order: str = "PNBRQKpnbrqk"

    # Token ID for empty squares.
    empty_token: int = 0

    # Number of board squares.
    n_squares: int = 64

    # Number of metadata tokens.
    n_meta: int = 7

    # Total sequence length: 1 (CLS) + n_squares + n_meta.
    seq_len: int = 72

    # --- Vocabulary offsets ---

    # Side-to-move offset (13 = black, 14 = white).
    side_offset: int = 13

    # Castling offset (15 = no, 16 = yes).
    castle_offset: int = 15

    # En-passant file offset (17-24 = a-h, 25 = none).
    ep_offset: int = 17

    # Halfmove clock offset (26-126 = 0-100).
    halfmove_offset: int = 26

    # Dedicated [CLS] token ID.
    cls_id: int = 127

    # Total vocabulary size.
    vocab_size: int = 128

    # Max halfmove clock before clamping.
    max_halfmove: int = 100


# Module-level singletons.
PIECE_PLANE = PiecePlaneLayout()
SQUARE_TOKEN = SquareTokenLayout()
