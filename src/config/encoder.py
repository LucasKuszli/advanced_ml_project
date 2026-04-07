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
class DynamicPiecePlaneLayout:
    """Channel layout for the dynamic piece-plane encoder.

    Extends PiecePlaneLayout with 12 dynamic move-aware planes.

    Channel map (31 total):
        0-18   Base piece-plane channels (see PiecePlaneLayout).
        19     White attack map (binary).
        20     Black attack map (binary).
        21     White defense map (binary).
        22     Black defense map (binary).
        23     White attack count per square.
        24     Black attack count per square.
        25     White reachability map (binary).
        26     Black reachability map (binary).
        27     White pinned pieces (binary).
        28     Black pinned pieces (binary).
        29     White king in check (uniform binary).
        30     Black king in check (uniform binary).
    """

    n_base: int = 19
    n_dynamic: int = 12
    num_channels: int = 31
    board_size: int = 8


@dataclass(frozen=True)
class FullPiecePlaneLayout:
    """Channel layout for the full piece-plane encoder.

    Extends DynamicPiecePlaneLayout with 6 structural pawn planes.

    Channel map (37 total):
        0-30   Dynamic piece-plane channels (see DynamicPiecePlaneLayout).
        31     White doubled pawns.
        32     Black doubled pawns.
        33     White isolated pawns.
        34     Black isolated pawns.
        35     White passed pawns.
        36     Black passed pawns.
    """

    n_base: int = 19
    n_dynamic: int = 12
    n_structural: int = 6
    num_channels: int = 37
    board_size: int = 8


# Module-level singletons.
PIECE_PLANE = PiecePlaneLayout()
DYNAMIC_PIECE_PLANE = DynamicPiecePlaneLayout()
FULL_PIECE_PLANE = FullPiecePlaneLayout()
