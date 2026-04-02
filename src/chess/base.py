"""Core chess position representation and FEN conversion."""

from __future__ import annotations

from dataclasses import dataclass, field

# Uppercase = white, lowercase = black.
PIECE_SYMBOLS = set("KQRBNPkqrbnp")

FILES = "abcdefgh"
RANKS = "12345678"
ALL_SQUARES = [f + r for f in FILES for r in RANKS]

STARTING_POSITION: dict[str, str] = {
    "a1": "R", "b1": "N", "c1": "B", "d1": "Q",
    "e1": "K", "f1": "B", "g1": "N", "h1": "R",
    "a2": "P", "b2": "P", "c2": "P", "d2": "P",
    "e2": "P", "f2": "P", "g2": "P", "h2": "P",
    "a7": "p", "b7": "p", "c7": "p", "d7": "p",
    "e7": "p", "f7": "p", "g7": "p", "h7": "p",
    "a8": "r", "b8": "n", "c8": "b", "d8": "q",
    "e8": "k", "f8": "b", "g8": "n", "h8": "r",
}


@dataclass
class ChessPosition:
    """A chess position stored as a square -> piece mapping.

    Args:
        pieces (dict[str, str]): Square to piece symbol map,
            e.g. {"e4": "K"}, (default=STARTING_POSITION).
        active_color (str): Side to move, "w" or "b",
            (default="w").
        castling (str): Castling availability in FEN
            notation, (default="KQkq").
        en_passant (str): En-passant target square,
            (default="-").
        halfmove_clock (int): Halfmove clock for the
            fifty-move rule, (default=0).
        fullmove_number (int): Fullmove counter,
            (default=1).
    """

    pieces: dict[str, str] = field(default_factory=lambda: dict(STARTING_POSITION))
    active_color: str = "w"
    castling: str = "KQkq"
    en_passant: str = "-"
    halfmove_clock: int = 0
    fullmove_number: int = 1

    def __post_init__(self) -> None:
        for sq, piece in self.pieces.items():
            if sq not in ALL_SQUARES:
                raise ValueError(f"Invalid square: {sq!r}")
            if piece not in PIECE_SYMBOLS:
                raise ValueError(f"Invalid piece symbol: {piece!r} on {sq}")
        if self.active_color not in ("w", "b"):
            raise ValueError(
                f"active_color must be 'w' or 'b',"
                f" got {self.active_color!r}"
            )

    def to_fen(self) -> str:
        """Convert this position to a FEN string.

        Returns:
            str.
        """
        rows: list[str] = []
        for rank in "87654321":
            empty = 0
            row = ""
            for file in FILES:
                piece = self.pieces.get(file + rank)
                if piece:
                    if empty:
                        row += str(empty)
                        empty = 0
                    row += piece
                else:
                    empty += 1
            if empty:
                row += str(empty)
            rows.append(row)

        placement = "/".join(rows)
        return (
            f"{placement} {self.active_color} {self.castling} "
            f"{self.en_passant} {self.halfmove_clock} {self.fullmove_number}"
        )

    @classmethod
    def from_fen(cls, fen: str) -> ChessPosition:
        """Parse a FEN string into a ChessPosition.

        Args:
            fen (str): FEN string with 6 space-separated
                fields.

        Returns:
            ChessPosition.
        """
        parts = fen.split()
        if len(parts) != 6:
            raise ValueError(f"FEN must have 6 space-separated fields, got {len(parts)}")

        placement, active, castling, ep, half, full = parts
        pieces: dict[str, str] = {}
        for rank_idx, row in enumerate(placement.split("/")):
            rank = str(8 - rank_idx)
            file_idx = 0
            for ch in row:
                if ch.isdigit():
                    file_idx += int(ch)
                else:
                    square = FILES[file_idx] + rank
                    pieces[square] = ch
                    file_idx += 1

        return cls(
            pieces=pieces,
            active_color=active,
            castling=castling,
            en_passant=ep,
            halfmove_clock=int(half),
            fullmove_number=int(full),
        )

    @classmethod
    def starting(cls) -> ChessPosition:
        """Return the standard starting position.

        Returns:
            ChessPosition.
        """
        return cls()

    @classmethod
    def empty(cls) -> ChessPosition:
        """Return an empty board.

        Returns:
            ChessPosition.
        """
        return cls(pieces={}, castling="-")
