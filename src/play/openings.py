"""Famous chess openings for the bot's first 4 moves.

Each opening is a tuple of ``(name, [uci_moves])`` where
``uci_moves`` contains **8 half-moves** (4 white + 4 black)
in UCI notation.  The bot randomly selects one opening at
the start of a game and replays its side's moves before
switching to model-based evaluation.
"""

from __future__ import annotations

import random

# ── Opening book ────────────────────────────────────────
# Each entry: (opening name, list of 8 half-moves in UCI).

OPENINGS: list[tuple[str, list[str]]] = [
    # — Open games ——————————————————————————————————————
    (
        "Italian Game",
        ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "d2d3", "g8f6"],
    ),
    (
        "Ruy López",
        ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"],
    ),
    (
        "Scotch Game",
        ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "f3d4", "f8c5"],
    ),
    (
        "King's Gambit",
        ["e2e4", "e7e5", "f2f4", "e5f4", "g1f3", "g7g5", "f1c4", "f8g7"],
    ),
    (
        "Vienna Game",
        ["e2e4", "e7e5", "b1c3", "g8f6", "f1c4", "b8c6", "d2d3", "f8c5"],
    ),
    (
        "Four Knights Game",
        ["e2e4", "e7e5", "g1f3", "b8c6", "b1c3", "g8f6", "f1b5", "f8b4"],
    ),
    # — Semi-open games ————————————————————————————————
    (
        "Sicilian Dragon",
        ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6"],
    ),
    (
        "French Defence",
        ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6", "c1g5", "f8e7"],
    ),
    (
        "Caro-Kann Defence",
        ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "b8d7"],
    ),
    (
        "Pirc Defence",
        ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6", "f1c4", "f8g7"],
    ),
    (
        "Alekhine Defence",
        ["e2e4", "g8f6", "e4e5", "f6d5", "d2d4", "d7d6", "g1f3", "c8g4"],
    ),
    # — Closed / d4 openings ———————————————————————————
    (
        "Queen's Gambit Declined",
        ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7"],
    ),
    (
        "Queen's Gambit Accepted",
        ["d2d4", "d7d5", "c2c4", "d5c4", "g1f3", "g8f6", "e2e3", "e7e6"],
    ),
    (
        "Slav Defence",
        ["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6", "b1c3", "d5c4"],
    ),
    (
        "King's Indian Defence",
        ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6"],
    ),
    (
        "Nimzo-Indian Defence",
        ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "d1c2", "d7d5"],
    ),
    (
        "Grünfeld Defence",
        ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5", "c4d5", "f6d5"],
    ),
    (
        "Catalan Opening",
        ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3", "d7d5", "f1g2", "f8e7"],
    ),
    (
        "Dutch Defence",
        ["d2d4", "f7f5", "c2c4", "g8f6", "g2g3", "e7e6", "f1g2", "f8e7"],
    ),
    # — Flank openings ——————————————————————————————————
    (
        "English Opening",
        ["c2c4", "e7e5", "b1c3", "g8f6", "g1f3", "b8c6", "g2g3", "d7d5"],
    ),
    (
        "Réti Opening",
        ["g1f3", "d7d5", "g2g3", "g8f6", "f1g2", "g7g6", "e1g1", "f8g7"],
    ),
]


def pick_opening(
    rng: random.Random | None = None,
) -> tuple[str, list[str]]:
    """Randomly select a famous opening.

    Args:
        rng: Optional ``random.Random`` instance for
            deterministic selection.

    Returns:
        ``(name, uci_moves)`` where *uci_moves* has 8
        half-moves.
    """
    rng = rng or random.Random()
    return rng.choice(OPENINGS)
