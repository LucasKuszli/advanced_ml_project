"""Play chess against a trained ChessTransformer.

Run with:
    uv run python -m src.play
    uv run python -m src.play --help
"""

from src.play.bot import ChessBot, EvalResult, MoveScore
from src.play.game import Game, GamePhase, PlayerColor
from src.play.gui import ChessGUI
from src.play.openings import OPENINGS, pick_opening

__all__ = [
    "ChessBot",
    "ChessGUI",
    "EvalResult",
    "Game",
    "GamePhase",
    "MoveScore",
    "OPENINGS",
    "PlayerColor",
    "pick_opening",
]
