"""Render a ChessPosition to an image using chessboard-image.
Default usage examples:


uv run python -m src.chess.visualize                    # starting position -> board.png
uv run python -m src.chess.visualize --name sicilian --fen "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"
uv run python -m src.chess.visualize --pov black --theme sakura --name custom
"""

from __future__ import annotations
import argparse
import warnings
from pathlib import Path
from typing import Literal

# Suppress pkg_resources deprecation warning from chessboard-image.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    from chessboard_image import fen_to_image, fen_to_pil

from PIL import Image

from src.chess.base import ChessPosition
from src.config.paths import IMG_DIR
from src.config.render import BOARD_DEFAULTS

CHESS_BOARD_DIR = IMG_DIR / "chess_boards"


Pov = Literal["white", "black"]
Theme = Literal["alpha", "wikipedia", "uscf", "wisteria", "sakura", "maestro"]


class BoardRenderer:
    """Generates chess board images from ChessPosition objects.

    Args:
        size (int): Board image width/height in pixels,
            (default=400).
        theme (Theme): Piece/board theme name,
            (default="wikipedia").
        show_coordinates (bool): Whether to draw file/rank
            labels on the board, (default=True).
    """

    def __init__(
        self,
        size: int = BOARD_DEFAULTS.size,
        theme: Theme = BOARD_DEFAULTS.theme,
        show_coordinates: bool = True,
    ) -> None:
        self.size = size
        self.theme = theme
        self.show_coordinates = show_coordinates

    def to_pil(
        self,
        position: ChessPosition,
        pov: Pov = "white",
    ) -> Image.Image:
        """Return a PIL Image of the position.

        Args:
            position (ChessPosition): Board state to render.
            pov (Pov): Player point of view,
                (default="white").

        Returns:
            Image.Image.
        """
        return fen_to_pil(
            position.to_fen(),
            size=self.size,
            theme_name=self.theme,
            player_pov=pov,
            show_coordinates=self.show_coordinates,
        )

    def save(
        self,
        position: ChessPosition,
        path: str | Path | None = None,
        pov: Pov = "white",
    ) -> Path:
        """Render the position and save to disk.

        Args:
            position (ChessPosition): Board state to render.
            path (str | Path | None): Output file path,
                (default=CHESS_BOARD_DIR/board.png).
            pov (Pov): Player point of view,
                (default="white").

        Returns:
            Path.
        """
        if path is None:
            path = CHESS_BOARD_DIR / (
                f"{BOARD_DEFAULTS.default_name}.png"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fen_to_image(
            position.to_fen(),
            output_path=str(path),
            size=self.size,
            theme_name=self.theme,
            player_pov=pov,
            show_coordinates=self.show_coordinates,
        )
        return path

def main(args: argparse.Namespace, position: ChessPosition) -> None:
    renderer = BoardRenderer(
        size=args.size, theme=args.theme,
    )
    out = renderer.save(
        position,
        path=CHESS_BOARD_DIR / f"{args.name}.png",
        pov=args.pov,
    )
    print(f"Saved to {out}")


def build_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Render a chess position to an image.",
    )
    parser.add_argument(
        "--fen",
        default=None,
        help="FEN string to render (default: starting position).",
    )
    parser.add_argument(
        "--name",
        default=BOARD_DEFAULTS.default_name,
        help="Output filename without extension (default: board).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=BOARD_DEFAULTS.size,
        help="Image size in pixels (default: 400).",
    )
    parser.add_argument(
        "--theme",
        default=BOARD_DEFAULTS.theme,
        choices=[
            "alpha", "wikipedia", "uscf",
            "wisteria", "sakura", "maestro",
        ],
        help="Board theme (default: wikipedia).",
    )
    parser.add_argument(
        "--pov",
        default=BOARD_DEFAULTS.pov,
        choices=["white", "black"],
        help="Player point of view (default: white).",
    )
    args = parser.parse_args()

    if args.fen:
        position = ChessPosition.from_fen(args.fen)
    else:
        position = ChessPosition.starting()

    return args, position
if __name__ == "__main__":
    args, position = build_args()
    main(args, position)