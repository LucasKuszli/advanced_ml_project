"""Configuration for chess board rendering and
visualization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoardDefaults:
    """Default rendering options for chess board images."""

    # Board image width and height in pixels.
    size: int = 400

    # Piece and board visual theme name.
    theme: str = "wikipedia"

    # Default player point-of-view.
    pov: str = "white"

    # Default output filename (without extension).
    default_name: str = "board"


@dataclass(frozen=True)
class EvalBarLayout:
    """Layout constants for the win-probability color bar."""

    # Figure size (width, height) in inches.
    figsize: tuple[float, float] = (5.5, 5.0)

    # GridSpec width ratio between board and bar columns.
    board_bar_ratio: tuple[int, int] = (20, 1)

    # Horizontal whitespace between board and bar.
    wspace: float = 0.05

    # Discrete steps in the evaluation colour gradient.
    gradient_steps: int = 256

    # Line width for the evaluation marker.
    marker_linewidth: int = 2

    # Colour of the evaluation marker line.
    marker_color: str = "red"

    # Font size for colour-bar tick labels.
    tick_fontsize: int = 8

    # Font size for the colour-bar axis title.
    title_fontsize: int = 9

    # Font size for the figure suptitle.
    suptitle_fontsize: int = 11

    # Vertical position of suptitle (normalised coords).
    suptitle_y: float = 0.02

    # DPI for saved figure output.
    save_dpi: int = 150

    # Colour stops: black-win → draw → white-win.
    gradient_colors: tuple[str, ...] = (
        "#222222",
        "#888888",
        "#ffffff",
    )


# Module-level singletons.
BOARD_DEFAULTS = BoardDefaults()
EVAL_BAR = EvalBarLayout()
