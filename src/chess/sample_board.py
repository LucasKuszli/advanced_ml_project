"""Sample random positions from split CSVs and render with
a win-probability color bar.

Usage:
    uv run python -m src.chess.sample_board
    uv run python -m src.chess.sample_board --n 5
    uv run python -m src.chess.sample_board --split val
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.chess.base import ChessPosition
from src.chess.visualize import BoardRenderer, CHESS_BOARD_DIR
from src.config.paths import DATA_DIR
from src.config.render import BOARD_DEFAULTS, EVAL_BAR

SPLIT_DIR = DATA_DIR / "splits"

# Black-win (0) -> draw (0.5) -> white-win (1).
_EVAL_CMAP = LinearSegmentedColormap.from_list(
    "eval", list(EVAL_BAR.gradient_colors),
)


def _load_split(split: str) -> list[tuple[str, float]]:
    """Load (fen, win_prob) pairs from a split CSV.

    Args:
        split (str): One of "train", "val", or "test".

    Returns:
        list[tuple[str, float]].
    """
    path = SPLIT_DIR / f"{split}.csv"
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["fen"], float(row["win_prob"])))
    return rows


def render_with_eval_bar(
    fen: str,
    win_prob: float,
    size: int = BOARD_DEFAULTS.size,
) -> plt.Figure:
    """Render a board with a vertical win-probability bar.

    Args:
        fen (str): FEN string.
        win_prob (float): Stockfish win probability in [0, 1].
        size (int): Board image size in pixels, (default=400).

    Returns:
        matplotlib.figure.Figure.
    """
    pos = ChessPosition.from_fen(fen)
    renderer = BoardRenderer(size=size)
    board_img = renderer.to_pil(pos)

    fig = plt.figure(figsize=EVAL_BAR.figsize)
    gs = gridspec.GridSpec(
        1, 2,
        width_ratios=list(EVAL_BAR.board_bar_ratio),
        wspace=EVAL_BAR.wspace,
    )

    # Board image.
    ax_board = fig.add_subplot(gs[0])
    ax_board.imshow(board_img)
    ax_board.set_axis_off()

    # Color bar showing win probability.
    ax_bar = fig.add_subplot(gs[1])
    gradient = np.linspace(
        1, 0, EVAL_BAR.gradient_steps,
    ).reshape(-1, 1)
    ax_bar.imshow(
        gradient, aspect="auto", cmap=_EVAL_CMAP,
        extent=[0, 1, 0, 1],
    )

    # Marker for the current evaluation.
    ax_bar.axhline(
        y=win_prob,
        color=EVAL_BAR.marker_color,
        linewidth=EVAL_BAR.marker_linewidth,
    )
    ax_bar.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_bar.set_yticklabels(
        ["0.0", "0.25", "0.5", "0.75", "1.0"],
        fontsize=EVAL_BAR.tick_fontsize,
    )
    ax_bar.yaxis.tick_right()
    ax_bar.set_xticks([])
    ax_bar.set_title(
        "W%", fontsize=EVAL_BAR.title_fontsize,
    )

    fig.suptitle(
        f"Win prob: {win_prob:.3f}",
        fontsize=EVAL_BAR.suptitle_fontsize,
        y=EVAL_BAR.suptitle_y,
    )
    return fig


def sample_and_render(
    split: str = "train",
    n: int = 1,
    seed: int | None = None,
) -> list[Path]:
    """Sample *n* random positions and save rendered images.

    Args:
        split (str): Data split to sample from, (default="train").
        n (int): Number of positions to sample, (default=1).
        seed (int | None): Random seed, (default=None).

    Returns:
        list[Path].
    """
    data = _load_split(split)
    if seed is not None:
        random.seed(seed)
    samples = random.sample(data, min(n, len(data)))

    out_dir = CHESS_BOARD_DIR / "stockfish_gt"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for i, (fen, win_prob) in enumerate(samples):
        fig = render_with_eval_bar(fen, win_prob)
        path = out_dir / f"sample_{i}.png"
        fig.savefig(
            path,
            dpi=EVAL_BAR.save_dpi,
            bbox_inches="tight",
        )
        plt.close(fig)
        print(f"Saved {path}")
        paths.append(path)

    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample and visualize chess positions.",
    )
    parser.add_argument(
        "--n", type=int, default=1,
        help="Number of positions to sample (default: 1).",
    )
    parser.add_argument(
        "--split", default="train",
        choices=["train", "val", "test"],
        help="Data split to sample from (default: train).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()
    sample_and_render(
        split=args.split, n=args.n, seed=args.seed,
    )
