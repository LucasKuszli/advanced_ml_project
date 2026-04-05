"""Visualize test-set evaluation results.

Reads ``test_results.json`` and produces bar charts and a
scatter plot of predictions vs. ground-truth labels.

Usage:
    uv run python -m src.evaluate.visualize
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config.paths import IMG_DIR, MODEL_DIR
from src.config.train import RUN_SEEDS, TrainConfig
from src.data.dataset import build_loader
from src.encoders import PiecePlaneEncoder
from src.evaluate.evaluate import _load_model, _predict

MODEL_TAG = "d128_L5"
ENCODER = "piece_plane"
TEST_SPLIT_DIR = Path("data/test_split")
OUT_DIR = IMG_DIR / "evaluation" / ENCODER

DPI = 150
FIGSIZE_BAR = (7.0, 4.5)
FIGSIZE_SCATTER = (5.5, 5.5)
GRID_ALPHA = 0.3
BAR_COLOR = "steelblue"
MEAN_COLOR = "tab:red"


def _load_results() -> dict:
    """Load test_results.json."""
    path = MODEL_DIR / MODEL_TAG / "test_results.json"
    with open(path) as f:
        return json.load(f)


def _bar_chart(
    seeds: list[int],
    values: list[float],
    mean: float,
    std: float,
    ylabel: str,
    title: str,
    out_path: Path,
    fmt: str = ".6f",
) -> None:
    """Save a grouped bar chart with per-seed bars.

    Args:
        seeds (list[int]): Seed labels.
        values (list[float]): Per-seed metric values.
        mean (float): Mean across seeds.
        std (float): Std across seeds.
        ylabel (str): Y-axis label.
        title (str): Plot title.
        out_path (Path): Destination path.
        fmt (str): Format string for annotations.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    x = np.arange(len(seeds))
    bars = ax.bar(x, values, color=BAR_COLOR, width=0.5)

    ax.axhline(
        mean,
        color=MEAN_COLOR,
        linestyle="--",
        linewidth=1.2,
        label=f"mean = {mean:{fmt}} ± {std:{fmt}}",
    )

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:{fmt}}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def _scatter_pred_vs_target(
    preds: np.ndarray,
    targets: np.ndarray,
    seed: int,
    out_path: Path,
) -> None:
    """Save a prediction-vs-target scatter plot.

    Args:
        preds (np.ndarray): Model predictions.
        targets (np.ndarray): Ground-truth labels.
        seed (int): Seed label for the title.
        out_path (Path): Destination path.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SCATTER)
    ax.scatter(
        targets,
        preds,
        s=1,
        alpha=0.15,
        rasterized=True,
    )
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.0)
    ax.set_xlabel("Ground Truth (win prob)")
    ax.set_ylabel("Predicted (win prob)")
    ax.set_title(f"Prediction vs Target — seed {seed}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(alpha=GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def main() -> None:
    """Generate all evaluation plots."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = _load_results()

    seeds = [s for s in RUN_SEEDS.seeds if str(s) in results]
    seed_results = [results[str(s)] for s in seeds]

    mean = results["mean"]
    std = results["std"]

    # MSE bar chart.
    _bar_chart(
        seeds,
        [r["mse"] for r in seed_results],
        mean["mse"],
        std["mse"],
        ylabel="MSE",
        title="Test MSE by Seed — piece_plane",
        out_path=OUT_DIR / "mse_by_seed.png",
    )

    # MAE bar chart.
    _bar_chart(
        seeds,
        [r["mae"] for r in seed_results],
        mean["mae"],
        std["mae"],
        ylabel="MAE",
        title="Test MAE by Seed — piece_plane",
        out_path=OUT_DIR / "mae_by_seed.png",
    )

    # Correlation bar chart.
    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    x = np.arange(len(seeds))
    w = 0.3
    pearson_vals = [r["pearson"] for r in seed_results]
    spearman_vals = [r["spearman"] for r in seed_results]

    bars_p = ax.bar(
        x - w / 2,
        pearson_vals,
        w,
        label="Pearson",
        color="steelblue",
    )
    bars_s = ax.bar(
        x + w / 2,
        spearman_vals,
        w,
        label="Spearman",
        color="darkorange",
    )

    for bar, val in zip(bars_p, pearson_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    for bar, val in zip(bars_s, spearman_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Correlation")
    ax.set_title("Test Correlation by Seed — piece_plane")
    ax.legend()
    ax.grid(axis="y", alpha=GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "correlation_by_seed.png", dpi=DPI)
    plt.close(fig)

    # Sign accuracy bar chart.
    _bar_chart(
        seeds,
        [r["sign_acc"] for r in seed_results],
        mean["sign_acc"],
        std["sign_acc"],
        ylabel="Accuracy",
        title="Test Sign Accuracy by Seed — piece_plane",
        out_path=OUT_DIR / "sign_acc_by_seed.png",
        fmt=".4f",
    )

    # Scatter plot: pred vs target for best seed.
    best_seed = min(seeds, key=lambda s: results[str(s)]["mse"])
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    cfg = TrainConfig(encoder_name=ENCODER, seed=best_seed)
    run_dir = MODEL_DIR / MODEL_TAG / f"{ENCODER}_seed{best_seed}"
    encoder = PiecePlaneEncoder()
    loader = build_loader(
        "test",
        encoder=encoder,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        split_dir=TEST_SPLIT_DIR,
    )
    model = _load_model(run_dir, cfg, device)
    preds, targets = _predict(model, loader, device)
    _scatter_pred_vs_target(
        preds,
        targets,
        best_seed,
        OUT_DIR / "pred_vs_target.png",
    )

    print(f"[eval] plots saved to {OUT_DIR}/")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
