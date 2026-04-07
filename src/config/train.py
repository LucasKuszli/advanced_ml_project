"""Configuration for training pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from src.config.model import TRANSFORMER


@dataclass
class TrainConfig:
    """All hyperparameters for a training run.

    Model-architecture defaults are drawn from
    ``TransformerDefaults`` so there is a single source of
    truth.

    Args:
        encoder_name (str): ``"piece_plane"``,
            ``"dynamic_piece_plane"``, or
            ``"full_piece_plane"``.
        seed (int): Random seed for reproducibility.
        epochs (int): Number of full passes over train set.
        batch_size (int): Mini-batch size.
        lr (float): Peak learning rate.
        weight_decay (float): AdamW weight decay.
        d_model (int): Transformer embedding dim.
        n_heads (int): Attention heads.
        n_layers (int): Transformer encoder layers.
        d_ff (int | None): FF hidden dim (default=4*d).
        dropout (float): Dropout rate.
        num_workers (int): DataLoader workers.
        amp (bool): Use automatic mixed precision.
        grad_clip (float): Max gradient norm (0=off).
        val_every (int): Validate every N epochs.
        patience (int): Early-stopping patience in
            validation epochs without improvement (0=off).
    """

    encoder_name: str = "piece_plane"
    seed: int = 42
    epochs: int = 40
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-2
    d_model: int = TRANSFORMER.d_model
    n_heads: int = TRANSFORMER.n_heads
    n_layers: int = TRANSFORMER.n_layers
    d_ff: int | None = None
    dropout: float = TRANSFORMER.dropout
    num_workers: int = 24
    amp: bool = True
    grad_clip: float = 1.0
    val_every: int = 1
    patience: int = 10


@dataclass(frozen=True)
class RunSeeds:
    """Seeds for multi-seed training pipelines.

    Each seed is used for both the data split and model
    initialisation so every run is fully independent.
    """

    seeds: tuple[int, ...] = (42, 67, 1337)


@dataclass(frozen=True)
class TuneGrid:
    """Search space for hyperparameter tuning runs."""

    # Learning rates to sweep.
    lr: tuple[float, ...] = (1e-2, 1e-3, 1e-4)

    # Dropout values to sweep.
    dropout: tuple[float, ...] = (0.0, 0.1, 0.2)

    # Short training epochs per trial.
    epochs: int = 5

    # Random seed for data sampling (different from
    # production seed=42 to avoid tuning on train data).
    data_seed: int = 123

    # Random seed for model initialisation.
    model_seed: int = 123


@dataclass(frozen=True)
class PlotLayout:
    """Layout constants for training-curve plots."""

    # Figure size (width, height) in inches.
    figsize: tuple[float, float] = (8.0, 6.0)

    # GridSpec height ratio (loss panel : LR panel).
    height_ratios: tuple[int, int] = (3, 1)

    # Saved image resolution.
    dpi: int = 150

    # Line width for loss and LR curves.
    linewidth: float = 1.5

    # Grid transparency.
    grid_alpha: float = 0.3

    # Colour for the learning-rate curve.
    lr_color: str = "tab:green"


# Module-level singletons.
RUN_SEEDS = RunSeeds()
TUNE_GRID = TuneGrid()
PLOT_LAYOUT = PlotLayout()
