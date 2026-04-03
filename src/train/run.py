"""CLI entrypoint for training.

Usage:
    uv run python -m src.train.run --encoder piece_plane --seed 42
    uv run python -m src.train.run --encoder square_token --seed 0 --epochs 20
"""

from __future__ import annotations

import argparse

from src.config.train import TrainConfig
from src.train.trainer import train

# Reference instance for argparse defaults.
_DEFAULTS = TrainConfig()


def build_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace.
    """
    p = argparse.ArgumentParser(
        description="Train a ChessTransformer run.",
    )
    p.add_argument(
        "--encoder",
        type=str,
        default=_DEFAULTS.encoder_name,
        help=(
            "Encoder name: piece_plane | square_token "
            f"(default={_DEFAULTS.encoder_name})."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=_DEFAULTS.seed,
        help=f"Random seed (default={_DEFAULTS.seed}).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=_DEFAULTS.epochs,
        help=(f"Training epochs (default={_DEFAULTS.epochs})."),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULTS.batch_size,
        help=(f"Batch size (default={_DEFAULTS.batch_size})."),
    )
    p.add_argument(
        "--lr",
        type=float,
        default=_DEFAULTS.lr,
        help=(f"Peak learning rate (default={_DEFAULTS.lr})."),
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=_DEFAULTS.weight_decay,
        help=(f"AdamW weight decay (default={_DEFAULTS.weight_decay})."),
    )
    p.add_argument(
        "--d-model",
        type=int,
        default=_DEFAULTS.d_model,
        help=(f"Embedding dimension (default={_DEFAULTS.d_model})."),
    )
    p.add_argument(
        "--n-heads",
        type=int,
        default=_DEFAULTS.n_heads,
        help=(f"Number of attention heads (default={_DEFAULTS.n_heads})."),
    )
    p.add_argument(
        "--n-layers",
        type=int,
        default=_DEFAULTS.n_layers,
        help=(f"Transformer layers (default={_DEFAULTS.n_layers})."),
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=_DEFAULTS.dropout,
        help=(f"Dropout rate (default={_DEFAULTS.dropout})."),
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=_DEFAULTS.num_workers,
        help=(f"DataLoader workers (default={_DEFAULTS.num_workers})."),
    )
    p.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision.",
    )
    p.add_argument(
        "--grad-clip",
        type=float,
        default=_DEFAULTS.grad_clip,
        help=(f"Max gradient norm, 0=off (default={_DEFAULTS.grad_clip})."),
    )
    p.add_argument(
        "--val-every",
        type=int,
        default=_DEFAULTS.val_every,
        help=(f"Validate every N epochs (default={_DEFAULTS.val_every})."),
    )
    p.add_argument(
        "--patience",
        type=int,
        default=_DEFAULTS.patience,
        help=(
            "Early-stopping patience in val epochs, "
            f"0=off (default={_DEFAULTS.patience})."
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint.",
    )
    return p.parse_args()


def main() -> None:
    """Build config from CLI args and launch training."""
    args = build_args()
    cfg = TrainConfig(
        encoder_name=args.encoder,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        grad_clip=args.grad_clip,
        val_every=args.val_every,
        patience=args.patience,
    )
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
