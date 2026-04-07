"""Train piece-plane encoder models over multiple seeds.

Each seed gets its own data split and model initialisation
so that runs are fully independent.

Usage:
    uv run python -m src.pipeline.train_piece_plane
    uv run python -m src.pipeline.train_piece_plane --encoder dynamic_piece_plane
    uv run python -m src.pipeline.train_piece_plane --encoder full_piece_plane --seed 42 67
"""

from __future__ import annotations

import argparse

from src.config.paths import DATA_DIR
from src.config.train import RUN_SEEDS, TrainConfig
from src.data.loader import (
    TEST_SPLIT_DIR,
    collect_split_keys,
    prepare_splits,
    prepare_test_split,
)
from src.data.precompute import precompute_split
from src.train.trainer import train

ENCODER_CHOICES = ("piece_plane", "dynamic_piece_plane", "full_piece_plane")


def main() -> None:
    """Run training for a given encoder across seeds."""
    parser = argparse.ArgumentParser(
        description="Train piece-plane encoder models.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        choices=ENCODER_CHOICES,
        help="Encoder to train.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=list(RUN_SEEDS.seeds),
        help="Seeds to train (default: all RUN_SEEDS).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint.",
    )
    args = parser.parse_args()

    prepare_test_split()
    exclude: set[str] = collect_split_keys(TEST_SPLIT_DIR)

    for seed in args.seed:
        split_dir = DATA_DIR / "splits" / f"seed{seed}"

        if not (split_dir / "train.csv").exists():
            print(f"\n[pipeline] Preparing splits (seed={seed}) …")
            prepare_splits(
                seed=seed,
                out_dir=split_dir,
                exclude_keys=exclude,
            )

        # Accumulate keys so the next seed is disjoint.
        exclude |= collect_split_keys(split_dir)

        # Precompute encoded tensors to disk (skips if exists).
        for split in ("train", "val"):
            precompute_split(args.encoder, split, split_dir)

        cfg = TrainConfig(
            encoder_name=args.encoder,
            seed=seed,
        )
        print(f"\n[pipeline] {args.encoder}  seed={seed}")
        train(cfg, split_dir=split_dir, resume=args.resume)


if __name__ == "__main__":
    main()
