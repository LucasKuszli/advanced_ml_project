"""Train square-token encoder models over multiple seeds.

Each seed gets its own data split and model initialisation
so that runs are fully independent.

Usage:
    uv run python -m src.pipeline.train_square_token
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
from src.train.trainer import train


def main() -> None:
    """Run square-token training for each seed."""
    parser = argparse.ArgumentParser(
        description="Train square-token encoder models.",
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

        cfg = TrainConfig(
            encoder_name="square_token",
            seed=seed,
        )
        print(f"\n[pipeline] square_token  seed={seed}")
        train(cfg, split_dir=split_dir, resume=args.resume)


if __name__ == "__main__":
    main()
