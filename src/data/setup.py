"""Prepare all disjoint data splits for training and tuning.

Downloads the ChessBench .bag files (if missing) and then
creates one shared test split plus 3 disjoint seed splits
and 1 tuning split, each with 10M positions and zero
overlap.

Usage:
    uv run python -m src.data.setup
"""

from __future__ import annotations

from src.config.paths import DATA_DIR
from src.config.train import RUN_SEEDS, TUNE_GRID
from src.data.loader import (
    TEST_SPLIT_DIR,
    collect_split_keys,
    download_chessbench,
    prepare_splits,
    prepare_test_split,
)

TUNE_SPLIT_DIR = DATA_DIR / "tune_splits"


def main() -> None:
    """Download data and create all disjoint splits."""
    download_chessbench()

    # --- Shared test set (sampled once, reused by all seeds) ---
    prepare_test_split()
    test_keys = collect_split_keys(TEST_SPLIT_DIR)
    print(f"[setup] Shared test: {len(test_keys):,} keys")

    exclude: set[str] = set(test_keys)

    # --- Production seed splits (train/val only) ---
    for seed in RUN_SEEDS.seeds:
        split_dir = DATA_DIR / "splits" / f"seed{seed}"
        if (split_dir / "train.csv").exists():
            print(f"[setup] seed{seed} already exists, collecting keys only")
        else:
            print(f"\n[setup] Preparing seed{seed} (excluding {len(exclude):,} keys)")
            prepare_splits(
                seed=seed,
                out_dir=split_dir,
                exclude_keys=exclude,
            )
        exclude |= collect_split_keys(split_dir)

    # --- Tuning split ---
    if (TUNE_SPLIT_DIR / "train.csv").exists():
        print(f"\n[setup] Tune splits already exist at {TUNE_SPLIT_DIR}")
    else:
        print(f"\n[setup] Preparing tune split (excluding {len(exclude):,} keys)")
        prepare_splits(
            seed=TUNE_GRID.data_seed,
            out_dir=TUNE_SPLIT_DIR,
            exclude_keys=exclude,
        )

    print("\n[setup] All splits ready.")


if __name__ == "__main__":
    main()
