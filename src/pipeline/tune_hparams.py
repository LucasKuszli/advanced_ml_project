"""Hyperparameter grid search over LR × dropout.

Samples a fresh 10M-position dataset (seed ≠ production)
and runs short training trials for each encoder.  Results
are saved to ``artifacts/tune/results.json``.

Usage:
    uv run python -m src.pipeline.tune_hparams
    uv run python -m src.pipeline.tune_hparams --encoder piece_plane
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from src.config.paths import ARTIFACTS_DIR, DATA_DIR
from src.config.train import RUN_SEEDS, TUNE_GRID, TrainConfig
from src.data.loader import (
    TEST_SPLIT_DIR,
    collect_split_keys,
    prepare_splits,
    prepare_test_split,
)
from src.train.trainer import train

TUNE_SPLIT_DIR = DATA_DIR / "tune_splits"
TUNE_OUT_DIR = ARTIFACTS_DIR / "tune"

ENCODERS = ("piece_plane", "square_token")


def _ensure_tune_data() -> None:
    """Sample tune splits if they don't exist yet.

    Collects dedup keys from the shared test set and all
    production seed splits so the tuning data is guaranteed
    disjoint.
    """
    marker = TUNE_SPLIT_DIR / "train.csv"
    if marker.exists():
        print(
            f"[tune] Tune splits already exist at {TUNE_SPLIT_DIR}, skipping re-sample."
        )
        return

    # Shared test set.
    prepare_test_split()
    exclude: set[str] = collect_split_keys(TEST_SPLIT_DIR)
    print(f"[tune] Collected {len(exclude):,} test keys")

    # Gather keys from every production seed split.
    for seed in RUN_SEEDS.seeds:
        seed_dir = DATA_DIR / "splits" / f"seed{seed}"
        if seed_dir.exists():
            keys = collect_split_keys(seed_dir)
            print(f"[tune] Collected {len(keys):,} keys from seed{seed}")
            exclude |= keys

    print(f"[tune] Excluding {len(exclude):,} total keys from tune data")

    print(f"[tune] Sampling tune data (seed={TUNE_GRID.data_seed}) …")
    prepare_splits(
        seed=TUNE_GRID.data_seed,
        out_dir=TUNE_SPLIT_DIR,
        exclude_keys=exclude,
    )


def _run_grid(encoder_name: str) -> list[dict]:
    """Run every (lr, dropout) combo for one encoder.

    Args:
        encoder_name (str): Encoder to tune.

    Returns:
        list[dict]: One result dict per trial, sorted by
        ascending val loss.
    """
    combos = list(
        itertools.product(
            TUNE_GRID.lr,
            TUNE_GRID.dropout,
        ),
    )
    results: list[dict] = []

    for trial, (lr, dropout) in enumerate(combos, 1):
        cfg = TrainConfig(
            encoder_name=encoder_name,
            seed=TUNE_GRID.model_seed,
            epochs=TUNE_GRID.epochs,
            lr=lr,
            dropout=dropout,
        )

        tag = f"d{cfg.d_model}_L{cfg.n_layers}/{encoder_name}_lr{lr:.0e}_do{dropout:.1f}"
        print(f"\n[tune] trial {trial}/{len(combos)}  {tag}")

        trial_dir = TUNE_OUT_DIR / tag
        run_dir = train(
            cfg,
            split_dir=TUNE_SPLIT_DIR,
            out_dir=trial_dir,
        )

        meta_path = run_dir / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)

        results.append(
            {
                "encoder": encoder_name,
                "lr": lr,
                "dropout": dropout,
                "best_val_loss": meta["best_val_loss"],
                "final_train_loss": meta["final_train_loss"],
                "epochs_completed": meta["epochs_completed"],
                "elapsed_s": meta["elapsed_seconds"],
            },
        )

    results.sort(key=lambda r: r["best_val_loss"])
    return results


def main() -> None:
    """Run tuning grid and save results JSON."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning grid search.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        help=("Tune only this encoder (default=both)."),
    )
    args = parser.parse_args()

    encoders = (args.encoder,) if args.encoder else ENCODERS

    _ensure_tune_data()

    TUNE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, list[dict]] = {}

    for enc in encoders:
        results = _run_grid(enc)
        all_results[enc] = results

        print(f"\n{'=' * 50}")
        print(f"  {enc} — top 3:")
        print(f"{'=' * 50}")
        for r in results[:3]:
            print(
                f"  lr={r['lr']:.0e}  "
                f"do={r['dropout']:.1f}  "
                f"val={r['best_val_loss']:.6f}  "
                f"({r['elapsed_s']:.0f}s)"
            )

    out_path = TUNE_OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[tune] Results saved to {out_path}")


if __name__ == "__main__":
    main()
