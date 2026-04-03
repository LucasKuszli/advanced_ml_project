"""PyTorch Dataset and DataLoader factory for ChessBench
split CSVs.

Usage:
    uv run python -m src.data.dataset
    uv run python -m src.data.dataset --split val
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset

from src.config.data import LOADER_CFG
from src.config.paths import DATA_DIR

SPLIT_DIR = DATA_DIR / "splits"


class ChessDataset(Dataset):
    """Lazy-encoding dataset backed by a split CSV.

    Each sample is a ``(tensor, win_prob)`` pair where
    *tensor* is produced by the supplied *encoder* callable
    and *win_prob* is a scalar ``torch.float32`` label.

    Args:
        split (str): One of ``"train"``, ``"val"``, or
            ``"test"``.
        encoder (Callable[[str], torch.Tensor] | None):
            Function mapping a FEN string to a tensor.
            If ``None``, raw FEN strings are returned
            instead.
        split_dir (Path | None): Directory containing the
            split CSVs.  Defaults to ``SPLIT_DIR``.
    """

    def __init__(
        self,
        split: str,
        encoder: Callable[[str], torch.Tensor] | None = None,
        split_dir: Path | None = None,
    ) -> None:
        root = split_dir if split_dir is not None else SPLIT_DIR
        path = root / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Split CSV not found: {path}. Run `python -m src.data.loader` first."
            )

        self._fens: list[str] = []
        self._labels: list[float] = []

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._fens.append(row["fen"])
                self._labels.append(
                    float(row["win_prob"]),
                )

        self._encoder = encoder

    def __len__(self) -> int:
        return len(self._fens)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor | str, torch.Tensor]:
        fen = self._fens[idx]
        label = torch.tensor(
            self._labels[idx],
            dtype=torch.float32,
        )
        if self._encoder is not None:
            return self._encoder(fen), label
        return fen, label


def build_loader(
    split: str,
    encoder: Callable[[str], torch.Tensor] | None = None,
    batch_size: int = LOADER_CFG.batch_size,
    num_workers: int = LOADER_CFG.num_workers,
    pin_memory: bool = LOADER_CFG.pin_memory,
    split_dir: Path | None = None,
) -> DataLoader:
    """Build a DataLoader for the given split.

    Args:
        split (str): One of ``"train"``, ``"val"``, or
            ``"test"``.
        encoder (Callable[[str], torch.Tensor] | None):
            FEN-to-tensor encoder, (default=None).
        batch_size (int): Batch size, (default=256).
        num_workers (int): Loader workers, (default=4).
        pin_memory (bool): Pin host memory, (default=True).
        split_dir (Path | None): Custom split directory,
            (default=None → ``SPLIT_DIR``).

    Returns:
        DataLoader.
    """
    dataset = ChessDataset(
        split,
        encoder=encoder,
        split_dir=split_dir,
    )
    shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke-test the ChessDataset.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Split to load (default: train).",
    )
    args = parser.parse_args()

    ds = ChessDataset(args.split)
    print(f"Split: {args.split}")
    print(f"Size:  {len(ds):,}")

    fen, label = ds[0]
    print(f"Sample 0:")
    print(f"  FEN:      {fen}")
    print(f"  win_prob: {label.item():.4f}")

    loader = build_loader(
        args.split,
        batch_size=8,
        num_workers=0,
    )
    batch_fens, batch_labels = next(iter(loader))
    print(f"\nBatch (size={len(batch_labels)}):")
    print(f"  Labels: {batch_labels.tolist()}")
