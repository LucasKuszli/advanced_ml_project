"""ChessBench data download, sampling, and splitting.

Downloads state-value .bag files from the ChessBench GCS
bucket, samples a subset, and splits into train/val/test CSV
files ready for model training.

Usage:
    uv run python -m src.data.loader
    uv run python -m src.data.loader --num-samples 1000000
    uv run python -m src.data.loader --skip-download
"""

from __future__ import annotations

import csv
import mmap
import os
import struct
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np

from src.config.data import (
    BAG_FMT,
    CHESSBENCH_URLS,
    DOWNLOAD_CFG,
    SAMPLING,
    SPLIT_CFG,
)
from src.config.paths import DATA_DIR

RAW_DIR = DATA_DIR / "raw"
SPLIT_DIR = DATA_DIR / "splits"


class BagReader:
    """Minimal reader for uncompressed .bag files.

    Args:
        bag_path (str | Path): Path to a .bag file.
    """

    def __init__(self, bag_path: str | Path) -> None:
        bag_path = str(bag_path)
        fd = os.open(bag_path, os.O_RDONLY)
        try:
            self._data = mmap.mmap(
                fd, 0, access=mmap.ACCESS_READ,
            )
            file_size = self._data.size()
        finally:
            os.close(fd)

        entry = BAG_FMT.index_entry_bytes
        if file_size < entry:
            raise ValueError(f"Bag file too small: {bag_path}")

        # Trailing bytes store the offset where the
        # record index starts.
        (index_start,) = struct.unpack(
            "<Q", self._data[-entry:],
        )
        index_size = file_size - index_start
        self._num_records = index_size // entry
        self._index_start = index_start

    def __len__(self) -> int:
        return self._num_records

    def __getitem__(self, idx: int) -> bytes:
        """Return raw bytes for record at *idx*.

        Args:
            idx (int): Record index.

        Returns:
            bytes.
        """
        if idx < 0:
            idx += self._num_records
        if not 0 <= idx < self._num_records:
            raise IndexError(
                f"Index {idx} out of range [0, "
                f"{self._num_records})"
            )
        entry = BAG_FMT.index_entry_bytes
        offset = idx * entry + self._index_start
        if idx:
            start, end = struct.unpack(
                "<2q",
                self._data[offset - entry:offset + entry],
            )
        else:
            (end,) = struct.unpack(
                "<q",
                self._data[offset:offset + entry],
            )
            start = 0
        return bytes(self._data[start:end])

    def close(self) -> None:
        """Release the memory-mapped file."""
        self._data.close()


def _decode_varint(buf: BytesIO) -> int:
    """Decode an Apache Beam variable-length integer."""
    shift = 0
    result = 0
    while True:
        raw = buf.read(1)
        if not raw:
            raise ValueError("Unexpected end of varint")
        b = raw[0]
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result


def decode_state_value(
    record: bytes,
) -> tuple[str, float]:
    """Decode a state-value record into (FEN, win_prob).

    The .bag record uses Apache Beam TupleCoder with
    StrUtf8Coder (FEN) and FloatCoder (win probability).
    Each component is prefixed by a varint byte-length.

    Args:
        record (bytes): Raw .bag record bytes.

    Returns:
        tuple[str, float].
    """
    buf = BytesIO(record)
    # FEN: varint length + UTF-8 bytes.
    fen_len = _decode_varint(buf)
    fen = buf.read(fen_len).decode("utf-8")
    # Win prob: raw big-endian double (no length prefix —
    # last element in Apache Beam TupleCoder).
    (win_prob,) = struct.unpack(
        ">d", buf.read(BAG_FMT.win_prob_bytes),
    )
    return fen, win_prob


def download_file(url: str, dest: Path) -> None:
    """Download a file from *url* to *dest* with progress.

    Args:
        url (str): Source URL.
        dest (Path): Local destination path.
    """
    if dest.exists():
        print(f"Already exists: {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    print(f"Downloading {url}")
    print(f"  -> {dest}")

    req = urllib.request.urlopen(url)  # noqa: S310
    total = int(req.headers.get("Content-Length", 0))
    downloaded = 0
    chunk_size = DOWNLOAD_CFG.chunk_bytes

    with open(tmp, "wb") as f:
        while True:
            chunk = req.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                mb_done = downloaded / 1e6
                mb_total = total / 1e6
                print(
                    f"\r  {mb_done:.0f}/"
                    f"{mb_total:.0f} MB"
                    f" ({pct:.1f}%)",
                    end="",
                    flush=True,
                )
    print()
    tmp.rename(dest)
    print(f"Done: {dest}")


def download_chessbench() -> tuple[Path, Path]:
    """Download ChessBench state-value .bag files.

    Returns:
        tuple[Path, Path].
    """
    train_path = RAW_DIR / "train_state_value.bag"
    test_path = RAW_DIR / "test_state_value.bag"
    download_file(CHESSBENCH_URLS.train, train_path)
    download_file(CHESSBENCH_URLS.test, test_path)
    return train_path, test_path


def _dedup_key(fen: str) -> str:
    """Extract the first 5 FEN fields as a dedup key.

    Keeps board placement, active colour, castling rights,
    en-passant square, and halfmove clock. The halfmove
    clock matters because proximity to the 50-move draw
    rule changes evaluation pressure. Only the fullmove
    number is stripped.

    Args:
        fen (str): Full FEN string.

    Returns:
        str.
    """
    return " ".join(fen.split()[:5])


def _sample_bag(
    bag_path: Path,
    num_samples: int,
    seed: int,
) -> list[tuple[str, float]]:
    """Sample and deduplicate records from a .bag file.

    Deduplicates on the first 4 FEN fields (board, turn,
    castling, en passant) and averages the win probability
    for collisions.

    Args:
        bag_path (Path): Path to the .bag file.
        num_samples (int): Maximum positions to sample.
        seed (int): Random seed.

    Returns:
        list[tuple[str, float]].
    """
    reader = BagReader(bag_path)
    total = len(reader)
    num_samples = min(num_samples, total)
    print(
        f"  {bag_path.name}: {total:,} records, "
        f"sampling {num_samples:,} (seed={seed})"
    )

    rng = np.random.default_rng(seed)
    indices = rng.choice(
        total, size=num_samples, replace=False,
    )
    indices.sort()

    # Accumulate (sum, count) per dedup key; keep full FEN
    # from the first occurrence.
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    fen_for_key: dict[str, str] = {}

    for pos, idx in enumerate(indices):
        fen, win_prob = decode_state_value(
            reader[int(idx)],
        )
        key = _dedup_key(fen)
        if key in sums:
            sums[key] += win_prob
            counts[key] += 1
        else:
            sums[key] = win_prob
            counts[key] = 1
            fen_for_key[key] = fen

        interval = DOWNLOAD_CFG.progress_interval
        done = pos + 1
        if done % interval == 0 or done == num_samples:
            print(
                f"\r    Decoded {done:,}/{num_samples:,}",
                end="",
                flush=True,
            )

    print()
    reader.close()

    rows = [
        (fen_for_key[k], sums[k] / counts[k])
        for k in sums
    ]
    n_dupes = num_samples - len(rows)
    if n_dupes:
        print(
            f"    Deduplicated: {num_samples:,} → "
            f"{len(rows):,} ({n_dupes:,} duplicates)"
        )
    return rows


def _write_csv(
    path: Path,
    rows: list[tuple[str, float]],
) -> None:
    """Write (fen, win_prob) rows to a CSV file.

    Args:
        path (Path): Output path.
        rows (list[tuple[str, float]]): Data rows.
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fen", "win_prob"])
        for fen, wp in rows:
            w.writerow([fen, wp])


def prepare_splits(
    num_train_samples: int = SAMPLING.num_samples,
    num_test_samples: int = SAMPLING.num_samples,
    seed: int = SAMPLING.seed,
) -> None:
    """Sample both bags and write train/val/test CSVs.

    Uses the paper's pre-split train and test .bag files.
    The test bag is processed first so that any positions
    appearing in both bags can be removed from train/val,
    guaranteeing disjoint splits. A validation set is carved
    from the train bag according to ``SPLIT_CFG.val_frac``.

    Args:
        num_train_samples (int): Positions to sample from
            the train bag, (default=10_000_000).
        num_test_samples (int): Positions to sample from
            the test bag, (default=10_000_000).
        seed (int): Random seed, (default=42).
    """
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Test bag first (so we can remove overlaps) ---
    test_bag = RAW_DIR / "test_state_value.bag"
    print("Processing test bag:")
    test_rows = _sample_bag(
        test_bag, num_test_samples, seed,
    )
    test_keys = {_dedup_key(fen) for fen, _ in test_rows}

    test_csv = SPLIT_DIR / "test.csv"
    _write_csv(test_csv, test_rows)
    print(
        f"  test:  {len(test_rows):,} records "
        f"-> {test_csv}"
    )

    # --- Train bag → train + val CSVs ---
    train_bag = RAW_DIR / "train_state_value.bag"
    print("Processing train bag:")
    rows = _sample_bag(train_bag, num_train_samples, seed)

    # Remove positions that also appear in the test set.
    before = len(rows)
    rows = [
        r for r in rows if _dedup_key(r[0]) not in test_keys
    ]
    n_leaked = before - len(rows)
    if n_leaked:
        print(
            f"  Removed {n_leaked:,} positions "
            f"overlapping with test set"
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(rows)

    n_val = int(len(rows) * SPLIT_CFG.val_frac)
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    train_csv = SPLIT_DIR / "train.csv"
    _write_csv(train_csv, train_rows)
    print(
        f"  train: {len(train_rows):,} records "
        f"-> {train_csv}"
    )

    val_csv = SPLIT_DIR / "val.csv"
    _write_csv(val_csv, val_rows)
    print(
        f"  val:   {len(val_rows):,} records "
        f"-> {val_csv}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Download and prepare ChessBench data."
        ),
    )
    parser.add_argument(
        "--num-train-samples",
        type=int,
        default=SAMPLING.num_samples,
        help=(
            "Positions to sample from the train bag "
            "(default: 10M)."
        ),
    )
    parser.add_argument(
        "--num-test-samples",
        type=int,
        default=SAMPLING.num_samples,
        help=(
            "Positions to sample from the test bag "
            "(default: 10M, clamped to bag size)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SAMPLING.seed,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, use existing .bag files.",
    )
    args = parser.parse_args()

    if not args.skip_download:
        download_chessbench()

    prepare_splits(
        num_train_samples=args.num_train_samples,
        num_test_samples=args.num_test_samples,
        seed=args.seed,
    )
