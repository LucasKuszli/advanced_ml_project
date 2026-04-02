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


def sample_and_split(
    bag_path: Path,
    num_samples: int = SAMPLING.num_samples,
    seed: int = SAMPLING.seed,
) -> None:
    """Sample records from a .bag and write train/val/test CSVs.

    Each CSV has columns ``fen`` and ``win_prob``. Indices are
    shuffled before assignment so positions from the same game
    are unlikely to leak across splits.

    Args:
        bag_path (Path): Path to the .bag file.
        num_samples (int): Positions to sample,
            (default=10_000_000).
        seed (int): Random seed, (default=42).
    """
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Opening {bag_path}")
    reader = BagReader(bag_path)
    total = len(reader)
    print(f"Total records: {total:,}")

    num_samples = min(num_samples, total)
    print(
        f"Sampling {num_samples:,} records "
        f"(seed={seed})"
    )

    rng = np.random.default_rng(seed)
    indices = rng.choice(
        total, size=num_samples, replace=False,
    )

    # Shuffle before assigning splits.
    rng.shuffle(indices)
    n_train = int(num_samples * SPLIT_CFG.train)
    n_val = int(num_samples * SPLIT_CFG.val)

    # 0 = train, 1 = val, 2 = test.
    split_ids = np.zeros(num_samples, dtype=np.int8)
    split_ids[n_train:n_train + n_val] = 1
    split_ids[n_train + n_val:] = 2

    # Sort by bag index for sequential mmap access.
    order = indices.argsort()
    sorted_indices = indices[order]
    sorted_splits = split_ids[order]

    split_names = {0: "train", 1: "val", 2: "test"}
    files = {}
    writers = {}
    counts = {0: 0, 1: 0, 2: 0}

    for sid, name in split_names.items():
        path = SPLIT_DIR / f"{name}.csv"
        f = open(path, "w", newline="")
        w = csv.writer(f)
        w.writerow(["fen", "win_prob"])
        files[sid] = f
        writers[sid] = w

    for i in range(num_samples):
        idx = int(sorted_indices[i])
        sid = int(sorted_splits[i])
        fen, win_prob = decode_state_value(reader[idx])
        writers[sid].writerow([fen, win_prob])
        counts[sid] += 1
        interval = DOWNLOAD_CFG.progress_interval
        if (i + 1) % interval == 0 or i + 1 == num_samples:
            print(
                f"\r  Decoded {i + 1:,}/{num_samples:,}",
                end="",
                flush=True,
            )

    print()
    reader.close()

    for f in files.values():
        f.close()

    for sid, name in split_names.items():
        path = SPLIT_DIR / f"{name}.csv"
        print(f"  {name}: {counts[sid]:,} records -> {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Download and prepare ChessBench data."
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=SAMPLING.num_samples,
        help=(
            "Number of positions to sample "
            "(default: 10M)."
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

    train_bag = RAW_DIR / "train_state_value.bag"

    if not args.skip_download:
        download_chessbench()

    sample_and_split(
        train_bag,
        num_samples=args.num_samples,
        seed=args.seed,
    )
