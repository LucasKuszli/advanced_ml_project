"""Configuration for the ChessBench data pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChessBenchUrls:
    """GCS bucket URLs for ChessBench state-value data."""

    # Google Cloud Storage base URL for the dataset.
    base: str = (
        "https://storage.googleapis.com/"
        "searchless_chess/data"
    )

    @property
    def train(self) -> str:
        """Training state-value .bag file URL."""
        return f"{self.base}/train/state_value_data.bag"

    @property
    def test(self) -> str:
        """Test state-value .bag file URL."""
        return f"{self.base}/test/state_value_data.bag"


@dataclass(frozen=True)
class SplitConfig:
    """Train/val/test split fractions (must sum to 1)."""

    # Fraction of sampled data used for training.
    train: float = 0.8
    # Fraction of sampled data used for validation.
    val: float = 0.1
    # Fraction of sampled data used for testing.
    test: float = 0.1


@dataclass(frozen=True)
class SamplingDefaults:
    """Default parameters for data sampling."""

    # Number of positions to sample from the .bag file.
    num_samples: int = 10_000_000
    # Random seed for reproducible sampling.
    seed: int = 42


@dataclass(frozen=True)
class BagFormat:
    """Apache Beam .bag binary format constants."""

    # Bytes per index entry (each entry is an int64 offset).
    index_entry_bytes: int = 8
    # Bytes for the win-probability value (float64).
    win_prob_bytes: int = 8
    # Minimum valid .bag file size (needs trailing offset).
    min_file_bytes: int = 8


@dataclass(frozen=True)
class DownloadConfig:
    """Download behaviour settings."""

    # Read chunk size for streaming downloads (8 MB).
    chunk_bytes: int = 8 * 1024 * 1024
    # Print progress every N decoded records.
    progress_interval: int = 500_000


# Module-level singletons.
CHESSBENCH_URLS = ChessBenchUrls()
SPLIT_CFG = SplitConfig()
SAMPLING = SamplingDefaults()
BAG_FMT = BagFormat()
DOWNLOAD_CFG = DownloadConfig()
