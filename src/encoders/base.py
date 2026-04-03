"""Abstract base class for board-state encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BoardEncoder(ABC):
    """Encodes a FEN string into a torch.Tensor.

    Subclasses implement ``encode`` and ``output_shape``.
    Instances are directly callable (for use as the
    ``encoder`` parameter of ``ChessDataset``).
    """

    @abstractmethod
    def encode(self, fen: str) -> torch.Tensor:
        """Convert a FEN string to a tensor.

        Args:
            fen (str): Full FEN string (6 fields).

        Returns:
            torch.Tensor.
        """

    @property
    @abstractmethod
    def output_shape(self) -> tuple[int, ...]:
        """Shape of the tensor returned by ``encode``."""

    def __call__(self, fen: str) -> torch.Tensor:
        return self.encode(fen)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(output_shape={self.output_shape})"
        )
