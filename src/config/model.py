"""Configuration for transformer model architecture."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerDefaults:
    """Default architecture hyper-parameters for
    ChessTransformer.

    Individual runs may override these via TrainConfig;
    the values here serve as the single source of truth
    for function signatures and CLI defaults.
    """

    # Embedding dimension.
    d_model: int = 128

    # Number of attention heads.
    n_heads: int = 4

    # Number of transformer encoder layers.
    n_layers: int = 5

    # Feed-forward expansion factor (d_ff = factor × d).
    ff_factor: int = 4

    # Dropout rate.
    dropout: float = 0.05

    # Activation function in feed-forward layers.
    activation: str = "gelu"

    # Use pre-layer-norm ordering.
    norm_first: bool = True

    # Std-dev for positional embedding initialisation.
    pos_embed_init_scale: float = 0.02


# Module-level singleton.
TRANSFORMER = TransformerDefaults()
