"""Test-set evaluation metrics for chess value models."""

from __future__ import annotations

import numpy as np
from scipy import stats


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error.

    Args:
        pred (np.ndarray): Predicted values.
        target (np.ndarray): Ground-truth values.

    Returns:
        float.
    """
    return float(np.mean((pred - target) ** 2))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute error.

    Args:
        pred (np.ndarray): Predicted values.
        target (np.ndarray): Ground-truth values.

    Returns:
        float.
    """
    return float(np.mean(np.abs(pred - target)))


def pearson_r(
    pred: np.ndarray,
    target: np.ndarray,
) -> float:
    """Pearson correlation coefficient.

    Args:
        pred (np.ndarray): Predicted values.
        target (np.ndarray): Ground-truth values.

    Returns:
        float.
    """
    r, _ = stats.pearsonr(pred, target)
    return float(r)


def spearman_r(
    pred: np.ndarray,
    target: np.ndarray,
) -> float:
    """Spearman rank correlation coefficient.

    Args:
        pred (np.ndarray): Predicted values.
        target (np.ndarray): Ground-truth values.

    Returns:
        float.
    """
    r, _ = stats.spearmanr(pred, target)
    return float(r)


def sign_accuracy(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Accuracy of predicting which side is winning.

    A position with win_prob > *threshold* is counted as
    "White winning"; ≤ *threshold* as "Black winning or
    draw".

    Args:
        pred (np.ndarray): Predicted win probabilities.
        target (np.ndarray): Ground-truth win probs.
        threshold (float): Decision boundary (default=0.5).

    Returns:
        float: Fraction of matching sign predictions.
    """
    pred_sign = pred > threshold
    target_sign = target > threshold
    return float(np.mean(pred_sign == target_sign))


def compute_all(
    pred: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        pred (np.ndarray): Predicted win probabilities.
        target (np.ndarray): Ground-truth win probs.

    Returns:
        dict with keys ``mse``, ``mae``, ``pearson``,
        ``spearman``, ``sign_acc``.
    """
    return {
        "mse": mse(pred, target),
        "mae": mae(pred, target),
        "pearson": pearson_r(pred, target),
        "spearman": spearman_r(pred, target),
        "sign_acc": sign_accuracy(pred, target),
    }
