"""Evaluate trained piece-plane models on the test set.

Loads each seed's best checkpoint, runs inference on the
held-out test split, computes metrics, and prints a
summary table.

Usage:
    uv run python -m src.evaluate.evaluate
    uv run python -m src.evaluate.evaluate --seed 42 67
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config.paths import EVAL_DIR, MODEL_DIR
from src.config.train import RUN_SEEDS, TrainConfig
from src.data.dataset import build_loader
from src.encoders import (
    DynamicPiecePlaneEncoder,
    FullPiecePlaneEncoder,
    PiecePlaneEncoder,
)
from src.evaluate.metrics import compute_all
from src.model.model import build_model

TEST_SPLIT_DIR = Path("data/test_split")
MODEL_TAG = "d128_L5"


def _load_model(
    run_dir: Path,
    cfg: TrainConfig,
    device: torch.device,
) -> torch.nn.Module:
    """Load best checkpoint into a fresh model.

    Args:
        run_dir (Path): Directory with ``best.pt``.
        cfg (TrainConfig): Model configuration.
        device (torch.device): Target device.

    Returns:
        nn.Module in eval mode.
    """
    model = build_model(
        cfg.encoder_name,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)

    ckpt_path = run_dir / "best.pt"
    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


@torch.no_grad()
def _predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model on loader, return predictions and labels.

    Args:
        model (nn.Module): Model in eval mode.
        loader (DataLoader): Test data loader.
        device (torch.device): Target device.

    Returns:
        Tuple of (predictions, targets) as numpy arrays.
    """
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        pred = model(x)
        preds.append(pred.cpu().numpy())
        targets.append(y.numpy())

    return np.concatenate(preds), np.concatenate(targets)


def evaluate_seed(
    seed: int,
    encoder_name: str,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a single seed on the test set.

    Args:
        seed (int): Training seed.
        encoder_name (str): Encoder name.
        device (torch.device): Target device.

    Returns:
        dict of metric name → value.
    """
    cfg = TrainConfig(encoder_name=encoder_name, seed=seed)
    run_dir = MODEL_DIR / MODEL_TAG / f"{encoder_name}_seed{seed}"

    if not (run_dir / "best.pt").exists():
        raise FileNotFoundError(f"No checkpoint at {run_dir / 'best.pt'}")

    encoder_map = {
        "piece_plane": PiecePlaneEncoder,
        "dynamic_piece_plane": DynamicPiecePlaneEncoder,
        "full_piece_plane": FullPiecePlaneEncoder,
    }
    encoder = encoder_map[encoder_name]()
    loader = build_loader(
        "test",
        encoder=encoder,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        split_dir=TEST_SPLIT_DIR,
    )

    model = _load_model(run_dir, cfg, device)
    preds, targets = _predict(model, loader, device)
    return compute_all(preds, targets)


def main() -> None:
    """Evaluate piece-plane models and print results."""
    parser = argparse.ArgumentParser(
        description="Evaluate piece-plane models on test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=list(RUN_SEEDS.seeds),
        help="Seeds to evaluate (default: all).",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="piece_plane",
        help="Encoder name (default=piece_plane).",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu",
    )

    results: dict[int, dict[str, float]] = {}
    for seed in args.seed:
        print(f"[eval] evaluating seed={seed} …")
        metrics = evaluate_seed(seed, args.encoder, device)
        results[seed] = metrics

    # Print table.
    header = (
        f"{'seed':>6s}  {'MSE':>10s}  {'MAE':>10s}  "
        f"{'Pearson':>10s}  {'Spearman':>10s}  "
        f"{'SignAcc':>10s}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    all_metrics: dict[str, list[float]] = {
        k: [] for k in ("mse", "mae", "pearson", "spearman", "sign_acc")
    }
    for seed in args.seed:
        m = results[seed]
        for k, v in m.items():
            all_metrics[k].append(v)
        print(
            f"{seed:>6d}  {m['mse']:>10.6f}  {m['mae']:>10.6f}  "
            f"{m['pearson']:>10.6f}  {m['spearman']:>10.6f}  "
            f"{m['sign_acc']:>10.4f}"
        )

    # Mean ± std across seeds.
    if len(args.seed) > 1:
        print("-" * len(header))
        means = {k: np.mean(v) for k, v in all_metrics.items()}
        stds = {k: np.std(v) for k, v in all_metrics.items()}
        print(
            f"{'mean':>6s}  "
            f"{means['mse']:>10.6f}  "
            f"{means['mae']:>10.6f}  "
            f"{means['pearson']:>10.6f}  "
            f"{means['spearman']:>10.6f}  "
            f"{means['sign_acc']:>10.4f}"
        )
        print(
            f"{'±std':>6s}  "
            f"{stds['mse']:>10.6f}  "
            f"{stds['mae']:>10.6f}  "
            f"{stds['pearson']:>10.6f}  "
            f"{stds['spearman']:>10.6f}  "
            f"{stds['sign_acc']:>10.4f}"
        )

    # Save results JSON.
    out_dir = EVAL_DIR / MODEL_TAG / args.encoder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    serializable = {str(k): v for k, v in results.items()}
    if len(args.seed) > 1:
        serializable["mean"] = {k: float(np.mean(v)) for k, v in all_metrics.items()}
        serializable["std"] = {k: float(np.std(v)) for k, v in all_metrics.items()}
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n[eval] results saved to {out_path}")


if __name__ == "__main__":
    main()
