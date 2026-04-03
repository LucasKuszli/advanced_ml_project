"""Training loop for chess evaluation transformer.

Handles a single training run for one (encoder, seed)
pair.  Produces a run directory under ``artifacts/models/``
with checkpoints, loss curves, and metadata JSON.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from src.config.paths import MODEL_DIR
from src.config.train import PLOT_LAYOUT, TrainConfig
from src.data.dataset import build_loader
from src.encoders import PiecePlaneEncoder, SquareTokenEncoder
from src.model.model import build_model


def _get_encoder(name: str):
    """Return the encoder callable for *name*."""
    encoders = {
        "piece_plane": PiecePlaneEncoder,
        "square_token": SquareTokenEncoder,
    }
    if name not in encoders:
        raise ValueError(f"Unknown encoder {name!r}. Choose from {list(encoders)}.")
    return encoders[name]()


def _model_tag(cfg: TrainConfig) -> str:
    """Short label encoding model size, e.g. ``d128_L5``."""
    return f"d{cfg.d_model}_L{cfg.n_layers}"


def _run_dir(cfg: TrainConfig) -> Path:
    """Build ``artifacts/models/d<D>_L<L>/<encoder>_seed<N>/``."""
    name = f"{cfg.encoder_name}_seed{cfg.seed}"
    return MODEL_DIR / _model_tag(cfg) / name


def _seed_everything(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute mean MSE over a full DataLoader.

    Args:
        model (nn.Module): The model in eval mode.
        loader (DataLoader): Val or test loader.
        device (torch.device): Target device.

    Returns:
        float: Mean MSE.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        total_loss += nn.functional.mse_loss(pred, y, reduction="sum").item()
        total_samples += y.size(0)
    return total_loss / max(total_samples, 1)


def _plot_curves(
    history: dict,
    out_path: Path,
) -> None:
    """Save a two-panel loss + LR plot.

    Args:
        history (dict): Training history with ``train_loss``
            ``val_loss`` and ``lr`` lists.
        out_path (Path): Destination PNG path.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_lr) = plt.subplots(
        2,
        1,
        figsize=PLOT_LAYOUT.figsize,
        sharex=True,
        gridspec_kw={
            "height_ratios": list(PLOT_LAYOUT.height_ratios),
        },
    )

    ax_loss.plot(
        epochs,
        history["train_loss"],
        label="train",
        linewidth=PLOT_LAYOUT.linewidth,
    )
    if history["val_loss"]:
        val_epochs = [e for e in epochs if e <= len(history["val_loss"])]
        ax_loss.plot(
            val_epochs,
            history["val_loss"][: len(val_epochs)],
            label="val",
            linewidth=PLOT_LAYOUT.linewidth,
        )
    ax_loss.set_ylabel("MSE Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=PLOT_LAYOUT.grid_alpha)
    ax_loss.set_title("Training Curves")

    ax_lr.plot(
        epochs,
        history["lr"],
        color=PLOT_LAYOUT.lr_color,
        linewidth=PLOT_LAYOUT.linewidth,
    )
    ax_lr.set_ylabel("Learning Rate")
    ax_lr.set_xlabel("Epoch")
    ax_lr.grid(True, alpha=PLOT_LAYOUT.grid_alpha)

    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_LAYOUT.dpi)
    plt.close(fig)


def train(
    cfg: TrainConfig,
    split_dir: Path | None = None,
    out_dir: Path | None = None,
    resume: bool = False,
) -> Path:
    """Execute a full training run.

    Args:
        cfg (TrainConfig): Run configuration.
        split_dir (Path | None): Custom directory for
            split CSVs (default=None → standard splits).
        out_dir (Path | None): Override the run output
            directory (default=None → ``_run_dir(cfg)``).
        resume (bool): If True, resume from the latest
            checkpoint in the run directory.

    Returns:
        Path to the run directory containing checkpoints,
        metadata, and loss-curve image.
    """
    _seed_everything(cfg.seed)
    run_dir = out_dir if out_dir is not None else _run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    encoder = _get_encoder(cfg.encoder_name)

    # Data.
    train_loader = build_loader(
        "train",
        encoder=encoder,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        split_dir=split_dir,
    )
    val_loader = build_loader(
        "val",
        encoder=encoder,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        split_dir=split_dir,
    )

    # Model.
    model = build_model(
        cfg.encoder_name,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
    )
    scaler = GradScaler("cuda", enabled=cfg.amp)

    # History tracking.
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }
    best_val = float("inf")
    wait = 0
    start_epoch = 1
    t_start = time.time()

    # Resume from checkpoint.
    if resume:
        ckpt_path = None
        for name in ("last.pt", "best.pt"):
            p = run_dir / name
            if p.exists():
                ckpt_path = p
                break
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and "model" in ckpt:
                model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                scheduler.load_state_dict(ckpt["scheduler"])
                scaler.load_state_dict(ckpt["scaler"])
                best_val = ckpt["best_val"]
                wait = ckpt["wait"]
                history = ckpt["history"]
                start_epoch = ckpt["epoch"] + 1
            else:
                # Legacy checkpoint (model-only state dict).
                model.load_state_dict(ckpt)
                start_epoch = 1
            print(f"[train] resumed from {ckpt_path.name} (start_epoch={start_epoch})")
        else:
            print("[train] --resume set but no checkpoint found, training from scratch.")

    print(
        f"[train] encoder={cfg.encoder_name}  "
        f"seed={cfg.seed}  device={device}  "
        f"params={n_params:,}"
    )
    print(f"[train] run_dir={run_dir}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        nan_detected = False

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                enabled=cfg.amp,
            ):
                pred = model(x)
                loss = nn.functional.mse_loss(pred, y)

            if not torch.isfinite(loss):
                nan_detected = True
                break

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.grad_clip,
                )
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * y.size(0)
            epoch_samples += y.size(0)

        if nan_detected:
            print(f"  epoch {epoch:>3d}/{cfg.epochs}  NaN detected, stopping.")
            break

        avg_train = epoch_loss / max(epoch_samples, 1)
        history["train_loss"].append(avg_train)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        scheduler.step()

        # Validation.
        if epoch % cfg.val_every == 0:
            avg_val = _evaluate(model, val_loader, device)
            history["val_loss"].append(avg_val)
            improved = avg_val < best_val
            if improved:
                best_val = avg_val
                wait = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "best_val": best_val,
                        "wait": wait,
                        "history": history,
                    },
                    run_dir / "best.pt",
                )
            else:
                wait += 1

            print(
                f"  epoch {epoch:>3d}/{cfg.epochs}  "
                f"train={avg_train:.6f}  "
                f"val={avg_val:.6f}"
                f"{'  *best' if improved else ''}"
            )

            if cfg.patience > 0 and wait >= cfg.patience:
                print(f"  early stopping at epoch {epoch} (patience={cfg.patience}).")
                break
        else:
            print(f"  epoch {epoch:>3d}/{cfg.epochs}  train={avg_train:.6f}")

    elapsed = time.time() - t_start

    # Save last checkpoint.
    torch.save(
        {
            "epoch": len(history["train_loss"]),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val": best_val,
            "wait": wait,
            "history": history,
        },
        run_dir / "last.pt",
    )

    # Save loss-curve image.
    _plot_curves(history, run_dir / "curves.png")

    # Save metadata.
    metadata = {
        "config": asdict(cfg),
        "n_params": n_params,
        "best_val_loss": best_val,
        "final_train_loss": history["train_loss"][-1],
        "epochs_completed": len(history["train_loss"]),
        "elapsed_seconds": round(elapsed, 1),
        "device": str(device),
        "history": history,
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[train] done in {elapsed:.0f}s  best_val={best_val:.6f}  dir={run_dir}")
    return run_dir
