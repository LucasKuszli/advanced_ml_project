"""Profile data-loading vs GPU compute for 5 epochs.

Measures wall-clock time spent waiting for each batch
(data) versus running the forward + backward pass (GPU)
and reports per-epoch and overall breakdowns.

Usage:
    uv run python -m src.pipeline.profile_training
    uv run python -m src.pipeline.profile_training --encoder square_token
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
from torch.amp import GradScaler

from src.config.paths import DATA_DIR
from src.config.train import TrainConfig
from src.data.dataset import build_loader
from src.train.trainer import _get_encoder, _seed_everything
from src.model.model import build_model

PROFILE_EPOCHS = 5
SPLIT_DIR = DATA_DIR / "splits" / "seed42"


def _fmt(seconds: float) -> str:
    """Format seconds as M:SS.s."""
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:04.1f}"


def profile(encoder_name: str) -> None:
    """Run a timed training loop and print breakdown.

    Args:
        encoder_name (str): ``"piece_plane"`` or
            ``"square_token"``.
    """
    cfg = TrainConfig(
        encoder_name=encoder_name,
        seed=42,
        epochs=PROFILE_EPOCHS,
    )
    _seed_everything(cfg.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    encoder = _get_encoder(cfg.encoder_name)

    train_loader = build_loader(
        "train",
        encoder=encoder,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        split_dir=SPLIT_DIR,
    )

    model = build_model(
        cfg.encoder_name,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scaler = GradScaler("cuda", enabled=cfg.amp)

    n_params = sum(p.numel() for p in model.parameters())
    n_batches = len(train_loader)

    print(f"[profile] encoder={encoder_name}  device={device}")
    print(f"[profile] params={n_params:,}  batches/epoch={n_batches}")
    print(f"[profile] batch_size={cfg.batch_size}  workers={cfg.num_workers}")
    print()

    total_data = 0.0
    total_gpu = 0.0
    total_misc = 0.0

    for epoch in range(1, PROFILE_EPOCHS + 1):
        model.train()
        epoch_data = 0.0
        epoch_gpu = 0.0
        epoch_misc = 0.0

        torch.cuda.synchronize()
        t_epoch = time.perf_counter()
        t_batch_start = time.perf_counter()

        for step, (x, y) in enumerate(train_loader):
            # --- Data time: waiting for the batch ---
            t_data_end = time.perf_counter()
            epoch_data += t_data_end - t_batch_start

            # --- GPU time: transfer + forward + backward ---
            torch.cuda.synchronize()
            t_gpu_start = time.perf_counter()

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                enabled=cfg.amp,
            ):
                pred = model(x)
                loss = nn.functional.mse_loss(pred, y)

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.grad_clip,
                )
            scaler.step(optimizer)
            scaler.update()

            torch.cuda.synchronize()
            t_gpu_end = time.perf_counter()
            epoch_gpu += t_gpu_end - t_gpu_start

            # --- Misc: python overhead between batches ---
            t_batch_start = time.perf_counter()
            epoch_misc += t_batch_start - t_gpu_end

        torch.cuda.synchronize()
        epoch_wall = time.perf_counter() - t_epoch

        total_data += epoch_data
        total_gpu += epoch_gpu
        total_misc += epoch_misc

        pct_data = epoch_data / epoch_wall * 100
        pct_gpu = epoch_gpu / epoch_wall * 100
        pct_misc = epoch_misc / epoch_wall * 100

        print(
            f"  epoch {epoch}/{PROFILE_EPOCHS}  "
            f"wall={_fmt(epoch_wall)}  "
            f"data={_fmt(epoch_data)} ({pct_data:.1f}%)  "
            f"gpu={_fmt(epoch_gpu)} ({pct_gpu:.1f}%)  "
            f"misc={_fmt(epoch_misc)} ({pct_misc:.1f}%)"
        )

    grand_total = total_data + total_gpu + total_misc
    print()
    print(f"  {'Total':>18s}  wall={_fmt(grand_total)}")
    print(
        f"  {'Data loading':>18s}  "
        f"{_fmt(total_data)} "
        f"({total_data / grand_total * 100:.1f}%)"
    )
    print(
        f"  {'GPU compute':>18s}  "
        f"{_fmt(total_gpu)} "
        f"({total_gpu / grand_total * 100:.1f}%)"
    )
    print(
        f"  {'Python overhead':>18s}  "
        f"{_fmt(total_misc)} "
        f"({total_misc / grand_total * 100:.1f}%)"
    )

    throughput = PROFILE_EPOCHS * n_batches * cfg.batch_size / grand_total
    print(f"\n  Throughput: {throughput:,.0f} samples/s")


def main() -> None:
    """Parse args and run profiling."""
    parser = argparse.ArgumentParser(
        description="Profile training data vs GPU time.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="piece_plane",
        help="Encoder to profile (default: piece_plane).",
    )
    args = parser.parse_args()
    profile(args.encoder)


if __name__ == "__main__":
    main()
