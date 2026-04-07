# ChessTransformer

An encoder-only transformer that learns to predict Stockfish
win-probabilities from raw board positions.  Trained on the
[ChessBench](https://github.com/google-deepmind/searchless_chess)
dataset (state-value data), the model outputs $P(\text{white wins}) \in [0, 1]$
for any legal chess position.

The project includes:

* **Three board encoders** of increasing richness
* **Multi-seed training** with disjoint data splits
* **Test-set evaluation** with publication-ready plots
* **Interactive GUI** — play against your trained model

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Repository Structure](#repository-structure)
4. [Data Download](#data-download)
5. [Board Encoders](#board-encoders)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Visualization](#visualization)
9. [Play Against the Model](#play-against-the-model)

---

## Requirements

| Dependency | Version |
|---|---|
| Python | ≥ 3.12 |
| PyTorch | ≥ 2.11.0 |
| pygame | ≥ 2.6.1 |
| python-chess | ≥ 1.11.2 |

The full list is declared in `pyproject.toml`.  A CUDA-capable
GPU is strongly recommended for training (CPU works for inference
and the GUI).

---

## Installation

The project uses **[uv](https://docs.astral.sh/uv/)** for
dependency management.

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repo
git clone <repo-url> && cd advanced_ml_project

# 3. Create the virtual environment and install all deps
uv sync
```

> After `uv sync` every command is run through `uv run` so that
> the correct environment is always activated.

---

## Repository Structure

```
advanced_ml_project/
├── pyproject.toml              # Project metadata & dependencies
├── README.md
├── data/
│   └── raw/                    # Downloaded .bag files (auto-created)
├── artifacts/
│   └── models/
│       └── d128_L5/            # Trained checkpoints per encoder & seed
├── evaluation/
│   └── d128_L5/                # Test-set results & plots per encoder
└── src/
    ├── config/                 # Centralised configuration dataclasses
    │   ├── data.py             #   ChessBench URLs, split config, sampling
    │   ├── encoder.py          #   Channel layouts for each encoder
    │   ├── model.py            #   Transformer architecture defaults
    │   ├── paths.py            #   Project directory constants
    │   ├── render.py           #   Board image rendering settings
    │   └── train.py            #   TrainConfig, seeds, tuning grid
    ├── data/                   # Data pipeline
    │   ├── setup.py            #   One-shot download & split creation
    │   ├── loader.py           #   BagReader, download, split helpers
    │   └── dataset.py          #   PyTorch Dataset & DataLoader factory
    ├── encoders/               # Board → tensor encoders
    │   ├── base.py             #   Abstract base class
    │   ├── piece_plane.py      #   PiecePlaneEncoder (19 channels)
    │   ├── enriched_piece_plane.py  # Dynamic (31ch) & Full (37ch)
    │   └── features.py         #   Dynamic & structural feature computation
    ├── model/                  # Neural network
    │   ├── model.py            #   ChessTransformer + build_model()
    │   └── input_stem.py       #   Per-encoder input stems
    ├── train/                  # Training loop
    │   ├── trainer.py          #   AMP, cosine LR, early stopping, grad clip
    │   └── run.py              #   Single-run entry point
    ├── pipeline/               # High-level orchestration scripts
    │   ├── train_piece_plane.py    # Multi-seed training CLI
    │   ├── tune_hparams.py         # LR × dropout grid search
    │   └── profile_training.py     # Data-loading vs GPU profiling
    ├── evaluate/               # Test-set evaluation
    │   ├── evaluate.py         #   Per-seed inference & metric computation
    │   ├── visualize.py        #   Bar charts & scatter plots
    │   └── metrics.py          #   MSE, MAE, Pearson, Spearman, sign accuracy
    └── play/                   # Interactive chess GUI
        ├── __main__.py         #   Entry point (python -m src.play)
        ├── gui.py              #   Pygame GUI (resizable window, eval bar)
        ├── game.py             #   Game engine & move logic
        ├── bot.py              #   Model-backed bot with encoder auto-detection
        ├── openings.py         #   21 famous openings database
        └── cli.py              #   Text-based CLI (legacy)
```

---

## Data Download

The training data comes from the **ChessBench** dataset hosted on
Google Cloud Storage.  The train bag is ≈ 18 GB; the test bag is
≈ 5 MB.

```bash
# Download bags + create all disjoint splits in one go
uv run python -m src.data.setup
```

This will:

1. Download `state_value_data.bag` for train and test into `data/raw/`.
2. Create a shared **test split** (sampled from the test bag).
3. Create **3 production splits** (seeds 42, 67, 1337) — each with
   10 M positions — from the train bag, with zero overlap.
4. Create a **tuning split** for hyperparameter search.

All splits are saved as CSV files under `data/`.

---

## Board Encoders

Each encoder converts a `chess.Board` into a float tensor of shape
$(C, 8, 8)$.  Three encoders are available, each a strict superset
of the previous one:

| Encoder | Channels | Description |
|---|---|---|
| `piece_plane` | 19 | 12 piece planes + side-to-move, castling, EP, halfmove |
| `dynamic_piece_plane` | 31 | + attack maps, defence maps, reachability, pins, check |
| `full_piece_plane` | 37 | + doubled / isolated / passed pawn structure planes |

---

## Training

Training uses a **ChessTransformer** — a small encoder-only
transformer with mean-pooling and a sigmoid head.

| Hyper-parameter | Default |
|---|---|
| `d_model` | 128 |
| `n_heads` | 4 |
| `n_layers` | 5 |
| `ff_factor` | 4 |
| `dropout` | 0.05 |
| `activation` | GELU |
| `norm_first` | True |
| `batch_size` | 4 096 |
| `lr` | 1 × 10⁻³ |
| `epochs` | 40 |
| Scheduler | Cosine annealing |
| Early stopping | patience = 10 |

### Run training

```bash
# Train all 3 seeds for the base encoder
uv run python -m src.pipeline.train_piece_plane --encoder piece_plane

# Train a specific seed
uv run python -m src.pipeline.train_piece_plane --encoder dynamic_piece_plane --seed 42 67

# Resume an interrupted run
uv run python -m src.pipeline.train_piece_plane --encoder full_piece_plane --seed 42 --resume
```

Checkpoints (`best.pt`, `last.pt`), `metadata.json`, and learning
curves (`curves.png`) are saved to:

```
artifacts/models/d128_L5/<encoder>_seed<N>/
```

### Hyperparameter tuning

```bash
# Grid search over LR × dropout (short 5-epoch trials)
uv run python -m src.pipeline.tune_hparams
uv run python -m src.pipeline.tune_hparams --encoder piece_plane
```

### Profiling

```bash
# Profile data-loading vs GPU compute for 5 epochs
uv run python -m src.pipeline.profile_training
uv run python -m src.pipeline.profile_training --encoder dynamic_piece_plane
```

---

## Evaluation

Evaluate trained models on the held-out test set.  Metrics
computed: **MSE**, **MAE**, **Pearson r**, **Spearman ρ**, and
**sign accuracy**.

```bash
# Evaluate all seeds for an encoder
uv run python -m src.evaluate.evaluate --encoder piece_plane

# Evaluate specific seeds
uv run python -m src.evaluate.evaluate --encoder dynamic_piece_plane --seed 42 67
```

Results are saved to:

```
evaluation/d128_L5/<encoder>/results.json
```

---

## Visualization

Generate publication-ready plots from evaluation results:

```bash
uv run python -m src.evaluate.visualize --encoder piece_plane
uv run python -m src.evaluate.visualize --encoder dynamic_piece_plane --num-workers 2
```

This produces per-metric bar charts and a prediction-vs-target
scatter plot, saved as PNGs alongside `results.json` in the
`evaluation/` directory.

---

## Play Against the Model

Launch an interactive **pygame** GUI where you play against your
trained model.  The bot evaluates every legal move and picks the
best (or samples stochastically with `--temperature`).

```bash
# Play against a trained model (white pieces, greedy bot)
uv run python -m src.play \
    --model artifacts/models/d128_L5/piece_plane_seed42/best.pt

# Play as black
uv run python -m src.play \
    --model artifacts/models/d128_L5/dynamic_piece_plane_seed67/best.pt \
    --color black

# Enable famous opening book for the first 4 moves
uv run python -m src.play \
    --model artifacts/models/d128_L5/piece_plane_seed42/best.pt \
    --openings

# Stochastic play (temperature > 0)
uv run python -m src.play \
    --model artifacts/models/d128_L5/piece_plane_seed42/best.pt \
    --temperature 0.5 \
    --seed 123

# Random-move bot (no model)
uv run python -m src.play
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--model PATH` | *None* | Path to `.pt` checkpoint. Omit for random moves. |
| `--encoder NAME` | *auto* | `piece_plane`, `dynamic_piece_plane`, or `full_piece_plane`. Auto-detected from the model path when omitted. |
| `--color` | `white` | Side you play (`white` or `black`). |
| `--temperature` | `0.0` | Bot temperature. 0 = greedy, > 0 = stochastic. |
| `--seed` | *None* | RNG seed for reproducibility. |
| `--openings` | off | Use pre-programmed famous openings for the first 4 moves. |

### GUI features

* **Click-to-move** — select a piece, then click its destination.
* **Eval bar** — vertical bar showing the model's assessment of
  the current position.
* **Top moves panel** — the bot's top-5 candidate moves with
  scores are displayed in the side panel.
* **Resizable window** — the board and panels adapt to any window
  size, including fullscreen.

---

## License

See [LICENSE](LICENSE).
