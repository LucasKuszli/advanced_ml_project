# Data

## Source

All data comes from **ChessBench**, released alongside the paper
*Amortized Planning with Large-Scale Transformers: A Case Study on Chess*
(Ruoss et al., NeurIPS 2024). The dataset contains millions of chess
positions extracted from Lichess games and annotated by Stockfish 16.

- Paper: https://arxiv.org/abs/2402.04494
- Repository: https://github.com/google-deepmind/searchless_chess
- License: CC-BY 4.0 (Stockfish annotations), CC0 (Lichess games)

We use only the **state-value** subset, where each record is a
`(FEN, win_probability)` pair — the board position in FEN notation
and Stockfish's estimated win probability (0.0 = black wins,
1.0 = white wins).

The full ChessBench release also includes action-value data
(per-move win probabilities, ~1.1 TB) and behavioral cloning data
(best-move labels, ~34 GB), which we do not use.

## Usage

Download the training `.bag` file (~36 GB), sample 10M positions,
and split into train/val/test CSVs (80/10/10):

```bash
uv run python -m src.data.loader
```

Use `--num-samples` for a smaller subset or `--skip-download` to
reuse an already-downloaded `.bag` file:

```bash
uv run python -m src.data.loader --num-samples 1000000
uv run python -m src.data.loader --skip-download --seed 123
```

## Directory layout

```
data/
├── raw/                        # Downloaded .bag files (gitignored)
│   ├── train_state_value.bag
│   └── test_state_value.bag
├── splits/                     # Sampled CSVs (gitignored)
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── README.md
```

Each CSV has two columns: `fen` (board position) and `win_prob`
(Stockfish win percentage as a float in [0, 1]).
