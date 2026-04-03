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

Download both train (~36 GB) and test (~5 MB) `.bag` files, then
sample and write train/val/test CSVs:

```bash
uv run python -m src.data.loader
```

Skip the download step to reuse already-downloaded `.bag` files:

```bash
uv run python -m src.data.loader --skip-download
```

Control how many positions are sampled from each bag:

```bash
uv run python -m src.data.loader --skip-download \
    --num-train-samples 1000000 \
    --num-test-samples 50000
```

## Preprocessing pipeline

1. **Test bag** is processed first. Records are decoded, then
   deduplicated on the first 5 FEN fields (board placement,
   active colour, castling rights, en passant square, and
   halfmove clock — the clock matters because proximity to
   the 50-move draw rule affects evaluation pressure).
   Duplicate positions have their win probabilities averaged.
   Results are written to `test.csv`.
2. **Train bag** is sampled and deduplicated the same way.
   Any positions whose dedup key also appears in the test set
   are removed to guarantee disjoint splits.
3. The remaining train positions are shuffled and split: 90%
   goes to `train.csv`, 10% to `val.csv`.

## Directory layout

```
data/
├── raw/                        # Downloaded .bag files (gitignored)
│   ├── train_state_value.bag
│   └── test_state_value.bag
└── splits/                     # Sampled CSVs (gitignored)
    ├── train.csv
    ├── val.csv
    └── test.csv
```

Each CSV has two columns: `fen` (board position) and `win_prob`
(Stockfish win percentage as a float in [0, 1]).
