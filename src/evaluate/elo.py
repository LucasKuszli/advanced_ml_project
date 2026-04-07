"""Estimate Elo ratings for trained models by playing
games against Stockfish at calibrated strength levels.

The script plays many games between each model (acting as
a chess bot) and Stockfish configured to various Elo
ratings via UCI_LimitStrength.  The win/draw/loss
statistics are used to estimate the model's Elo via
maximum-likelihood fitting of the standard Elo expected
score formula.

Usage:
    uv run python -m src.evaluate.elo
    uv run python -m src.evaluate.elo --encoder full_piece_plane
    uv run python -m src.evaluate.elo --games 40 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
from tqdm import tqdm

from src.config.paths import EVAL_DIR, MODEL_DIR
from src.config.train import RUN_SEEDS
from src.play.bot import ChessBot

MODEL_TAG = "d128_L5"

STOCKFISH_PATH = shutil.which("stockfish") or "/usr/games/stockfish"

# Stockfish Elo levels to test against.
# Stockfish 16 minimum UCI_Elo is 1320.
DEFAULT_LEVELS = (1320, 1500, 1700, 1900, 2100, 2300, 2500)

# Maximum moves per game before declaring a draw.
MAX_MOVES = 200

# Stockfish thinking time per move (seconds).
SF_TIME_LIMIT = 0.1


@dataclass
class MatchResult:
    """Results of a match at one Stockfish Elo level."""

    sf_elo: int
    wins: int
    draws: int
    losses: int

    @property
    def total(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        """Score as fraction: win=1, draw=0.5, loss=0."""
        if self.total == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.total


def _play_game(
    bot: ChessBot,
    sf_engine: chess.engine.SimpleEngine,
    sf_time: float,
    bot_color: chess.Color,
    rng: random.Random,
) -> str:
    """Play one game and return the result string.

    Args:
        bot: The model-based chess bot.
        sf_engine: Running Stockfish UCI engine.
        sf_time: Time limit per Stockfish move.
        bot_color: chess.WHITE or chess.BLACK.
        rng: RNG for the bot's move selection.

    Returns:
        ``"1-0"``, ``"0-1"``, or ``"1/2-1/2"``.
    """
    board = chess.Board()

    for _ in range(MAX_MOVES * 2):
        if board.is_game_over():
            break

        if board.turn == bot_color:
            move, _ = bot.select_move(board, rng)
        else:
            result = sf_engine.play(
                board,
                chess.engine.Limit(time=sf_time),
            )
            move = result.move

        board.push(move)

    return board.result()


def _result_to_wdl(
    result_str: str,
    bot_color: chess.Color,
) -> tuple[int, int, int]:
    """Convert a game result string to (win, draw, loss)
    from the bot's perspective.

    Returns:
        Tuple of (win, draw, loss) — exactly one is 1.
    """
    if result_str == "1/2-1/2" or result_str == "*":
        return (0, 1, 0)
    bot_wins = (result_str == "1-0" and bot_color == chess.WHITE) or (
        result_str == "0-1" and bot_color == chess.BLACK
    )
    if bot_wins:
        return (1, 0, 0)
    return (0, 0, 1)


def _play_single_game(
    bot: ChessBot,
    sf_elo: int,
    sf_time: float,
    bot_color: chess.Color,
    game_seed: int,
    sf_path: str,
) -> tuple[int, int, int]:
    """Play one game in its own Stockfish process.

    Returns:
        ``(win, draw, loss)`` from the bot's perspective.
    """
    rng = random.Random(game_seed)
    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    engine.configure(
        {
            "UCI_LimitStrength": True,
            "UCI_Elo": sf_elo,
            "Threads": 1,
            "Hash": 16,
        }
    )
    try:
        result_str = _play_game(
            bot,
            engine,
            sf_time,
            bot_color,
            rng,
        )
    finally:
        engine.quit()
    return _result_to_wdl(result_str, bot_color)


def play_match(
    bot: ChessBot,
    sf_elo: int,
    n_games: int,
    sf_time: float = SF_TIME_LIMIT,
    seed: int = 42,
    sf_path: str = STOCKFISH_PATH,
    workers: int = 1,
) -> MatchResult:
    """Play a match of ``n_games`` against Stockfish at a
    fixed Elo level.

    Half the games are played as white, half as black.
    When ``workers > 1``, games run in parallel threads,
    each with its own Stockfish process.

    Args:
        bot: The model-based chess bot.
        sf_elo: Stockfish's Elo limit.
        n_games: Total games to play (split evenly by
            color).
        sf_time: Time per Stockfish move in seconds.
        seed: RNG seed for reproducibility.
        sf_path: Path to stockfish binary.
        workers: Number of parallel threads.

    Returns:
        Aggregated ``MatchResult``.
    """
    # Pre-compute per-game seeds so results are
    # deterministic regardless of parallelism.
    base_rng = random.Random(seed)
    game_specs = [
        (
            chess.WHITE if i % 2 == 0 else chess.BLACK,
            base_rng.randint(0, 2**31),
        )
        for i in range(n_games)
    ]

    wins, draws, losses = 0, 0, 0
    desc = f"vs SF {sf_elo}"

    if workers <= 1:
        for bot_color, game_seed in tqdm(
            game_specs,
            desc=desc,
            unit="game",
        ):
            w, d, l = _play_single_game(
                bot,
                sf_elo,
                sf_time,
                bot_color,
                game_seed,
                sf_path,
            )
            wins += w
            draws += d
            losses += l
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _play_single_game,
                    bot,
                    sf_elo,
                    sf_time,
                    bot_color,
                    game_seed,
                    sf_path,
                )
                for bot_color, game_seed in game_specs
            ]
            for fut in tqdm(
                as_completed(futures),
                total=n_games,
                desc=desc,
                unit="game",
            ):
                w, d, l = fut.result()
                wins += w
                draws += d
                losses += l

    return MatchResult(
        sf_elo=sf_elo,
        wins=wins,
        draws=draws,
        losses=losses,
    )


def estimate_elo(
    match_results: list[MatchResult],
) -> float:
    """Estimate model Elo from match results at multiple
    Stockfish levels via maximum-likelihood.

    Uses the standard Elo expected-score formula:
        E = 1 / (1 + 10^((R_opp - R_model) / 400))

    and finds R_model that maximises the log-likelihood of
    the observed scores.

    Args:
        match_results: List of ``MatchResult`` from games
            at different Stockfish Elo levels.

    Returns:
        Estimated Elo rating.
    """
    import math

    def neg_log_likelihood(elo: float) -> float:
        nll = 0.0
        for mr in match_results:
            if mr.total == 0:
                continue
            expected = 1.0 / (1.0 + 10.0 ** ((mr.sf_elo - elo) / 400.0))
            # Clamp to avoid log(0).
            expected = max(1e-10, min(1 - 1e-10, expected))
            # Wins contribute log(E), losses log(1-E),
            # draws log(0.5) approximately — use the
            # standard approach: score * log(E) +
            # (1 - score) * log(1 - E), weighted by
            # number of games.
            score = mr.score
            n = mr.total
            nll -= n * (
                score * math.log(expected) + (1 - score) * math.log(1 - expected)
            )
        return nll

    # Golden-section search over a plausible Elo range.
    lo, hi = 200.0, 3000.0
    for _ in range(200):
        if hi - lo < 0.1:
            break
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if neg_log_likelihood(m1) < neg_log_likelihood(m2):
            hi = m2
        else:
            lo = m1

    return round((lo + hi) / 2)


def run_elo_estimation(
    encoder_name: str,
    train_seed: int,
    n_games: int,
    sf_levels: tuple[int, ...],
    sf_time: float,
    game_seed: int,
    device: str,
    sf_path: str = STOCKFISH_PATH,
    workers: int = 1,
) -> dict:
    """Run full Elo estimation for one model checkpoint.

    Args:
        encoder_name: Encoder name.
        train_seed: Training seed (selects checkpoint).
        n_games: Games per Stockfish level.
        sf_levels: Stockfish Elo levels to play against.
        sf_time: Stockfish time per move.
        game_seed: RNG seed for game play.
        device: Torch device.

    Returns:
        Dict with match results and estimated Elo.
    """
    model_dir = MODEL_DIR / MODEL_TAG / f"{encoder_name}_seed{train_seed}"
    ckpt = model_dir / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt}",
        )

    bot = ChessBot(
        model_path=str(ckpt),
        encoder_name=encoder_name,
        device=device,
        temperature=0.0,
    )

    match_results: list[MatchResult] = []
    for sf_elo in sf_levels:
        mr = play_match(
            bot,
            sf_elo=sf_elo,
            n_games=n_games,
            sf_time=sf_time,
            seed=game_seed,
            sf_path=sf_path,
            workers=workers,
        )
        print(
            f"  SF {sf_elo}: W={mr.wins}  D={mr.draws}  "
            f"L={mr.losses}  score={mr.score:.2%}",
        )
        match_results.append(mr)

    elo = estimate_elo(match_results)
    print(f"  → Estimated Elo: {elo}\n")

    return {
        "encoder": encoder_name,
        "train_seed": train_seed,
        "estimated_elo": elo,
        "matches": [
            {
                "sf_elo": mr.sf_elo,
                "wins": mr.wins,
                "draws": mr.draws,
                "losses": mr.losses,
                "score": round(mr.score, 4),
            }
            for mr in match_results
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate Elo by playing vs Stockfish.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="full_piece_plane",
        help="Encoder name (default=full_piece_plane).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=list(RUN_SEEDS.seeds),
        help="Training seeds to evaluate.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=20,
        help="Games per Stockfish level (default=20).",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=list(DEFAULT_LEVELS),
        help="Stockfish Elo levels to test against.",
    )
    parser.add_argument(
        "--sf-time",
        type=float,
        default=SF_TIME_LIMIT,
        help="Stockfish time per move in seconds.",
    )
    parser.add_argument(
        "--game-seed",
        type=int,
        default=42,
        help="RNG seed for game play.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device (default=cuda).",
    )
    parser.add_argument(
        "--stockfish",
        type=str,
        default=STOCKFISH_PATH,
        help="Path to stockfish binary.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel games per match (default=1).",
    )
    args = parser.parse_args()

    sf_path = args.stockfish

    all_results: list[dict] = []
    for train_seed in args.seed:
        print(
            f"[elo] {args.encoder} seed={train_seed}",
        )
        result = run_elo_estimation(
            encoder_name=args.encoder,
            train_seed=train_seed,
            n_games=args.games,
            sf_levels=tuple(args.levels),
            sf_time=args.sf_time,
            game_seed=args.game_seed,
            device=args.device,
            sf_path=sf_path,
            workers=args.workers,
        )
        all_results.append(result)

    # Summary table.
    print("=" * 50)
    print(f"{'seed':>6s}  {'Elo':>6s}")
    print("-" * 50)
    elos = []
    for r in all_results:
        print(f"{r['train_seed']:>6d}  {r['estimated_elo']:>6d}")
        elos.append(r["estimated_elo"])
    if len(elos) > 1:
        import numpy as np

        print("-" * 50)
        print(
            f"{'mean':>6s}  {np.mean(elos):>6.0f} ± {np.std(elos):.0f}",
        )
    print("=" * 50)

    # Save results.
    out_dir = EVAL_DIR / MODEL_TAG / args.encoder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "elo_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
