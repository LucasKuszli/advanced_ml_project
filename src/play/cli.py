"""Interactive terminal interface for playing chess against
the bot.

Usage:
    uv run python -m src.play
    uv run python -m src.play --color black
    uv run python -m src.play --encoder square_token
    uv run python -m src.play --model artifacts/models/piece_plane_best.pt
"""

from __future__ import annotations

import argparse
import sys

import chess
from src.play.bot import ChessBot
from src.play.game import Game, GamePhase, PlayerColor

# ── Unicode piece symbols ───────────────────────────────

_UNICODE_PIECES: dict[str | None, str] = {
    "R": "♜",
    "N": "♞",
    "B": "♝",
    "Q": "♛",
    "K": "♚",
    "P": "♟",
    "r": "♖",
    "n": "♘",
    "b": "♗",
    "q": "♕",
    "k": "♔",
    "p": "♙",
    None: " ",
}

# ── Board rendering ─────────────────────────────────────


def render_board(board: chess.Board, pov_white: bool = True) -> str:
    """Render the board as a coloured Unicode string.

    Args:
        board: A ``python-chess`` Board.
        pov_white: If ``True`` show from white's
            perspective (rank 8 at the top).

    Returns:
        Multi-line string.
    """
    ranks = range(7, -1, -1) if pov_white else range(8)
    files = range(8) if pov_white else range(7, -1, -1)
    lines: list[str] = []
    lines.append("")
    for r in ranks:
        rank_label = str(r + 1)
        row_chars: list[str] = [f"  {rank_label} "]
        for f in files:
            sq = chess.square(f, r)
            piece = board.piece_at(sq)
            symbol = _UNICODE_PIECES.get(piece.symbol() if piece else None, " ")
            # Alternate square shading.
            if (r + f) % 2 == 0:
                bg = "\033[48;5;180m"  # light square
            else:
                bg = "\033[48;5;95m"  # dark square
            row_chars.append(f"{bg} {symbol} \033[0m")
        lines.append("".join(row_chars))
    # File labels.
    file_labels = "  abcdefgh" if pov_white else "  hgfedcba"
    lines.append("     " + "  ".join(file_labels[i] for i in range(2, 10)))
    lines.append("")
    return "\n".join(lines)


def _print_move_history(game: Game) -> None:
    """Print a compact move history."""
    history = game.state.history
    if not history:
        return
    print("\n  Move history:")
    i = 0
    while i < len(history):
        rec = history[i]
        line = f"  {rec.move_number}."
        if rec.color == "White":
            tag = " (book)" if rec.is_opening else ""
            line += f" {rec.san}{tag}"
            if i + 1 < len(history):
                nxt = history[i + 1]
                tag2 = " (book)" if nxt.is_opening else ""
                line += f"  {nxt.san}{tag2}"
                i += 1
        else:
            tag = " (book)" if rec.is_opening else ""
            line += f" ...  {rec.san}{tag}"
        print(line)
        i += 1


# ── Main game loop ──────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    """Run the interactive chess game."""
    parser = argparse.ArgumentParser(
        description="Play chess against your trained model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=("Path to model checkpoint. If omitted, the bot plays random moves."),
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["piece_plane", "square_token"],
        default=None,
        help=(
            "Board encoder. Auto-detected from the model "
            "path when omitted (default: piece_plane)."
        ),
    )
    parser.add_argument(
        "--color",
        type=str,
        choices=["white", "black"],
        default="white",
        help="Side you play (default: white).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "Bot move-selection temperature. 0 = greedy, >0 = stochastic (default: 0)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducibility.",
    )
    args = parser.parse_args(argv)

    human_color = PlayerColor.WHITE if args.color == "white" else PlayerColor.BLACK

    bot = ChessBot(
        model_path=args.model,
        encoder_name=args.encoder,
        device="cpu",
        temperature=args.temperature,
    )

    game = Game(
        bot=bot,
        human_color=human_color,
        seed=args.seed,
    )

    pov_white = human_color == PlayerColor.WHITE
    you = "White" if pov_white else "Black"
    bot_side = "Black" if pov_white else "White"

    # ── Banner ──────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  ♚  CHESS — Play against your trained model  ♚")
    print("=" * 50)
    if bot.has_model:
        print(f"  Model : {args.model}")
        print(f"  Encoder: {args.encoder}")
    else:
        print("  Model : none (bot plays random moves)")
    print(f"  You   : {you}")
    print(f"  Bot   : {bot_side}")
    print(f"  Opening: {game.state.opening_name}")
    print("-" * 50)
    print(
        "  Enter moves in UCI notation (e.g. e2e4).\n"
        "  Type 'quit' to resign, 'moves' for legal\n"
        "  moves, 'history' for the move list.\n"
    )

    # ── Opening phase ──────────────────────────────────
    print(f"  Playing opening: {game.state.opening_name}")
    print("  (first 4 moves from book)\n")
    game.play_opening_moves()
    print(render_board(game.state.board, pov_white))
    _print_move_history(game)
    print()

    if game.is_over:
        print(f"\n  {game.result_description()}")
        return

    # ── Main game loop ──────────────────────────────────
    while not game.is_over:
        if game.state.is_human_turn:
            # Human turn.
            prompt = (
                f"  [{game.state.move_number}. "
                f"{game.state.active_color_name}] "
                f"Your move: "
            )
            try:
                user_input = input(prompt).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n\n  Game aborted.")
                return

            if user_input in ("quit", "q", "resign"):
                print("\n  You resigned. Bot wins!")
                return
            if user_input in ("moves", "legal"):
                legal = [m.uci() for m in game.state.board.legal_moves]
                print("  Legal moves: " + ", ".join(sorted(legal)))
                continue
            if user_input == "history":
                _print_move_history(game)
                continue
            if user_input in ("board", "show"):
                print(
                    render_board(
                        game.state.board,
                        pov_white,
                    )
                )
                continue
            if not user_input:
                continue

            try:
                game.human_move(user_input)
            except ValueError as exc:
                print(f"  ✗ {exc}")
                continue

            san = game.state.history[-1].san
            print(f"  You played: {san}")
            print(render_board(game.state.board, pov_white))

            if game.is_over:
                break

        # Bot turn.
        if not game.state.is_human_turn:
            print("  Bot is thinking…")
            game.bot_move()
            san = game.state.history[-1].san
            print(f"  Bot played: {san}")
            print(render_board(game.state.board, pov_white))

    # ── End of game ─────────────────────────────────────
    print()
    print("=" * 50)
    print(f"  {game.result_description()}")
    print(f"  Result: {game.result}")
    print("=" * 50)
    _print_move_history(game)
    print()


if __name__ == "__main__":
    main()
