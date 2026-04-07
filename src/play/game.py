"""Game engine that orchestrates a human-vs-bot chess game.

Responsibilities:
* Manage the ``python-chess`` board.
* Play the opening phase (bot's first 4 moves come from a
  randomly selected famous opening).
* Delegate bot moves to ``ChessBot`` after the opening.
* Track game result and move history.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto

import chess
from src.play.bot import ChessBot, EvalResult
from src.play.openings import pick_opening


class GamePhase(Enum):
    """Current phase of the game."""

    OPENING = auto()
    PLAY = auto()
    FINISHED = auto()


class PlayerColor(Enum):
    """Side the human plays."""

    WHITE = auto()
    BLACK = auto()


@dataclass
class MoveRecord:
    """One half-move in the game history."""

    move_number: int
    color: str
    uci: str
    san: str
    is_opening: bool = False


@dataclass
class GameState:
    """Full state of a game in progress."""

    board: chess.Board = field(
        default_factory=chess.Board,
    )
    human_color: PlayerColor = PlayerColor.WHITE
    opening_name: str = ""
    opening_moves: list[str] = field(default_factory=list)
    opening_index: int = 0
    phase: GamePhase = GamePhase.OPENING
    history: list[MoveRecord] = field(default_factory=list)
    move_number: int = 1

    @property
    def is_human_turn(self) -> bool:
        """Whether it is the human's turn to move."""
        if self.human_color == PlayerColor.WHITE:
            return self.board.turn == chess.WHITE
        return self.board.turn == chess.BLACK

    @property
    def active_color_name(self) -> str:
        return "White" if self.board.turn == chess.WHITE else "Black"


class Game:
    """High-level game controller.

    Args:
        bot: A ``ChessBot`` instance.
        human_color: Side the human plays.
        seed: RNG seed for opening selection.
        use_openings: If ``True`` the first 4 moves come
            from a randomly selected famous opening.
            Default is ``False`` (model plays from move 1).
    """

    def __init__(
        self,
        bot: ChessBot,
        human_color: PlayerColor = PlayerColor.WHITE,
        seed: int | None = None,
        use_openings: bool = False,
    ) -> None:
        self.bot = bot
        self.rng = random.Random(seed)
        self.use_openings = use_openings
        self.state = GameState(human_color=human_color)

        if use_openings:
            name, moves = pick_opening(self.rng)
            self.state.opening_name = name
            self.state.opening_moves = moves
        else:
            self.state.phase = GamePhase.PLAY

    # ── Opening phase ───────────────────────────────────

    def _is_opening_move(self) -> bool:
        """True if the next move is part of the opening
        book (first 8 half-moves = 4 full moves)."""
        return self.state.phase == GamePhase.OPENING and self.state.opening_index < len(
            self.state.opening_moves
        )

    def _play_opening_move(self) -> chess.Move:
        """Play the next opening-book move on the board."""
        uci = self.state.opening_moves[self.state.opening_index]
        move = chess.Move.from_uci(uci)
        # Validate the move is legal (sanity check).
        if move not in self.state.board.legal_moves:
            # Fallback: if the opening line is somehow
            # illegal (shouldn't happen with a curated
            # book), skip remaining opening moves.
            self.state.phase = GamePhase.PLAY
            move, _ = self.bot.select_move(
                self.state.board,
                self.rng,
            )
        san = self.state.board.san(move)
        self.state.board.push(move)
        self._record(move, san, is_opening=True)
        self.state.opening_index += 1
        if self.state.opening_index >= len(self.state.opening_moves):
            self.state.phase = GamePhase.PLAY
        return move

    # ── Move recording ──────────────────────────────────

    def _record(
        self,
        move: chess.Move,
        san: str,
        is_opening: bool = False,
    ) -> None:
        color = "Black" if self.state.board.turn == chess.WHITE else "White"
        # After push, turn has flipped — so the mover
        # is the opposite of current turn.
        rec = MoveRecord(
            move_number=self.state.move_number,
            color=color,
            uci=move.uci(),
            san=san,
            is_opening=is_opening,
        )
        self.state.history.append(rec)
        # Increment full-move number after black moves.
        if color == "Black":
            self.state.move_number += 1

    # ── Public API ──────────────────────────────────────

    def play_opening_moves(self) -> list[chess.Move]:
        """Auto-play all opening book moves.

        Returns:
            List of moves played.
        """
        played: list[chess.Move] = []
        while self._is_opening_move():
            played.append(self._play_opening_move())
            self._check_terminal()
        return played

    def human_move(self, uci: str) -> chess.Move:
        """Attempt a human move in UCI notation.

        Args:
            uci: Move string, e.g. ``"e2e4"``.

        Returns:
            The ``chess.Move`` that was played.

        Raises:
            ValueError: If the move is illegal or it is
                not the human's turn.
        """
        if not self.state.is_human_turn:
            raise ValueError("It is not your turn!")
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            raise ValueError(f"Invalid UCI notation: {uci!r}")
        if move not in self.state.board.legal_moves:
            # Check for promotion — if the player typed
            # e.g. "e7e8" but meant "e7e8q", try adding
            # queen promotion.
            promo = chess.Move.from_uci(uci + "q")
            if promo in self.state.board.legal_moves:
                move = promo
            else:
                raise ValueError(
                    f"Illegal move: {uci}. Legal moves: "
                    + ", ".join(m.uci() for m in self.state.board.legal_moves)
                )
        san = self.state.board.san(move)
        self.state.board.push(move)
        self._record(move, san)
        self._check_terminal()
        return move

    def bot_move(self) -> tuple[chess.Move, EvalResult]:
        """Let the bot make a move.

        Returns:
            ``(move, eval_result)`` — the chosen move and
            the model's evaluation data.
        """
        if self.state.is_human_turn:
            raise ValueError("It is not the bot's turn!")
        move, eval_result = self.bot.select_move(
            self.state.board,
            self.rng,
        )
        san = self.state.board.san(move)
        self.state.board.push(move)
        self._record(move, san)
        self._check_terminal()
        return move, eval_result

    def _check_terminal(self) -> None:
        """Update phase if the game is over."""
        if self.state.board.is_game_over():
            self.state.phase = GamePhase.FINISHED

    @property
    def is_over(self) -> bool:
        return self.state.phase == GamePhase.FINISHED

    @property
    def result(self) -> str:
        """Game result string (e.g. '1-0', '0-1',
        '1/2-1/2') or '*' if still in progress."""
        return self.state.board.result()

    def result_description(self) -> str:
        """Human-readable game result."""
        board = self.state.board
        if board.is_checkmate():
            winner = "Black" if board.turn == chess.WHITE else "White"
            return f"Checkmate! {winner} wins."
        if board.is_stalemate():
            return "Draw by stalemate."
        if board.is_insufficient_material():
            return "Draw by insufficient material."
        if board.is_fifty_moves():
            return "Draw by fifty-move rule."
        if board.is_repetition():
            return "Draw by threefold repetition."
        return f"Game over: {self.result}"
