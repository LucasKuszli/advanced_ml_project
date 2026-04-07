"""Chess bot that uses a trained ChessTransformer to pick
moves.

The bot evaluates every legal move by:
1. Playing the move on a copy of the board.
2. Encoding the resulting position with the appropriate
   encoder (piece-plane, dynamic-piece-plane, or
   full-piece-plane).
3. Running the model forward pass to get a win-probability.
4. Choosing the move that maximises its own winning
   chances.

When no model checkpoint is available the bot falls back to
a random legal move so the game can still be played.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import torch

import chess
from src.chess.base import ChessPosition
from src.encoders.base import BoardEncoder
from src.encoders.enriched_piece_plane import (
    DynamicPiecePlaneEncoder,
    FullPiecePlaneEncoder,
)
from src.encoders.piece_plane import PiecePlaneEncoder
from src.model.model import build_model

# ── Evaluation result ───────────────────────────────────


@dataclass
class MoveScore:
    """A candidate move with its model evaluation."""

    move: chess.Move
    san: str
    win_prob: float  # P(white wins) for this move


@dataclass
class EvalResult:
    """Full evaluation output from the bot.

    Attributes:
        position_eval: Model's P(white wins) for the
            *current* position (before any move).
        chosen: The ``MoveScore`` the bot selected.
        top_moves: The best N candidate moves sorted by
            desirability for the side to move.
        all_moves: Every legal move with its score.
        used_model: Whether the model was used (vs random).
    """

    position_eval: float | None = None
    chosen: MoveScore | None = None
    top_moves: list[MoveScore] = field(default_factory=list)
    all_moves: list[MoveScore] = field(default_factory=list)
    used_model: bool = False


# ── Encoder registry ────────────────────────────────────

_ENCODERS: dict[str, type[BoardEncoder]] = {
    "piece_plane": PiecePlaneEncoder,
    "dynamic_piece_plane": DynamicPiecePlaneEncoder,
    "full_piece_plane": FullPiecePlaneEncoder,
}


def _fen_from_board(board: chess.Board) -> str:
    """Return the full FEN of a ``python-chess`` Board."""
    return board.fen()


def infer_encoder_from_path(
    model_path: str | Path,
) -> str | None:
    """Infer the encoder name from a model checkpoint path.

    Expects a path like
    ``artifacts/models/<arch>/<encoder>_seed<N>/best.pt``
    where ``<encoder>`` is ``piece_plane``,
    ``dynamic_piece_plane``, or ``full_piece_plane``.

    The longest matching encoder name wins so that e.g.
    ``dynamic_piece_plane`` is preferred over
    ``piece_plane``.

    Args:
        model_path: Path to the checkpoint file.

    Returns:
        Encoder name string, or ``None`` if the encoder
        cannot be determined.
    """
    parts = Path(model_path).parts
    # Sort encoder names longest-first so that
    # "dynamic_piece_plane" matches before "piece_plane".
    candidates = sorted(_ENCODERS, key=len, reverse=True)
    for part in parts:
        for name in candidates:
            if part.startswith(name):
                return name
    return None


class ChessBot:
    """Model-backed chess player.

    Args:
        model_path: Path to a saved ``ChessTransformer``
            checkpoint (``state_dict`` format).  If
            ``None`` or the file does not exist the bot
            uses random moves.
        encoder_name: ``"piece_plane"``,
            ``"dynamic_piece_plane"``, or
            ``"full_piece_plane"``.  When ``None`` (the
            default) the encoder is inferred from the
            model path (e.g. a path containing
            ``dynamic_piece_plane_seed42`` →
            ``"dynamic_piece_plane"``).
            Falls back to ``"piece_plane"`` if detection
            fails.
        device: Torch device string (``"cpu"`` /
            ``"cuda"``).
        temperature: Softmax temperature applied to
            win-probabilities to control randomness.
            Lower → greedier, higher → more exploratory.
            When ``0`` the bot is fully greedy.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        encoder_name: str | None = None,
        device: str = "cpu",
        temperature: float = 0.0,
    ) -> None:
        if encoder_name is None:
            if model_path is not None:
                encoder_name = infer_encoder_from_path(
                    model_path,
                )
            if encoder_name is None:
                encoder_name = "piece_plane"
        self.encoder_name = encoder_name
        self.encoder = _ENCODERS[encoder_name]()
        self.device = torch.device(device)
        self.temperature = temperature
        self.model = self._load_model(model_path)

    # ── Model loading ───────────────────────────────────

    def _load_model(
        self,
        model_path: str | Path | None,
    ) -> torch.nn.Module | None:
        if model_path is None:
            return None
        path = Path(model_path)
        if not path.exists():
            print(
                f"[bot] Model checkpoint not found at "
                f"{path} — falling back to random moves."
            )
            return None
        model = build_model(self.encoder_name)
        state = torch.load(path, map_location=self.device)
        # Support raw state_dict and wrapped checkpoints
        # saved by the training pipeline.
        if isinstance(state, dict):
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "model" in state:
                state = state["model"]
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        print(f"[bot] Loaded {self.encoder_name} model from {path}")
        return model

    # ── Evaluation ──────────────────────────────────────

    @torch.no_grad()
    def _evaluate(self, fen: str) -> float:
        """Return model win-probability for *white*.

        Args:
            fen: Full FEN string of the position.

        Returns:
            Float in ``[0, 1]``.
        """
        tensor = self.encoder.encode(fen).unsqueeze(0)
        tensor = tensor.to(self.device)
        prob = self.model(tensor).item()
        return prob

    def select_move(
        self,
        board: chess.Board,
        rng: random.Random | None = None,
        top_n: int = 5,
    ) -> tuple[chess.Move, EvalResult]:
        """Pick the best legal move for the side to move.

        When the model is loaded, every legal move is
        evaluated.  Otherwise a random legal move is
        returned.

        Args:
            board: Current ``python-chess`` Board.
            rng: Optional RNG for tie-breaking / fallback.
            top_n: Number of top moves to include in the
                evaluation result.

        Returns:
            ``(chosen_move, eval_result)``.
        """
        rng = rng or random.Random()
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("No legal moves available.")

        if self.model is None:
            move = rng.choice(legal)
            return move, EvalResult(used_model=False)

        # Evaluate the current position first.
        position_eval = self._evaluate(
            _fen_from_board(board),
        )

        # The model predicts P(white wins).
        # If it is white's turn we maximise; if black's
        # we minimise.
        maximise = board.turn == chess.WHITE

        scored: list[MoveScore] = []
        for move in legal:
            san = board.san(move)
            board.push(move)
            score = self._evaluate(_fen_from_board(board))
            board.pop()
            scored.append(
                MoveScore(
                    move=move,
                    san=san,
                    win_prob=score,
                )
            )

        # Sort by desirability for the side to move.
        scored.sort(
            key=lambda ms: ms.win_prob,
            reverse=maximise,
        )

        if self.temperature <= 0:
            chosen_ms = scored[0]
        else:
            # Stochastic selection via softmax temperature.
            import math

            raw = [ms.win_prob for ms in scored]
            if not maximise:
                raw = [1.0 - v for v in raw]
            max_r = max(raw)
            exp_vals = [math.exp((v - max_r) / self.temperature) for v in raw]
            total = sum(exp_vals)
            probs = [e / total for e in exp_vals]
            idx = rng.choices(range(len(scored)), weights=probs, k=1)[0]
            chosen_ms = scored[idx]

        result = EvalResult(
            position_eval=position_eval,
            chosen=chosen_ms,
            top_moves=scored[:top_n],
            all_moves=scored,
            used_model=True,
        )
        return chosen_ms.move, result

    def evaluate_position(
        self,
        board: chess.Board,
    ) -> float | None:
        """Return P(white wins) for the current position.

        Returns ``None`` when no model is loaded.
        """
        if self.model is None:
            return None
        return self._evaluate(_fen_from_board(board))

    @property
    def has_model(self) -> bool:
        """Whether a trained model is loaded."""
        return self.model is not None
