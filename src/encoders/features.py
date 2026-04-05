"""Board feature computation helpers using python-chess.

Provides functions to compute dynamic move-aware and structural
board features from a ``chess.Board``, returning ``(C, 8, 8)``
float tensors.

Dynamic features (12 planes):
    0   White attack map (binary).
    1   Black attack map (binary).
    2   White defense map (binary — white piece defended by another).
    3   Black defense map (binary — black piece defended by another).
    4   White attack count per square (raw count as float).
    5   Black attack count per square (raw count as float).
    6   White reachability map (legal / pseudo-legal destinations).
    7   Black reachability map (legal / pseudo-legal destinations).
    8   White pinned pieces (binary).
    9   Black pinned pieces (binary).
    10  White king in check (uniform binary plane).
    11  Black king in check (uniform binary plane).

Structural features (6 planes):
    0   White doubled pawns.
    1   Black doubled pawns.
    2   White isolated pawns.
    3   Black isolated pawns.
    4   White passed pawns.
    5   Black passed pawns.
"""

from __future__ import annotations

import chess
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CH_W_ATTACK = 0
_CH_B_ATTACK = 1
_CH_W_DEFENSE = 2
_CH_B_DEFENSE = 3
_CH_W_ATK_COUNT = 4
_CH_B_ATK_COUNT = 5
_CH_W_REACH = 6
_CH_B_REACH = 7
_CH_W_PINNED = 8
_CH_B_PINNED = 9
_CH_W_CHECK = 10
_CH_B_CHECK = 11

NUM_DYNAMIC = 12

_CH_W_DOUBLED = 0
_CH_B_DOUBLED = 1
_CH_W_ISOLATED = 2
_CH_B_ISOLATED = 3
_CH_W_PASSED = 4
_CH_B_PASSED = 5

NUM_STRUCTURAL = 6


def _sq_to_rc(sq: int) -> tuple[int, int]:
    """Convert a python-chess square index to ``(row, col)``.

    Row 0 corresponds to rank 8, row 7 to rank 1 — matching the
    layout used by ``PiecePlaneEncoder``.
    """
    return 7 - chess.square_rank(sq), chess.square_file(sq)


# ---------------------------------------------------------------------------
# Dynamic features
# ---------------------------------------------------------------------------


def _reachability(board: chess.Board, color: chess.Color) -> set[int]:
    """Compute reachable destination squares for *color*.

    For the active side, uses legal moves.  For the non-active
    side, uses a null-move trick; if that is illegal (active
    side in check), falls back to pseudo-legal attacks with
    pawn pushes.
    """
    if board.turn == color:
        return {m.to_square for m in board.legal_moves}

    # Non-active side: null-move trick.
    if not board.is_check():
        board.push(chess.Move.null())
        dests = {m.to_square for m in board.legal_moves}
        board.pop()
        return dests

    # Fallback: active side is in check so null move is illegal.
    dests: set[int] = set()
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.color != color:
            continue
        for target in board.attacks(sq):
            target_piece = board.piece_at(target)
            if target_piece is None or target_piece.color != color:
                dests.add(target)
        if piece.piece_type == chess.PAWN:
            _add_pawn_pushes(board, sq, color, dests)
    return dests


def _add_pawn_pushes(
    board: chess.Board,
    sq: int,
    color: chess.Color,
    dests: set[int],
) -> None:
    """Append forward pawn push squares to *dests* (mutating)."""
    direction = 8 if color == chess.WHITE else -8
    one = sq + direction
    if 0 <= one < 64 and board.piece_at(one) is None:
        dests.add(one)
        rank = chess.square_rank(sq)
        start_rank = 1 if color == chess.WHITE else 6
        if rank == start_rank:
            two = sq + 2 * direction
            if 0 <= two < 64 and board.piece_at(two) is None:
                dests.add(two)


def compute_dynamic_features(board: chess.Board) -> torch.Tensor:
    """Return 12 dynamic move-aware planes as a ``(12, 8, 8)`` tensor.

    See module docstring for the channel layout.

    Args:
        board: A ``chess.Board`` instance.

    Returns:
        ``torch.Tensor`` of shape ``(12, 8, 8)`` and dtype
        ``float32``.
    """
    planes = torch.zeros(NUM_DYNAMIC, 8, 8, dtype=torch.float32)

    # --- Per-square features: attacks, defense, pins ----------------------
    for sq in chess.SQUARES:
        r, c = _sq_to_rc(sq)
        piece = board.piece_at(sq)

        w_attackers = board.attackers(chess.WHITE, sq)
        if w_attackers:
            planes[_CH_W_ATTACK, r, c] = 1.0
            planes[_CH_W_ATK_COUNT, r, c] = float(len(w_attackers))
            if piece is not None and piece.color == chess.WHITE:
                planes[_CH_W_DEFENSE, r, c] = 1.0

        b_attackers = board.attackers(chess.BLACK, sq)
        if b_attackers:
            planes[_CH_B_ATTACK, r, c] = 1.0
            planes[_CH_B_ATK_COUNT, r, c] = float(len(b_attackers))
            if piece is not None and piece.color == chess.BLACK:
                planes[_CH_B_DEFENSE, r, c] = 1.0

        if piece is not None:
            if piece.color == chess.WHITE and board.is_pinned(chess.WHITE, sq):
                planes[_CH_W_PINNED, r, c] = 1.0
            elif piece.color == chess.BLACK and board.is_pinned(chess.BLACK, sq):
                planes[_CH_B_PINNED, r, c] = 1.0

    # --- Reachability maps ------------------------------------------------
    for sq in _reachability(board, chess.WHITE):
        r, c = _sq_to_rc(sq)
        planes[_CH_W_REACH, r, c] = 1.0

    for sq in _reachability(board, chess.BLACK):
        r, c = _sq_to_rc(sq)
        planes[_CH_B_REACH, r, c] = 1.0

    # --- King in check (uniform plane) ------------------------------------
    if board.is_check():
        if board.turn == chess.WHITE:
            planes[_CH_W_CHECK] = 1.0
        else:
            planes[_CH_B_CHECK] = 1.0

    return planes


# ---------------------------------------------------------------------------
# Structural features
# ---------------------------------------------------------------------------


def _mark_isolated(
    pawn_files: dict[int, list[int]],
    pawns: list[int],
    planes: torch.Tensor,
    channel: int,
) -> None:
    """Set *channel* to 1 on squares with isolated pawns (mutating)."""
    for sq in pawns:
        f = chess.square_file(sq)
        has_neighbour = False
        if f > 0 and pawn_files[f - 1]:
            has_neighbour = True
        if not has_neighbour and f < 7 and pawn_files[f + 1]:
            has_neighbour = True
        if not has_neighbour:
            r, c = _sq_to_rc(sq)
            planes[channel, r, c] = 1.0


def _mark_passed(
    pawns: list[int],
    opponent_files: dict[int, list[int]],
    color: chess.Color,
    planes: torch.Tensor,
    channel: int,
) -> None:
    """Set *channel* to 1 on squares with passed pawns (mutating)."""
    for sq in pawns:
        f = chess.square_file(sq)
        rank = chess.square_rank(sq)
        passed = True
        for check_f in range(max(0, f - 1), min(8, f + 2)):
            for opp_sq in opponent_files[check_f]:
                opp_rank = chess.square_rank(opp_sq)
                if color == chess.WHITE and opp_rank > rank:
                    passed = False
                    break
                elif color == chess.BLACK and opp_rank < rank:
                    passed = False
                    break
            if not passed:
                break
        if passed:
            r, c = _sq_to_rc(sq)
            planes[channel, r, c] = 1.0


def compute_structural_features(board: chess.Board) -> torch.Tensor:
    """Return 6 structural pawn planes as a ``(6, 8, 8)`` tensor.

    See module docstring for the channel layout.

    Args:
        board: A ``chess.Board`` instance.

    Returns:
        ``torch.Tensor`` of shape ``(6, 8, 8)`` and dtype
        ``float32``.
    """
    planes = torch.zeros(NUM_STRUCTURAL, 8, 8, dtype=torch.float32)

    # Collect pawn positions by colour and file.
    white_pawns: list[int] = []
    black_pawns: list[int] = []
    white_pawn_files: dict[int, list[int]] = {f: [] for f in range(8)}
    black_pawn_files: dict[int, list[int]] = {f: [] for f in range(8)}

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None and piece.piece_type == chess.PAWN:
            f = chess.square_file(sq)
            if piece.color == chess.WHITE:
                white_pawns.append(sq)
                white_pawn_files[f].append(sq)
            else:
                black_pawns.append(sq)
                black_pawn_files[f].append(sq)

    # Doubled pawns: more than one friendly pawn on the same file.
    for f in range(8):
        if len(white_pawn_files[f]) > 1:
            for sq in white_pawn_files[f]:
                r, c = _sq_to_rc(sq)
                planes[_CH_W_DOUBLED, r, c] = 1.0
        if len(black_pawn_files[f]) > 1:
            for sq in black_pawn_files[f]:
                r, c = _sq_to_rc(sq)
                planes[_CH_B_DOUBLED, r, c] = 1.0

    # Isolated pawns: no friendly pawn on adjacent files.
    _mark_isolated(white_pawn_files, white_pawns, planes, _CH_W_ISOLATED)
    _mark_isolated(black_pawn_files, black_pawns, planes, _CH_B_ISOLATED)

    # Passed pawns: no opposing pawn ahead on same or adjacent files.
    _mark_passed(white_pawns, black_pawn_files, chess.WHITE, planes, _CH_W_PASSED)
    _mark_passed(black_pawns, white_pawn_files, chess.BLACK, planes, _CH_B_PASSED)

    return planes
