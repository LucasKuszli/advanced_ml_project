"""Tests for board encoders and feature computation."""

from __future__ import annotations

import chess
import pytest
import torch

from src.encoders.base import BoardEncoder
from src.encoders.enriched_piece_plane import (
    DynamicPiecePlaneEncoder,
    FullPiecePlaneEncoder,
)
from src.encoders.features import (
    NUM_DYNAMIC,
    NUM_STRUCTURAL,
    compute_dynamic_features,
    compute_structural_features,
)
from src.encoders.piece_plane import PiecePlaneEncoder

# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# 1. e4
AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

# Black in check (Scholar's-mate approach: Qxf7+).
CHECK_FEN = "r1bqkbnr/pppp1Qpp/2n5/4p3/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 0 3"

# Nc3 pinned by Bb4 to the white king on e1.
PIN_FEN = "rnbqk1nr/pppp1ppp/4p3/8/1b6/2NP4/PPP1PPPP/R1BQKBNR w KQkq - 2 3"

# White doubled d-pawns (d4 + d5).
DOUBLED_FEN = "rnbqkb1r/ppp1pppp/5n2/3P4/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3"

# No castling rights, no en passant, halfmove = 50.
ENDGAME_FEN = "8/5k2/8/8/8/3K4/8/8 w - - 50 80"

# Isolated white a-pawn, passed black a-pawn.
PAWN_STRUCTURE_FEN = "8/8/8/p7/8/8/P7/8 w - - 0 1"


@pytest.fixture
def base_encoder() -> PiecePlaneEncoder:
    return PiecePlaneEncoder()


@pytest.fixture
def dynamic_encoder() -> DynamicPiecePlaneEncoder:
    return DynamicPiecePlaneEncoder()


@pytest.fixture
def full_encoder() -> FullPiecePlaneEncoder:
    return FullPiecePlaneEncoder()


# ──────────────────────────────────────────────────────────────
# PiecePlaneEncoder
# ──────────────────────────────────────────────────────────────


class TestPiecePlaneEncoder:
    def test_output_shape(self, base_encoder: PiecePlaneEncoder) -> None:
        t = base_encoder.encode(STARTING_FEN)
        assert t.shape == (19, 8, 8)

    def test_output_shape_matches_property(
        self, base_encoder: PiecePlaneEncoder
    ) -> None:
        assert base_encoder.output_shape == (19, 8, 8)

    def test_dtype(self, base_encoder: PiecePlaneEncoder) -> None:
        t = base_encoder.encode(STARTING_FEN)
        assert t.dtype == torch.float32

    def test_piece_planes_start(self, base_encoder: PiecePlaneEncoder) -> None:
        """White pawns on rank 2 in the starting position."""
        t = base_encoder.encode(STARTING_FEN)
        # Channel 0 = P (white pawn). Rank 2 → row 6.
        assert t[0, 6, :].sum().item() == 8.0
        # No white pawns elsewhere.
        assert t[0, :6, :].sum().item() == 0.0
        assert t[0, 7, :].sum().item() == 0.0

    def test_black_pawns_start(self, base_encoder: PiecePlaneEncoder) -> None:
        t = base_encoder.encode(STARTING_FEN)
        # Channel 6 = p (black pawn). Rank 7 → row 1.
        assert t[6, 1, :].sum().item() == 8.0

    def test_side_to_move_white(self, base_encoder: PiecePlaneEncoder) -> None:
        t = base_encoder.encode(STARTING_FEN)
        # Channel 12 should be all 1s for white to move.
        assert t[12].sum().item() == 64.0

    def test_side_to_move_black(self, base_encoder: PiecePlaneEncoder) -> None:
        t = base_encoder.encode(AFTER_E4)
        assert t[12].sum().item() == 0.0

    def test_castling_all(self, base_encoder: PiecePlaneEncoder) -> None:
        t = base_encoder.encode(STARTING_FEN)
        for i in range(4):
            assert t[13 + i].sum().item() == 64.0

    def test_castling_none(self, base_encoder: PiecePlaneEncoder) -> None:
        t = base_encoder.encode(ENDGAME_FEN)
        for i in range(4):
            assert t[13 + i].sum().item() == 0.0

    def test_en_passant(self, base_encoder: PiecePlaneEncoder) -> None:
        t = base_encoder.encode(AFTER_E4)
        # EP target = e3 → row 5 (rank 3), col 4 (file e).
        assert t[17, 5, 4].item() == 1.0
        # Only that one square should be set.
        assert t[17].sum().item() == 1.0

    def test_no_en_passant(self, base_encoder: PiecePlaneEncoder) -> None:
        t = base_encoder.encode(STARTING_FEN)
        assert t[17].sum().item() == 0.0

    def test_halfmove_clock_zero(self, base_encoder: PiecePlaneEncoder) -> None:
        t = base_encoder.encode(STARTING_FEN)
        assert t[18].sum().item() == 0.0

    def test_halfmove_clock_normalized(self, base_encoder: PiecePlaneEncoder) -> None:
        t = base_encoder.encode(ENDGAME_FEN)
        # halfmove=50, normalised = 50/100 = 0.5, uniform plane.
        assert t[18, 0, 0].item() == pytest.approx(0.5)

    def test_callable(self, base_encoder: PiecePlaneEncoder) -> None:
        t = base_encoder(STARTING_FEN)
        assert t.shape == (19, 8, 8)

    def test_is_board_encoder(self, base_encoder: PiecePlaneEncoder) -> None:
        assert isinstance(base_encoder, BoardEncoder)

    def test_repr(self, base_encoder: PiecePlaneEncoder) -> None:
        assert "PiecePlaneEncoder" in repr(base_encoder)

    def test_piece_placement_e4(self, base_encoder: PiecePlaneEncoder) -> None:
        """After 1. e4, white pawn should be on e4, not e2."""
        t = base_encoder.encode(AFTER_E4)
        # e4 → row 4, col 4.
        assert t[0, 4, 4].item() == 1.0
        # e2 (row 6, col 4) should be empty for white pawns.
        assert t[0, 6, 4].item() == 0.0


# ──────────────────────────────────────────────────────────────
# compute_dynamic_features
# ──────────────────────────────────────────────────────────────


class TestDynamicFeatures:
    def test_shape(self) -> None:
        board = chess.Board(STARTING_FEN)
        t = compute_dynamic_features(board)
        assert t.shape == (NUM_DYNAMIC, 8, 8)

    def test_dtype(self) -> None:
        board = chess.Board(STARTING_FEN)
        t = compute_dynamic_features(board)
        assert t.dtype == torch.float32

    def test_white_attacks_starting(self) -> None:
        """In the starting position, white attacks the 3rd rank
        (pawn attacks) and some squares via knights."""
        board = chess.Board(STARTING_FEN)
        t = compute_dynamic_features(board)
        # Channel 0 = white attack map. Should have nonzero entries.
        assert t[0].sum().item() > 0

    def test_black_attacks_starting(self) -> None:
        board = chess.Board(STARTING_FEN)
        t = compute_dynamic_features(board)
        assert t[1].sum().item() > 0

    def test_attack_count_positive(self) -> None:
        board = chess.Board(STARTING_FEN)
        t = compute_dynamic_features(board)
        # Attack counts should be >= 0 everywhere.
        assert (t[4] >= 0).all()
        assert (t[5] >= 0).all()
        # At least some squares should have count > 0.
        assert t[4].sum().item() > 0
        assert t[5].sum().item() > 0

    def test_defense_starting(self) -> None:
        """Knights and rooks are defended in the starting position."""
        board = chess.Board(STARTING_FEN)
        t = compute_dynamic_features(board)
        # White defense (ch 2): at least pawns defend each other's
        # adjacent square, and back-rank pieces defend neighbours.
        assert t[2].sum().item() > 0
        assert t[3].sum().item() > 0

    def test_reachability_starting(self) -> None:
        """White has 20 legal moves but only 16 unique destination
        squares (knight destinations overlap with pawn pushes)."""
        board = chess.Board(STARTING_FEN)
        t = compute_dynamic_features(board)
        assert t[6].sum().item() == 16.0

    def test_reachability_black_starting(self) -> None:
        """Black (non-active) should also have 16 unique destination
        squares via the null-move trick."""
        board = chess.Board(STARTING_FEN)
        t = compute_dynamic_features(board)
        assert t[7].sum().item() == 16.0

    def test_no_pins_starting(self) -> None:
        board = chess.Board(STARTING_FEN)
        t = compute_dynamic_features(board)
        assert t[8].sum().item() == 0.0
        assert t[9].sum().item() == 0.0

    def test_pin_detected(self) -> None:
        """Nc3 is pinned by Bb4 to the white king."""
        board = chess.Board(PIN_FEN)
        t = compute_dynamic_features(board)
        # White pinned (ch 8): Nc3 → row 5, col 2.
        assert t[8, 5, 2].item() == 1.0
        assert t[8].sum().item() >= 1.0

    def test_no_check_starting(self) -> None:
        board = chess.Board(STARTING_FEN)
        t = compute_dynamic_features(board)
        assert t[10].sum().item() == 0.0
        assert t[11].sum().item() == 0.0

    def test_black_in_check(self) -> None:
        """Black king is in check from Qf7."""
        board = chess.Board(CHECK_FEN)
        t = compute_dynamic_features(board)
        # Black check (ch 11) should be a uniform 1.0 plane.
        assert t[11].sum().item() == 64.0
        # White check (ch 10) should be zero.
        assert t[10].sum().item() == 0.0

    def test_white_in_check(self) -> None:
        """Construct a position where white is in check."""
        board = chess.Board()
        # Play moves to get white in check.
        board.push_san("f3")
        board.push_san("e5")
        board.push_san("g4")
        board.push_san("Qh4")  # Fool's mate check
        assert board.is_check()
        assert board.turn == chess.WHITE
        t = compute_dynamic_features(board)
        assert t[10].sum().item() == 64.0
        assert t[11].sum().item() == 0.0

    def test_reachability_while_in_check(self) -> None:
        """When active side is in check, non-active reachability
        should still work (fallback path)."""
        board = chess.Board(CHECK_FEN)
        assert board.is_check()
        t = compute_dynamic_features(board)
        # Black is in check and active → limited legal moves.
        # White (non-active) reachability should still be computed.
        assert t[6].sum().item() > 0  # White reachability


# ──────────────────────────────────────────────────────────────
# compute_structural_features
# ──────────────────────────────────────────────────────────────


class TestStructuralFeatures:
    def test_shape(self) -> None:
        board = chess.Board(STARTING_FEN)
        t = compute_structural_features(board)
        assert t.shape == (NUM_STRUCTURAL, 8, 8)

    def test_dtype(self) -> None:
        board = chess.Board(STARTING_FEN)
        t = compute_structural_features(board)
        assert t.dtype == torch.float32

    def test_no_doubled_starting(self) -> None:
        board = chess.Board(STARTING_FEN)
        t = compute_structural_features(board)
        assert t[0].sum().item() == 0.0
        assert t[1].sum().item() == 0.0

    def test_no_isolated_starting(self) -> None:
        board = chess.Board(STARTING_FEN)
        t = compute_structural_features(board)
        assert t[2].sum().item() == 0.0
        assert t[3].sum().item() == 0.0

    def test_no_passed_starting(self) -> None:
        board = chess.Board(STARTING_FEN)
        t = compute_structural_features(board)
        assert t[4].sum().item() == 0.0
        assert t[5].sum().item() == 0.0

    def test_doubled_pawns(self) -> None:
        """White has doubled d-pawns on d4 and d5."""
        board = chess.Board(DOUBLED_FEN)
        t = compute_structural_features(board)
        # White doubled (ch 0): d4 → (4,3), d5 → (3,3).
        assert t[0, 4, 3].item() == 1.0
        assert t[0, 3, 3].item() == 1.0
        assert t[0].sum().item() == 2.0

    def test_isolated_pawn(self) -> None:
        """White a-pawn with no neighbours is isolated."""
        board = chess.Board(PAWN_STRUCTURE_FEN)
        t = compute_structural_features(board)
        # White isolated (ch 2): a2 → (6,0).
        assert t[2, 6, 0].item() == 1.0
        # Black isolated (ch 3): a5 → (3,0).
        assert t[3, 3, 0].item() == 1.0

    def test_passed_pawn(self) -> None:
        """White a-pawn on a2 vs black a-pawn on a5: neither is
        passed (they block each other on the same file)."""
        board = chess.Board(PAWN_STRUCTURE_FEN)
        t = compute_structural_features(board)
        # Neither pawn is passed since they share the same file.
        assert t[4].sum().item() == 0.0
        assert t[5].sum().item() == 0.0

    def test_actual_passed_pawn(self) -> None:
        """White pawn on e5, no black pawns on d/e/f files."""
        fen = "8/8/8/4P3/8/8/8/8 w - - 0 1"
        board = chess.Board(fen)
        t = compute_structural_features(board)
        # e5 → (3, 4).
        assert t[4, 3, 4].item() == 1.0

    def test_not_isolated_adjacent(self) -> None:
        """Pawns on adjacent files are not isolated."""
        fen = "8/8/8/8/8/8/PP6/8 w - - 0 1"
        board = chess.Board(fen)
        t = compute_structural_features(board)
        assert t[2].sum().item() == 0.0

    def test_empty_board(self) -> None:
        """No pawns → all structural planes are zero."""
        board = chess.Board(ENDGAME_FEN)
        t = compute_structural_features(board)
        assert t.sum().item() == 0.0


# ──────────────────────────────────────────────────────────────
# DynamicPiecePlaneEncoder
# ──────────────────────────────────────────────────────────────


class TestDynamicPiecePlaneEncoder:
    def test_output_shape(self, dynamic_encoder: DynamicPiecePlaneEncoder) -> None:
        t = dynamic_encoder.encode(STARTING_FEN)
        assert t.shape == (31, 8, 8)

    def test_output_shape_matches_property(
        self, dynamic_encoder: DynamicPiecePlaneEncoder
    ) -> None:
        assert dynamic_encoder.output_shape == (31, 8, 8)

    def test_dtype(self, dynamic_encoder: DynamicPiecePlaneEncoder) -> None:
        t = dynamic_encoder.encode(STARTING_FEN)
        assert t.dtype == torch.float32

    def test_base_channels_preserved(
        self,
        base_encoder: PiecePlaneEncoder,
        dynamic_encoder: DynamicPiecePlaneEncoder,
    ) -> None:
        base = base_encoder.encode(STARTING_FEN)
        dyn = dynamic_encoder.encode(STARTING_FEN)
        assert torch.equal(base, dyn[:19])

    def test_base_channels_preserved_complex(
        self,
        base_encoder: PiecePlaneEncoder,
        dynamic_encoder: DynamicPiecePlaneEncoder,
    ) -> None:
        """Verify base channels match on a non-starting position."""
        base = base_encoder.encode(CHECK_FEN)
        dyn = dynamic_encoder.encode(CHECK_FEN)
        assert torch.equal(base, dyn[:19])

    def test_dynamic_channels_nonzero(
        self, dynamic_encoder: DynamicPiecePlaneEncoder
    ) -> None:
        t = dynamic_encoder.encode(STARTING_FEN)
        assert t[19:31].sum().item() > 0

    def test_is_board_encoder(self, dynamic_encoder: DynamicPiecePlaneEncoder) -> None:
        assert isinstance(dynamic_encoder, BoardEncoder)

    def test_callable(self, dynamic_encoder: DynamicPiecePlaneEncoder) -> None:
        t = dynamic_encoder(STARTING_FEN)
        assert t.shape == (31, 8, 8)

    def test_check_plane(self, dynamic_encoder: DynamicPiecePlaneEncoder) -> None:
        t = dynamic_encoder.encode(CHECK_FEN)
        # Channel 30 = black king in check.
        assert t[30].sum().item() == 64.0
        assert t[29].sum().item() == 0.0

    def test_pin_plane(self, dynamic_encoder: DynamicPiecePlaneEncoder) -> None:
        t = dynamic_encoder.encode(PIN_FEN)
        # Channel 27 = white pinned. Nc3 → (5, 2).
        assert t[27, 5, 2].item() == 1.0


# ──────────────────────────────────────────────────────────────
# FullPiecePlaneEncoder
# ──────────────────────────────────────────────────────────────


class TestFullPiecePlaneEncoder:
    def test_output_shape(self, full_encoder: FullPiecePlaneEncoder) -> None:
        t = full_encoder.encode(STARTING_FEN)
        assert t.shape == (37, 8, 8)

    def test_output_shape_matches_property(
        self, full_encoder: FullPiecePlaneEncoder
    ) -> None:
        assert full_encoder.output_shape == (37, 8, 8)

    def test_dtype(self, full_encoder: FullPiecePlaneEncoder) -> None:
        t = full_encoder.encode(STARTING_FEN)
        assert t.dtype == torch.float32

    def test_base_channels_preserved(
        self,
        base_encoder: PiecePlaneEncoder,
        full_encoder: FullPiecePlaneEncoder,
    ) -> None:
        base = base_encoder.encode(STARTING_FEN)
        full = full_encoder.encode(STARTING_FEN)
        assert torch.equal(base, full[:19])

    def test_dynamic_channels_match(
        self,
        dynamic_encoder: DynamicPiecePlaneEncoder,
        full_encoder: FullPiecePlaneEncoder,
    ) -> None:
        dyn = dynamic_encoder.encode(STARTING_FEN)
        full = full_encoder.encode(STARTING_FEN)
        assert torch.equal(dyn[19:31], full[19:31])

    def test_dynamic_channels_match_complex(
        self,
        dynamic_encoder: DynamicPiecePlaneEncoder,
        full_encoder: FullPiecePlaneEncoder,
    ) -> None:
        dyn = dynamic_encoder.encode(DOUBLED_FEN)
        full = full_encoder.encode(DOUBLED_FEN)
        assert torch.equal(dyn[19:31], full[19:31])

    def test_structural_channels_start_zero(
        self, full_encoder: FullPiecePlaneEncoder
    ) -> None:
        """No structural features in the starting position."""
        t = full_encoder.encode(STARTING_FEN)
        assert t[31:37].sum().item() == 0.0

    def test_doubled_pawns_in_full(self, full_encoder: FullPiecePlaneEncoder) -> None:
        t = full_encoder.encode(DOUBLED_FEN)
        # Channel 31 = white doubled. d4 and d5.
        assert t[31].sum().item() == 2.0

    def test_is_board_encoder(self, full_encoder: FullPiecePlaneEncoder) -> None:
        assert isinstance(full_encoder, BoardEncoder)

    def test_callable(self, full_encoder: FullPiecePlaneEncoder) -> None:
        t = full_encoder(STARTING_FEN)
        assert t.shape == (37, 8, 8)

    def test_endgame_no_structural(self, full_encoder: FullPiecePlaneEncoder) -> None:
        t = full_encoder.encode(ENDGAME_FEN)
        assert t[31:37].sum().item() == 0.0

    def test_repr(self, full_encoder: FullPiecePlaneEncoder) -> None:
        assert "FullPiecePlaneEncoder" in repr(full_encoder)
