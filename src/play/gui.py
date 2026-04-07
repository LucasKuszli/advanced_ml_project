"""Graphical chess game — play against your trained model.

Uses ``pygame`` to render an interactive chess board where
you click to select and move your pieces.  The bot
automatically responds with its own moves.

Usage:
    uv run python -m src.play
    uv run python -m src.play --color black
    uv run python -m src.play --model artifacts/models/piece_plane_best.pt
    uv run python -m src.play --openings
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import pygame

import chess
from src.play.bot import ChessBot, EvalResult
from src.play.game import Game, GamePhase, PlayerColor

# ── Colours ─────────────────────────────────────────────

LIGHT_SQ = (240, 217, 181)
DARK_SQ = (181, 136, 99)
HIGHLIGHT_FROM = (186, 202, 68)  # selected square
HIGHLIGHT_TO = (246, 246, 105)  # legal destination
HIGHLIGHT_LAST_FROM = (205, 210, 106)  # last move origin
HIGHLIGHT_LAST_TO = (170, 162, 58)  # last move dest
CHECK_COLOUR = (235, 97, 80)  # king in check
BG_COLOUR = (48, 46, 43)
PANEL_BG = (39, 37, 34)
TEXT_COLOUR = (200, 200, 200)
TEXT_DIM = (140, 140, 140)
BUTTON_BG = (80, 76, 72)
BUTTON_HOVER = (100, 96, 92)
BUTTON_TEXT = (220, 220, 220)

# ── Layout defaults ──────────────────────────────────────

DEFAULT_SQ_SIZE = 80
DEFAULT_PANEL_W = 280
MIN_SQ_SIZE = 40  # smallest readable square
MIN_PANEL_W = 200

# ── Piece glyphs (fallback when no images are available) ─

_PIECE_UNICODE: dict[int, dict[int, str]] = {
    chess.PAWN: {chess.WHITE: "♙", chess.BLACK: "♟"},
    chess.KNIGHT: {chess.WHITE: "♘", chess.BLACK: "♞"},
    chess.BISHOP: {chess.WHITE: "♗", chess.BLACK: "♝"},
    chess.ROOK: {chess.WHITE: "♖", chess.BLACK: "♜"},
    chess.QUEEN: {chess.WHITE: "♕", chess.BLACK: "♛"},
    chess.KING: {chess.WHITE: "♔", chess.BLACK: "♚"},
}

# ── Helpers ─────────────────────────────────────────────


def _sq_to_pixel(
    sq: int,
    flipped: bool,
    sq_size: int,
) -> tuple[int, int]:
    """Convert a chess square index to pixel coordinates of
    the top-left corner."""
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)
    if flipped:
        col = 7 - file
        row = rank
    else:
        col = file
        row = 7 - rank
    return col * sq_size, row * sq_size


def _pixel_to_sq(
    x: int,
    y: int,
    flipped: bool,
    sq_size: int,
    board_px: int,
) -> int | None:
    """Convert pixel position to a chess square index.
    Returns None if outside the board."""
    if x < 0 or x >= board_px or y < 0 or y >= board_px:
        return None
    col = x // sq_size
    row = y // sq_size
    if flipped:
        file = 7 - col
        rank = row
    else:
        file = col
        rank = 7 - row
    return chess.square(file, rank)


# ── Piece renderer ──────────────────────────────────────

# Ordered list of TrueType font paths known to contain
# chess piece glyphs (U+2654–U+265F).  We load the file
# directly with ``pygame.font.Font`` so we don't depend on
# the unreliable ``SysFont`` name lookup.
_CHESS_FONT_PATHS: list[str] = [
    # Linux (Debian / Ubuntu).
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    # macOS.
    "/System/Library/Fonts/Apple Symbols.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    # Windows.
    "C:/Windows/Fonts/seguisym.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


class PieceRenderer:
    """Renders chess pieces as text glyphs on a
    ``pygame.Surface``."""

    def __init__(self, sq_size: int) -> None:
        self.sq_size = sq_size
        self.font_size = int(sq_size * 0.78)
        self._font: pygame.font.Font | None = None
        self._cache: dict[
            tuple[int, int, tuple[int, int, int]],
            pygame.Surface,
        ] = {}

    def resize(self, sq_size: int) -> None:
        """Rebuild the renderer for a new square size."""
        if sq_size == self.sq_size:
            return
        self.sq_size = sq_size
        self.font_size = int(sq_size * 0.78)
        self._font = None
        self._cache.clear()

    @property
    def font(self) -> pygame.font.Font:
        if self._font is None:
            pygame.font.init()
            # First try loading a known TTF file directly.
            for path in _CHESS_FONT_PATHS:
                try:
                    f = pygame.font.Font(path, self.font_size)
                    # Verify the font actually has the
                    # chess king glyph (♔ = U+2654).
                    test = f.render("♔", True, (0, 0, 0))
                    if test.get_width() > 5:
                        self._font = f
                        break
                except (FileNotFoundError, OSError):
                    continue
            # Fallback to SysFont if no file worked.
            if self._font is None:
                for name in [
                    "DejaVu Sans",
                    "Noto Sans Symbols2",
                    "Segoe UI Symbol",
                    "symbola",
                ]:
                    try:
                        f = pygame.font.SysFont(
                            name,
                            self.font_size,
                        )
                        test = f.render("♔", True, (0, 0, 0))
                        if test.get_width() > 5:
                            self._font = f
                            break
                    except Exception:
                        continue
            # Last resort: default font.
            if self._font is None:
                self._font = pygame.font.Font(
                    None,
                    self.font_size,
                )
        return self._font

    def _render_glyph(
        self,
        piece: chess.Piece,
        color: tuple[int, int, int],
    ) -> pygame.Surface:
        """Render a piece glyph with caching."""
        key = (piece.piece_type, piece.color, color)
        if key not in self._cache:
            glyph = _PIECE_UNICODE.get(
                piece.piece_type,
                {},
            ).get(piece.color, "?")
            self._cache[key] = self.font.render(
                glyph,
                True,
                color,
            )
        return self._cache[key]

    def draw(
        self,
        surface: pygame.Surface,
        piece: chess.Piece,
        x: int,
        y: int,
    ) -> None:
        """Draw a piece centred in the square at (x, y)."""
        # Render with outline for visibility on any
        # background.
        txt = self._render_glyph(piece, (255, 255, 255))
        outline = self._render_glyph(piece, (0, 0, 0))
        cx = x + (self.sq_size - txt.get_width()) // 2
        cy = y + (self.sq_size - txt.get_height()) // 2
        # Draw outline (offset in 8 directions).
        for dx, dy in [
            (-2, 0),
            (2, 0),
            (0, -2),
            (0, 2),
            (-1, -1),
            (1, -1),
            (-1, 1),
            (1, 1),
        ]:
            surface.blit(outline, (cx + dx, cy + dy))
        # White pieces → white fill, black pieces → dark.
        if piece.color == chess.WHITE:
            fill = self._render_glyph(
                piece,
                (255, 255, 255),
            )
        else:
            fill = self._render_glyph(
                piece,
                (40, 40, 40),
            )
        surface.blit(fill, (cx, cy))


# ── Button widget ───────────────────────────────────────


class Button:
    """Simple clickable button."""

    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        font: pygame.font.Font,
    ) -> None:
        self.rect = rect
        self.text = text
        self.font = font

    def draw(
        self,
        surface: pygame.Surface,
        mouse_pos: tuple[int, int],
    ) -> None:
        hovered = self.rect.collidepoint(mouse_pos)
        colour = BUTTON_HOVER if hovered else BUTTON_BG
        pygame.draw.rect(surface, colour, self.rect, border_radius=6)
        pygame.draw.rect(surface, TEXT_DIM, self.rect, 1, border_radius=6)
        txt = self.font.render(
            self.text,
            True,
            BUTTON_TEXT,
        )
        surface.blit(
            txt,
            (
                self.rect.centerx - txt.get_width() // 2,
                self.rect.centery - txt.get_height() // 2,
            ),
        )

    def is_clicked(
        self,
        pos: tuple[int, int],
    ) -> bool:
        return self.rect.collidepoint(pos)


# ── Main GUI ────────────────────────────────────────────


class ChessGUI:
    """Interactive pygame chess interface.

    Args:
        game: A ``Game`` instance (with bot already set).
        flipped: If ``True`` the board is shown from
            black's perspective.
    """

    def __init__(
        self,
        game: Game,
        flipped: bool = False,
    ) -> None:
        self.game = game
        self.flipped = flipped

        pygame.init()
        pygame.display.set_caption("♚ Chess — vs Model")

        # Dynamic layout dimensions.
        self.sq_size = DEFAULT_SQ_SIZE
        self.board_px = self.sq_size * 8
        self.panel_w = DEFAULT_PANEL_W
        self.window_w = self.board_px + self.panel_w
        self.window_h = self.board_px

        self.screen = pygame.display.set_mode(
            (self.window_w, self.window_h),
            pygame.RESIZABLE,
        )
        self.clock = pygame.time.Clock()
        self.piece_renderer = PieceRenderer(self.sq_size)

        # Fonts (scaled to current layout).
        self._rebuild_fonts()

        # Interaction state.
        self.selected_sq: int | None = None
        self.legal_dests: list[int] = []
        self.last_move: chess.Move | None = None
        self.dragging: bool = False
        self.drag_piece: chess.Piece | None = None
        self.drag_origin: int | None = None
        self.drag_pos: tuple[int, int] = (0, 0)
        self.status_msg: str = ""
        self.bot_thinking: bool = False
        self.promotion_pending: dict | None = None

        # Model evaluation state.
        self.last_eval: EvalResult | None = None
        self.position_eval: float | None = None

        # Buttons (positioned by _update_buttons).
        self.btn_new_game = Button(
            pygame.Rect(0, 0, 0, 0),
            "New Game",
            self.font_md,
        )
        self.btn_resign = Button(
            pygame.Rect(0, 0, 0, 0),
            "Resign",
            self.font_md,
        )
        self.btn_undo = Button(
            pygame.Rect(0, 0, 0, 0),
            "Undo Move",
            self.font_md,
        )
        self._update_buttons()

    # ── Layout helpers ──────────────────────────────────

    def _rebuild_fonts(self) -> None:
        """Rebuild font objects scaled to current layout."""
        scale = self.sq_size / DEFAULT_SQ_SIZE
        self.font_sm = pygame.font.SysFont(
            "",
            max(12, int(20 * scale)),
        )
        self.font_md = pygame.font.SysFont(
            "",
            max(14, int(26 * scale)),
        )
        self.font_lg = pygame.font.SysFont(
            "",
            max(16, int(34 * scale)),
        )
        self.font_coord = pygame.font.SysFont(
            "",
            max(10, int(18 * scale)),
        )

    def _update_buttons(self) -> None:
        """Reposition buttons based on current layout."""
        bx = self.board_px + 20
        bw = self.panel_w - 40
        btn_h = max(30, int(40 * self.sq_size / DEFAULT_SQ_SIZE))
        gap = max(8, int(10 * self.sq_size / DEFAULT_SQ_SIZE))
        self.btn_undo.rect = pygame.Rect(
            bx,
            self.window_h - 3 * btn_h - 3 * gap,
            bw,
            btn_h,
        )
        self.btn_new_game.rect = pygame.Rect(
            bx,
            self.window_h - 2 * btn_h - 2 * gap,
            bw,
            btn_h,
        )
        self.btn_resign.rect = pygame.Rect(
            bx,
            self.window_h - btn_h - gap,
            bw,
            btn_h,
        )
        # Update button fonts.
        self.btn_new_game.font = self.font_md
        self.btn_resign.font = self.font_md
        self.btn_undo.font = self.font_md

    def _handle_resize(self, width: int, height: int) -> None:
        """Recompute layout dimensions after a window
        resize."""
        min_w = MIN_SQ_SIZE * 8 + MIN_PANEL_W
        min_h = MIN_SQ_SIZE * 8
        width = max(width, min_w)
        height = max(height, min_h)

        # Square size determined by window height.
        self.sq_size = height // 8
        self.board_px = self.sq_size * 8
        # Panel gets remaining horizontal space.
        self.panel_w = max(MIN_PANEL_W, width - self.board_px)
        self.window_w = self.board_px + self.panel_w
        self.window_h = self.board_px

        self.screen = pygame.display.set_mode(
            (self.window_w, self.window_h),
            pygame.RESIZABLE,
        )
        self.piece_renderer.resize(self.sq_size)
        self._rebuild_fonts()
        self._update_buttons()

    # ── Drawing ─────────────────────────────────────────

    def _draw_board(self) -> None:
        """Draw the 8×8 board with highlights."""
        board = self.game.state.board
        sq_size = self.sq_size
        board_px = self.board_px
        in_check_sq: int | None = None
        if board.is_check():
            in_check_sq = board.king(board.turn)

        for sq in chess.SQUARES:
            px, py = _sq_to_pixel(sq, self.flipped, sq_size)
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            is_light = (file + rank) % 2 != 0

            # Base colour.
            colour = LIGHT_SQ if is_light else DARK_SQ

            # Last move highlight.
            if self.last_move is not None:
                if sq == self.last_move.from_square:
                    colour = HIGHLIGHT_LAST_FROM
                elif sq == self.last_move.to_square:
                    colour = HIGHLIGHT_LAST_TO

            # Selected square.
            if sq == self.selected_sq:
                colour = HIGHLIGHT_FROM

            # King in check.
            if sq == in_check_sq:
                colour = CHECK_COLOUR

            pygame.draw.rect(
                self.screen,
                colour,
                (px, py, sq_size, sq_size),
            )

            # Legal move dots.
            if sq in self.legal_dests:
                piece_on_sq = board.piece_at(sq)
                cx = px + sq_size // 2
                cy = py + sq_size // 2
                if piece_on_sq is not None:
                    # Capture: draw ring.
                    pygame.draw.circle(
                        self.screen,
                        HIGHLIGHT_TO,
                        (cx, cy),
                        sq_size // 2,
                        max(2, sq_size // 16),
                    )
                else:
                    # Quiet move: draw dot.
                    pygame.draw.circle(
                        self.screen,
                        HIGHLIGHT_TO,
                        (cx, cy),
                        sq_size // 6,
                    )

        # Coordinates.
        for i in range(8):
            # Rank labels (left edge).
            rank_idx = i if self.flipped else 7 - i
            label = self.font_coord.render(
                str(rank_idx + 1),
                True,
                TEXT_DIM,
            )
            self.screen.blit(label, (3, i * sq_size + 3))
            # File labels (bottom edge).
            file_idx = 7 - i if self.flipped else i
            label = self.font_coord.render(
                chr(ord("a") + file_idx),
                True,
                TEXT_DIM,
            )
            self.screen.blit(
                label,
                (
                    i * sq_size + sq_size - label.get_width() - 2,
                    board_px - label.get_height() - 2,
                ),
            )

    def _draw_pieces(self) -> None:
        """Draw all pieces on the board."""
        board = self.game.state.board
        sq_size = self.sq_size
        for sq in chess.SQUARES:
            if sq == self.drag_origin and self.dragging:
                continue  # piece is being dragged
            piece = board.piece_at(sq)
            if piece is not None:
                px, py = _sq_to_pixel(sq, self.flipped, sq_size)
                self.piece_renderer.draw(
                    self.screen,
                    piece,
                    px,
                    py,
                )

        # Draw dragged piece under cursor.
        if self.dragging and self.drag_piece is not None:
            dx = self.drag_pos[0] - sq_size // 2
            dy = self.drag_pos[1] - sq_size // 2
            self.piece_renderer.draw(
                self.screen,
                self.drag_piece,
                dx,
                dy,
            )

    def _draw_promotion_dialog(self) -> None:
        """Draw promotion piece selection overlay."""
        if self.promotion_pending is None:
            return
        sq_size = self.sq_size
        board_px = self.board_px
        # Dim the board.
        overlay = pygame.Surface(
            (board_px, board_px),
            pygame.SRCALPHA,
        )
        overlay.fill((0, 0, 0, 140))
        self.screen.blit(overlay, (0, 0))

        color = self.promotion_pending["color"]
        pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        box_w = sq_size * 4 + 20
        box_h = sq_size + 40
        bx = (board_px - box_w) // 2
        by = (board_px - box_h) // 2

        # Background box.
        pygame.draw.rect(
            self.screen,
            PANEL_BG,
            (bx, by, box_w, box_h),
            border_radius=10,
        )
        pygame.draw.rect(
            self.screen,
            TEXT_DIM,
            (bx, by, box_w, box_h),
            2,
            border_radius=10,
        )
        # Title.
        title = self.font_md.render(
            "Promote to:",
            True,
            TEXT_COLOUR,
        )
        self.screen.blit(
            title,
            (bx + (box_w - title.get_width()) // 2, by + 5),
        )

        # Piece options.
        self.promotion_pending["rects"] = []
        for i, pt in enumerate(pieces):
            px = bx + 10 + i * sq_size
            py_piece = by + 30
            rect = pygame.Rect(px, py_piece, sq_size, sq_size)
            self.promotion_pending["rects"].append(
                (rect, pt),
            )
            # Hover highlight.
            mouse = pygame.mouse.get_pos()
            if rect.collidepoint(mouse):
                pygame.draw.rect(
                    self.screen,
                    HIGHLIGHT_FROM,
                    rect,
                    border_radius=4,
                )
            piece_obj = chess.Piece(pt, color)
            self.piece_renderer.draw(
                self.screen,
                piece_obj,
                px,
                py_piece,
            )

    def _draw_eval_bar(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> int:
        """Draw the evaluation bar and return the new y.

        The bar shows P(white wins) as a horizontal
        gradient: left = black winning, right = white
        winning.
        """
        if self.position_eval is None:
            return y

        label = self.font_sm.render(
            "Evaluation",
            True,
            TEXT_DIM,
        )
        self.screen.blit(label, (x, y))
        y += label.get_height() + 4

        bar_rect = pygame.Rect(x, y, width, height)
        # Background (black side).
        pygame.draw.rect(
            self.screen,
            (60, 60, 60),
            bar_rect,
            border_radius=4,
        )
        # White portion.
        white_w = max(
            1,
            int(width * self.position_eval),
        )
        white_rect = pygame.Rect(x, y, white_w, height)
        pygame.draw.rect(
            self.screen,
            (235, 235, 235),
            white_rect,
            border_radius=4,
        )
        # Border.
        pygame.draw.rect(
            self.screen,
            TEXT_DIM,
            bar_rect,
            1,
            border_radius=4,
        )
        # Percentage labels.
        w_pct = f"{self.position_eval * 100:.0f}%"
        b_pct = f"{(1 - self.position_eval) * 100:.0f}%"
        w_txt = self.font_sm.render(w_pct, True, (60, 60, 60))
        b_txt = self.font_sm.render(b_pct, True, (220, 220, 220))
        pct_y = y + (height - w_txt.get_height()) // 2
        # White % on the left if white is ahead, else right.
        if self.position_eval >= 0.5:
            self.screen.blit(w_txt, (x + 5, pct_y))
            self.screen.blit(
                b_txt,
                (x + width - b_txt.get_width() - 5, pct_y),
            )
        else:
            self.screen.blit(
                w_txt,
                (x + width - w_txt.get_width() - 5, pct_y),
            )
            self.screen.blit(b_txt, (x + 5, pct_y))
        y += height + 8

        # Numeric eval.
        eval_str = f"White: {self.position_eval:.1%}"
        eval_lbl = self.font_sm.render(
            eval_str,
            True,
            TEXT_COLOUR,
        )
        self.screen.blit(eval_lbl, (x, y))
        y += eval_lbl.get_height() + 4
        return y

    def _draw_top_moves(self, x: int, y: int) -> int:
        """Draw the bot's top candidate moves and return
        the new y position."""
        if self.last_eval is None or not self.last_eval.used_model:
            return y
        if not self.last_eval.top_moves:
            return y

        header = self.font_sm.render(
            "Bot's last evaluation",
            True,
            TEXT_DIM,
        )
        self.screen.blit(header, (x, y))
        y += header.get_height() + 4

        bar_width = self.panel_w - 60
        line_h = self.font_sm.get_height()
        for i, ms in enumerate(self.last_eval.top_moves):
            is_chosen = (
                self.last_eval.chosen is not None
                and ms.move == self.last_eval.chosen.move
            )
            # Move label.
            prefix = "▸ " if is_chosen else "  "
            colour = (120, 255, 120) if is_chosen else TEXT_COLOUR
            pct_str = f"{ms.win_prob:.1%}"
            move_lbl = self.font_sm.render(
                f"{prefix}{ms.san}",
                True,
                colour,
            )
            pct_lbl = self.font_sm.render(
                pct_str,
                True,
                TEXT_DIM,
            )
            self.screen.blit(move_lbl, (x, y))
            self.screen.blit(
                pct_lbl,
                (
                    x + bar_width + 30 - pct_lbl.get_width(),
                    y,
                ),
            )

            # Small inline bar.
            bar_h = max(3, line_h // 5)
            bar_y = y + line_h + 2
            bg_rect = pygame.Rect(x, bar_y, bar_width, bar_h)
            pygame.draw.rect(
                self.screen,
                (60, 60, 60),
                bg_rect,
                border_radius=2,
            )
            fill_w = max(
                1,
                int(bar_width * ms.win_prob),
            )
            fill_rect = pygame.Rect(x, bar_y, fill_w, bar_h)
            bar_colour = (120, 255, 120) if is_chosen else (180, 180, 180)
            pygame.draw.rect(
                self.screen,
                bar_colour,
                fill_rect,
                border_radius=2,
            )
            y += line_h + bar_h + 4
        return y

    def _draw_panel(self) -> None:
        """Draw the side panel with game info."""
        panel_x = self.board_px
        panel_w = self.panel_w
        window_h = self.window_h

        pygame.draw.rect(
            self.screen,
            PANEL_BG,
            (panel_x, 0, panel_w, window_h),
        )
        # Divider line.
        pygame.draw.line(
            self.screen,
            TEXT_DIM,
            (panel_x, 0),
            (panel_x, window_h),
        )

        x = panel_x + 15
        y = 15

        # Title.
        title = self.font_lg.render(
            "♚ Chess",
            True,
            TEXT_COLOUR,
        )
        self.screen.blit(title, (x, y))
        y += title.get_height() + 8

        # Opening info (only when openings are enabled).
        if self.game.use_openings and self.game.state.opening_name:
            opening = self.font_sm.render(
                f"Opening: {self.game.state.opening_name}",
                True,
                TEXT_DIM,
            )
            self.screen.blit(opening, (x, y))
            y += opening.get_height() + 4

        # Model info.
        model_txt = "Model loaded" if self.game.bot.has_model else "No model (random)"
        model_lbl = self.font_sm.render(
            model_txt,
            True,
            TEXT_DIM,
        )
        self.screen.blit(model_lbl, (x, y))
        y += model_lbl.get_height() + 10

        # Turn / status.
        if self.game.is_over:
            status = self.game.result_description()
            colour = (255, 200, 80)
        elif self.bot_thinking:
            status = "Bot is thinking…"
            colour = (180, 180, 255)
        elif self.game.state.is_human_turn:
            status = "Your turn"
            colour = (120, 255, 120)
        else:
            status = "Bot's turn"
            colour = (180, 180, 255)
        status_surf = self.font_md.render(
            status,
            True,
            colour,
        )
        self.screen.blit(status_surf, (x, y))
        y += status_surf.get_height() + 6

        # Move counter.
        move_txt = self.font_sm.render(
            f"Move {self.game.state.move_number}  •  "
            f"{self.game.state.active_color_name} to play",
            True,
            TEXT_DIM,
        )
        self.screen.blit(move_txt, (x, y))
        y += move_txt.get_height() + 6

        # ── Eval bar ───────────────────────────────────
        pygame.draw.line(
            self.screen,
            TEXT_DIM,
            (x, y),
            (panel_x + panel_w - 15, y),
        )
        y += 8
        y = self._draw_eval_bar(
            x,
            y,
            panel_w - 40,
            max(16, int(22 * self.sq_size / DEFAULT_SQ_SIZE)),
        )

        # ── Top moves ─────────────────────────────────
        y = self._draw_top_moves(x, y)
        y += 5

        # ── Move history ──────────────────────────────
        pygame.draw.line(
            self.screen,
            TEXT_DIM,
            (x, y),
            (panel_x + panel_w - 15, y),
        )
        y += 8
        history_label = self.font_sm.render(
            "Moves",
            True,
            TEXT_DIM,
        )
        self.screen.blit(history_label, (x, y))
        y += history_label.get_height() + 4

        history = self.game.state.history
        # Available space above the buttons.
        btn_top = self.btn_undo.rect.top
        line_h = self.font_sm.get_height() + 2
        max_rows = max(1, (btn_top - y - 10) // line_h)
        # Build move pairs.
        pairs: list[str] = []
        i = 0
        while i < len(history):
            rec = history[i]
            if rec.color == "White":
                line = f"{rec.move_number}. {rec.san}"
                if i + 1 < len(history):
                    nxt = history[i + 1]
                    line += f"  {nxt.san}"
                    i += 1
            else:
                line = f"{rec.move_number}. …  {rec.san}"
            pairs.append(line)
            i += 1
        # Show last N rows.
        visible = pairs[-max_rows:] if max_rows > 0 else []
        for line in visible:
            txt = self.font_sm.render(
                line,
                True,
                TEXT_COLOUR,
            )
            self.screen.blit(txt, (x, y))
            y += line_h

        # Buttons.
        mouse = pygame.mouse.get_pos()
        self.btn_undo.draw(self.screen, mouse)
        self.btn_resign.draw(self.screen, mouse)
        self.btn_new_game.draw(self.screen, mouse)

    def _draw(self) -> None:
        """Full redraw."""
        self.screen.fill(BG_COLOUR)
        self._draw_board()
        self._draw_pieces()
        self._draw_promotion_dialog()
        self._draw_panel()
        pygame.display.flip()

    # ── Interaction ─────────────────────────────────────

    def _select_square(self, sq: int) -> None:
        """Handle a click/release on a square."""
        board = self.game.state.board

        # If promotion dialog is open, ignore board clicks.
        if self.promotion_pending is not None:
            return

        # If game is over or not human's turn, ignore.
        if self.game.is_over or not self.game.state.is_human_turn:
            return

        if self.bot_thinking:
            return

        # If we have a selected piece and click a legal
        # destination → make the move.
        if self.selected_sq is not None and sq in self.legal_dests:
            self._try_move(self.selected_sq, sq)
            return

        # Otherwise, select the clicked square if it has
        # one of our pieces.
        piece = board.piece_at(sq)
        human_is_white = self.game.state.human_color == PlayerColor.WHITE
        if piece is not None:
            if (piece.color == chess.WHITE) == human_is_white:
                self.selected_sq = sq
                self.legal_dests = [
                    m.to_square for m in board.legal_moves if m.from_square == sq
                ]
                return

        # Click on empty / opponent piece → deselect.
        self.selected_sq = None
        self.legal_dests = []

    def _try_move(self, from_sq: int, to_sq: int) -> None:
        """Attempt to make a move from from_sq to to_sq."""
        board = self.game.state.board
        # Check for promotion.
        piece = board.piece_at(from_sq)
        is_promo = (
            piece is not None
            and piece.piece_type == chess.PAWN
            and (chess.square_rank(to_sq) == 7 or chess.square_rank(to_sq) == 0)
        )
        if is_promo:
            self.promotion_pending = {
                "from": from_sq,
                "to": to_sq,
                "color": piece.color,
                "rects": [],
            }
            self.selected_sq = None
            self.legal_dests = []
            return

        uci = chess.Move(from_sq, to_sq).uci()
        self._execute_human_move(uci)

    def _execute_human_move(self, uci: str) -> None:
        """Execute a human move and trigger bot reply."""
        try:
            self.game.human_move(uci)
        except ValueError:
            self.selected_sq = None
            self.legal_dests = []
            return

        self.last_move = self.game.state.board.peek()
        self.selected_sq = None
        self.legal_dests = []

        # Update position eval after human's move.
        new_eval = self.game.bot.evaluate_position(
            self.game.state.board,
        )
        if new_eval is not None:
            self.position_eval = new_eval

        # Trigger bot response in a background thread.
        if not self.game.is_over:
            self._trigger_bot_move()

    def _trigger_bot_move(self) -> None:
        """Run the bot move in a background thread so the
        GUI doesn't freeze."""
        self.bot_thinking = True

        def _think() -> None:
            # Small delay so the human can see their move.
            time.sleep(0.4)
            if not self.game.is_over and not self.game.state.is_human_turn:
                _, eval_result = self.game.bot_move()
                self.last_eval = eval_result
                if eval_result.position_eval is not None:
                    self.position_eval = eval_result.position_eval
                self.last_move = self.game.state.board.peek()
                # Update position eval after bot's move.
                new_eval = self.game.bot.evaluate_position(
                    self.game.state.board,
                )
                if new_eval is not None:
                    self.position_eval = new_eval
            self.bot_thinking = False

        t = threading.Thread(target=_think, daemon=True)
        t.start()

    def _handle_promotion_click(
        self,
        pos: tuple[int, int],
    ) -> None:
        """Handle click in the promotion dialog."""
        if self.promotion_pending is None:
            return
        for rect, piece_type in self.promotion_pending["rects"]:
            if rect.collidepoint(pos):
                from_sq = self.promotion_pending["from"]
                to_sq = self.promotion_pending["to"]
                move = chess.Move(
                    from_sq,
                    to_sq,
                    promotion=piece_type,
                )
                uci = move.uci()
                self.promotion_pending = None
                self._execute_human_move(uci)
                return

    def _undo_move(self) -> None:
        """Undo the last full move (human + bot)."""
        board = self.game.state.board
        history = self.game.state.history
        if not history or self.bot_thinking:
            return
        # Undo bot move + human move (2 half-moves).
        for _ in range(2):
            if board.move_stack:
                board.pop()
                if history:
                    history.pop()
        if board.move_stack:
            self.last_move = board.peek()
        else:
            self.last_move = None
        self.game.state.phase = GamePhase.PLAY
        self.selected_sq = None
        self.legal_dests = []

    def _new_game(self) -> None:
        """Reset to a fresh game with the same settings."""
        self.game = Game(
            bot=self.game.bot,
            human_color=self.game.state.human_color,
            use_openings=self.game.use_openings,
        )
        self.selected_sq = None
        self.legal_dests = []
        self.last_move = None
        self.bot_thinking = False
        self.promotion_pending = None
        self.last_eval = None
        self.position_eval = None
        self.game.play_opening_moves()
        if self.game.state.board.move_stack:
            self.last_move = self.game.state.board.peek()
        if not self.game.is_over and not self.game.state.is_human_turn:
            self._trigger_bot_move()

    # ── Main loop ───────────────────────────────────────

    def run(self) -> None:
        """Start the game loop."""
        # Play opening moves first.
        self.game.play_opening_moves()
        if self.game.state.board.move_stack:
            self.last_move = self.game.state.board.peek()

        # If bot goes first after opening, trigger it.
        if not self.game.is_over and not self.game.state.is_human_turn:
            self._trigger_bot_move()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

                # ── Window resize ───────────────────────
                if event.type == pygame.VIDEORESIZE:
                    self._handle_resize(event.w, event.h)
                    continue

                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos

                    # Promotion dialog?
                    if self.promotion_pending is not None:
                        self._handle_promotion_click(pos)
                        continue

                    # Panel buttons?
                    if pos[0] >= self.board_px:
                        if self.btn_new_game.is_clicked(pos):
                            self._new_game()
                            continue
                        if self.btn_resign.is_clicked(pos):
                            if not self.game.is_over:
                                self.game.state.phase = GamePhase.FINISHED
                                self.status_msg = "You resigned."
                            continue
                        if self.btn_undo.is_clicked(pos):
                            self._undo_move()
                            continue
                        continue

                    # Board click → start drag or select.
                    sq = _pixel_to_sq(
                        pos[0],
                        pos[1],
                        self.flipped,
                        self.sq_size,
                        self.board_px,
                    )
                    if sq is not None:
                        board = self.game.state.board
                        piece = board.piece_at(sq)
                        human_is_white = (
                            self.game.state.human_color == PlayerColor.WHITE
                        )
                        if (
                            piece is not None
                            and (piece.color == chess.WHITE) == human_is_white
                            and self.game.state.is_human_turn
                            and not self.game.is_over
                            and not self.bot_thinking
                        ):
                            # Start dragging.
                            self.dragging = True
                            self.drag_piece = piece
                            self.drag_origin = sq
                            self.drag_pos = pos
                            self.selected_sq = sq
                            self.legal_dests = [
                                m.to_square
                                for m in board.legal_moves
                                if m.from_square == sq
                            ]
                        else:
                            self._select_square(sq)

                if event.type == pygame.MOUSEMOTION:
                    if self.dragging:
                        self.drag_pos = event.pos

                if event.type == pygame.MOUSEBUTTONUP:
                    if self.dragging:
                        self.dragging = False
                        sq = _pixel_to_sq(
                            event.pos[0],
                            event.pos[1],
                            self.flipped,
                            self.sq_size,
                            self.board_px,
                        )
                        if (
                            sq is not None
                            and sq != self.drag_origin
                            and sq in self.legal_dests
                        ):
                            self._try_move(
                                self.drag_origin,
                                sq,
                            )
                        else:
                            # Dropped back / invalid →
                            # keep selection.
                            pass
                        self.drag_piece = None
                        self.drag_origin = None

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.selected_sq = None
                        self.legal_dests = []
                        self.promotion_pending = None
                    if event.key == pygame.K_q:
                        running = False
                        break
                    if event.key == pygame.K_u:
                        self._undo_move()
                    if event.key == pygame.K_n:
                        self._new_game()

            self._draw()
            self.clock.tick(60)

        pygame.quit()


# ── Entry point ─────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    """Parse args and launch the GUI."""
    parser = argparse.ArgumentParser(
        description="Play chess against your trained model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=("Path to model checkpoint. If omitted the bot plays random moves."),
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=[
            "piece_plane",
            "dynamic_piece_plane",
            "full_piece_plane",
        ],
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
        help=("Bot temperature. 0 = greedy, >0 = stochastic (default: 0)."),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--openings",
        action="store_true",
        default=False,
        help=(
            "Use pre-programmed openings for the first 4 moves. Disabled by default."
        ),
    )
    args = parser.parse_args(argv)

    human_color = PlayerColor.WHITE if args.color == "white" else PlayerColor.BLACK
    flipped = human_color == PlayerColor.BLACK

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
        use_openings=args.openings,
    )

    gui = ChessGUI(game=game, flipped=flipped)
    gui.run()


if __name__ == "__main__":
    main()
