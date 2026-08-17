"""Board parsing and game logic: grid snap, debounce, legality, win, diff.

Pure logic — no camera, no model — so all of it unit-tests offline.

Conventions (project-wide):
  - board[row][col], 6 rows x 7 cols, row 0 = BOTTOM (gravity pulls to row 0).
  - Cells: 0 empty, ROBOT = 1 (red pieces), HUMAN = 2 (blue pieces).
  - Detections arrive as (x, y, w, h, class_name, confidence) in full-frame
    pixels (vision.py's format); the homography from calibrate.py warps the
    (x, y) centroids into board space where cell centers form a regular grid.

Plan rules implemented here (§Grid mapping): snap to nearest cell center,
reject detections farther than half a cell from any center, debounce =
5 identical consecutive parses, physics check = no floating pieces.
"""

import json
from pathlib import Path

import numpy as np

ROWS, COLS = 6, 7
EMPTY, ROBOT, HUMAN = 0, 1, 2

# model class name -> player. Robot plays RED (owner default).
COLOR_TO_PLAYER = {"red": ROBOT, "blue": HUMAN}

DEBOUNCE_N = 5
CALIBRATION_PATH = Path(__file__).parent / "calibration.json"


class Calibration:
    """Homography + cell geometry saved by calibrate.py."""

    def __init__(self, homography, cell_w, cell_h):
        self.H = np.asarray(homography, dtype=np.float64)
        self.Hinv = np.linalg.inv(self.H)
        self.cell_w = float(cell_w)
        self.cell_h = float(cell_h)

    @classmethod
    def load(cls, path=CALIBRATION_PATH):
        d = json.loads(Path(path).read_text())
        return cls(d["homography"], d["cell_w"], d["cell_h"])

    def warp(self, x, y):
        """Frame pixel -> board-space coordinates (units: pixels of the
        rectified board, origin top-left corner of the play area)."""
        v = self.H @ np.array([x, y, 1.0])
        return v[0] / v[2], v[1] / v[2]

    def snap(self, x, y):
        """Frame pixel -> (row, col) or None if > half a cell from any center.
        Board space has row 0 at the TOP of the image; we flip so row 0 = bottom."""
        bx, by = self.warp(x, y)
        col = round(bx / self.cell_w - 0.5)
        row_top = round(by / self.cell_h - 0.5)
        if not (0 <= col < COLS and 0 <= row_top < ROWS):
            return None
        cx, cy = (col + 0.5) * self.cell_w, (row_top + 0.5) * self.cell_h
        if abs(bx - cx) > self.cell_w / 2 or abs(by - cy) > self.cell_h / 2:
            return None
        return (ROWS - 1 - row_top, col)

    def cell_center_frame(self, row, col):
        """(row, col) -> full-frame pixel of that cell's center (for drawing)."""
        row_top = ROWS - 1 - row
        pt = np.array([(col + 0.5) * self.cell_w, (row_top + 0.5) * self.cell_h, 1.0])
        v = self.Hinv @ pt
        return float(v[0] / v[2]), float(v[1] / v[2])


def empty_board():
    return [[EMPTY] * COLS for _ in range(ROWS)]


def parse_detections(dets, calib, min_confidence=0.5):
    """Detections -> board, or None if the frame is unusable.

    Unusable: two detections snap to the same cell, or the result fails the
    physics check (floating piece). Off-grid/low-confidence detections are
    dropped silently (hands, pieces in the feeder, reflections).
    """
    board = empty_board()
    for x, y, _w, _h, cls, conf in dets:
        if conf < min_confidence or cls not in COLOR_TO_PLAYER:
            continue
        hit = calib.snap(x, y)
        if hit is None:
            continue
        r, c = hit
        player = COLOR_TO_PLAYER[cls]
        if board[r][c] not in (EMPTY, player):
            return None  # red and blue claiming one cell: garbage frame
        board[r][c] = player
    return board if physics_ok(board) else None


def physics_ok(board):
    """No floating pieces: every column filled contiguously from the bottom."""
    for c in range(COLS):
        seen_empty = False
        for r in range(ROWS):
            if board[r][c] == EMPTY:
                seen_empty = True
            elif seen_empty:
                return False
    return True


class Debouncer:
    """A parsed board only counts after DEBOUNCE_N identical consecutive reads."""

    def __init__(self, n=DEBOUNCE_N):
        self.n = n
        self._last = None
        self._count = 0

    def feed(self, board):
        """board (or None for an unusable frame) -> stable board or None."""
        if board is None:
            self._last, self._count = None, 0
            return None
        if board == self._last:
            self._count += 1
        else:
            self._last, self._count = board, 1
        return board if self._count >= self.n else None

    def reset(self):
        self._last, self._count = None, 0


def diff(old, new):
    """Cells added between two boards -> [(row, col, player), ...].
    Returns None if any old piece vanished or changed color (bad scan)."""
    added = []
    for r in range(ROWS):
        for c in range(COLS):
            if old[r][c] != new[r][c]:
                if old[r][c] != EMPTY:
                    return None
                added.append((r, c, new[r][c]))
    return added


def landing_row(board, col):
    """Row a piece dropped in `col` lands at, or None if the column is full."""
    for r in range(ROWS):
        if board[r][col] == EMPTY:
            return r
    return None


def legal_human_move(old, new):
    """Exactly one new HUMAN piece, resting where gravity puts it.
    Returns (row, col) or None."""
    added = diff(old, new)
    if not added or len(added) != 1:
        return None
    r, c, player = added[0]
    if player != HUMAN or landing_row(old, c) != r:
        return None
    return (r, c)


def check_win(board, player):
    """Winning four cells as [(row, col) * 4], or None."""
    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] != player:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (-1, 1)):
                cells = [(r + i * dr, c + i * dc) for i in range(4)]
                if all(
                    0 <= rr < ROWS and 0 <= cc < COLS and board[rr][cc] == player
                    for rr, cc in cells
                ):
                    return cells
    return None


def is_full(board):
    return all(board[ROWS - 1][c] != EMPTY for c in range(COLS))
