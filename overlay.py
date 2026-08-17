"""The demo surface: one fullscreen window, camera feed + layers.

The draw loop OWNS this window and does nothing but draw (plan rule).
Everything else — detections, board state, eval, commentary, banner state —
is published into Overlay from other threads via publish(); the loop renders
whatever is newest. Static chrome (panel background, banner plates) is
pre-rendered once and blitted.

Run standalone to polish visuals with a mocked state machine (plan phase 3):

    .venv/bin/python overlay.py            # synthetic camera, no hardware
    keys: 1-8 cycle states, g ghost drop, v win highlight, q quit
"""

import time
from threading import Lock

import cv2
import numpy as np

from board import COLOR_TO_PLAYER, COLS, EMPTY, HUMAN, ROBOT, ROWS

FEED_W, FEED_H = 1280, 720
PANEL_W, TICKER_H = 360, 80
W, H = FEED_W + PANEL_W, FEED_H + TICKER_H

RED = (60, 60, 230)      # robot pieces (BGR)
YELLOW = (0, 205, 235)   # human pieces (BGR)
GOLD = (60, 200, 255)
GREEN = (80, 220, 80)
DIM = (46, 40, 36)

BANNERS = {
    "WAIT_HUMAN": ("YOUR TURN, DROP A PIECE", GREEN),
    "SCANNING": ("READING BOARD", (200, 200, 200)),
    "THINKING": ("THINKING", GOLD),
    "ROBOT_MOVING": ("ROBOT PLAYING COLUMN {col}", RED),
    "VERIFYING": (None, None),
    "HUMAN_WIN": ("YOU WIN", YELLOW),
    "ROBOT_WIN": ("ROBOT WINS", RED),
    "ERROR": ("BOARD LOOKS WRONG, FIX IT", (40, 40, 255)),
}

CELL = 42  # digital twin cell size
TWIN_X, TWIN_Y = FEED_W + 26, 64


class Overlay:
    def __init__(self, calib=None, show_fps=True):
        self.calib = calib
        self.show_fps = show_fps
        self._lock = Lock()
        self._s = {
            "state": "WAIT_HUMAN", "col": None, "dets": [], "board": None,
            "eval": 0, "history": [], "ticker": "", "ghost": None,
            "considered": [], "win_cells": None, "expected": None,
            "infer_fps": 0.0, "turn": 0,
        }
        self._base = self._prerender()
        self.fps = 0.0
        self._last_draw = time.time()

    def publish(self, **kw):
        """Thread-safe partial state update from the game/vision side."""
        with self._lock:
            self._s.update(kw)

    def ghost_drop(self, col, row, duration=0.8):
        """Flourish #1: announce the move on the twin before the arm plays it."""
        with self._lock:
            self._s["ghost"] = (col, row, time.time(), duration)

    # ---- pre-rendered chrome ----

    def _prerender(self):
        base = np.zeros((H, W, 3), dtype=np.uint8)
        base[:] = (24, 20, 18)
        p = base[0:FEED_H, FEED_W:W]
        p[:] = (34, 28, 24)
        cv2.rectangle(base, (FEED_W, 0), (W - 1, FEED_H), (60, 50, 44), 2)
        cv2.putText(base, "ROBOT MIND", (TWIN_X, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (150, 140, 130), 1)
        # twin board plate
        cv2.rectangle(base, (TWIN_X - 8, TWIN_Y - 8),
                      (TWIN_X + COLS * CELL + 8, TWIN_Y + ROWS * CELL + 8),
                      (70, 55, 40), -1)
        for r in range(ROWS):
            for c in range(COLS):
                cx = TWIN_X + c * CELL + CELL // 2
                cy = TWIN_Y + r * CELL + CELL // 2
                cv2.circle(base, (cx, cy), CELL // 2 - 4, DIM, -1)
        cv2.putText(base, "HISTORY", (TWIN_X, TWIN_Y + ROWS * CELL + 42),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (150, 140, 130), 1)
        # advantage bar frame (right edge of panel)
        bx = W - 40
        cv2.rectangle(base, (bx, TWIN_Y), (bx + 22, TWIN_Y + 560), (70, 60, 52), 1)
        cv2.putText(base, "BOT", (bx - 4, TWIN_Y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1)
        cv2.putText(base, "YOU", (bx - 4, TWIN_Y + 580),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1)
        return base

    # ---- per-frame composition ----

    def compose(self, frame):
        with self._lock:
            s = dict(self._s)
        canvas = self._base.copy()
        if frame is not None:
            fh, fw = frame.shape[:2]
            if (fw, fh) != (FEED_W, FEED_H):
                frame = cv2.resize(frame, (FEED_W, FEED_H))
            canvas[0:FEED_H, 0:FEED_W] = frame
        self._draw_grid(canvas)
        self._draw_dets(canvas, s["dets"])
        self._draw_win(canvas, s["win_cells"])
        self._draw_banner(canvas, s)
        self._draw_twin(canvas, s)
        self._draw_panel_text(canvas, s)
        self._draw_ticker(canvas, s)
        now = time.time()
        self.fps = 0.92 * self.fps + 0.08 / max(now - self._last_draw, 1e-6)
        self._last_draw = now
        if self.show_fps:
            cv2.putText(canvas, f"{self.fps:4.0f} fps  infer {s['infer_fps']:.0f}",
                        (12, FEED_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (120, 255, 120), 1)
        return canvas

    def _draw_grid(self, canvas):
        if self.calib is None:
            return
        for c in range(COLS + 1):
            a = self.calib.cell_center_frame(-0.5 + ROWS, c - 0.5)  # top edge
            b = self.calib.cell_center_frame(-0.5, c - 0.5)
            cv2.line(canvas, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     (90, 200, 90), 1, cv2.LINE_AA)
        for r in range(ROWS + 1):
            a = self.calib.cell_center_frame(r - 0.5, -0.5)
            b = self.calib.cell_center_frame(r - 0.5, COLS - 0.5)
            cv2.line(canvas, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     (90, 200, 90), 1, cv2.LINE_AA)

    def _draw_dets(self, canvas, dets):
        for x, y, w, h, cls, conf in dets:
            player = COLOR_TO_PLAYER.get(cls)
            if player is None:
                continue  # the model also emits 'board' / 'no piece' — not pieces
            color = RED if player == ROBOT else YELLOW
            p1 = (int(x - w / 2), int(y - h / 2))
            p2 = (int(x + w / 2), int(y + h / 2))
            cv2.rectangle(canvas, p1, p2, color, 2)
            cv2.putText(canvas, f"{cls} {conf:.2f}", (p1[0], p1[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def _draw_win(self, canvas, cells):
        if not cells or self.calib is None:
            return
        pts = [self.calib.cell_center_frame(r, c) for r, c in cells]
        cv2.line(canvas, (int(pts[0][0]), int(pts[0][1])),
                 (int(pts[-1][0]), int(pts[-1][1])), GOLD, 8, cv2.LINE_AA)

    def _draw_banner(self, canvas, s):
        text, color = BANNERS.get(s["state"], (None, None))
        if not text:
            return
        text = text.format(col=s["col"])
        pulse = 0.75 + 0.25 * np.sin(time.time() * 4) if s["state"] == "WAIT_HUMAN" else 1.0
        col = tuple(int(v * pulse) for v in color)
        if s["state"] == "THINKING":
            text += "." * (int(time.time() * 2.5) % 4)
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.3, 3)[0]
        x = (FEED_W - size[0]) // 2
        cv2.rectangle(canvas, (x - 18, 12), (x + size[0] + 18, 64), (20, 16, 14), -1)
        cv2.rectangle(canvas, (x - 18, 12), (x + size[0] + 18, 64), col, 2)
        cv2.putText(canvas, text, (x, 52), cv2.FONT_HERSHEY_DUPLEX, 1.3, col, 3)
        if s["state"] == "ERROR" and s["expected"]:
            cv2.putText(canvas, s["expected"], (x - 18, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 255), 1)

    def _twin_center(self, row, col):
        return (TWIN_X + col * CELL + CELL // 2,
                TWIN_Y + (ROWS - 1 - row) * CELL + CELL // 2)

    def _draw_twin(self, canvas, s):
        board = s["board"] or [[EMPTY] * COLS for _ in range(ROWS)]
        for r in range(ROWS):
            for c in range(COLS):
                if board[r][c] != EMPTY:
                    color = RED if board[r][c] == ROBOT else YELLOW
                    cv2.circle(canvas, self._twin_center(r, c), CELL // 2 - 4,
                               color, -1)
        # flourish #3: flash the columns minimax is considering
        if s["state"] == "THINKING" and s["considered"]:
            col = s["considered"][int(time.time() * 9) % len(s["considered"])][0]
            x = TWIN_X + col * CELL
            cv2.rectangle(canvas, (x, TWIN_Y), (x + CELL, TWIN_Y + ROWS * CELL),
                          GOLD, 2)
        # flourish #1: ghost piece dropping before the arm moves
        if s["ghost"]:
            col, row, t0, dur = s["ghost"]
            t = min((time.time() - t0) / dur, 1.0)
            top = self._twin_center(ROWS - 1, col)[1] - CELL
            bottom = self._twin_center(row, col)[1]
            y = int(top + (bottom - top) * (t * t))  # accelerate like gravity
            x = self._twin_center(0, col)[0]
            overlay_c = canvas.copy()
            cv2.circle(overlay_c, (x, y), CELL // 2 - 4, RED, -1)
            cv2.addWeighted(overlay_c, 0.55, canvas, 0.45, 0, canvas)
        # target column highlight while the arm plays
        if s["state"] == "ROBOT_MOVING" and s["col"] is not None:
            x = TWIN_X + s["col"] * CELL
            cv2.rectangle(canvas, (x, TWIN_Y), (x + CELL, TWIN_Y + ROWS * CELL),
                          RED, 2)

    def _draw_panel_text(self, canvas, s):
        y = TWIN_Y + ROWS * CELL + 68
        for line in s["history"][-8:]:
            cv2.putText(canvas, line, (TWIN_X, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 190, 180), 1)
            y += 24
        # flourish #4: chips
        board = s["board"]
        n = sum(1 for r in range(ROWS) for c in range(COLS)
                if board and board[r][c] != EMPTY)
        chip = f"turn {s['turn']}   pieces {n}"
        cv2.putText(canvas, chip, (TWIN_X, FEED_H - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 140, 130), 1)
        # advantage bar: eval > 0 favors the robot
        bx = W - 40
        mid = TWIN_Y + 280
        extent = int(278 * np.tanh(s["eval"] / 400.0))
        if extent >= 0:
            cv2.rectangle(canvas, (bx + 1, mid - extent), (bx + 21, mid), RED, -1)
        else:
            cv2.rectangle(canvas, (bx + 1, mid), (bx + 21, mid - extent), YELLOW, -1)
        cv2.line(canvas, (bx - 2, mid), (bx + 24, mid), (150, 140, 130), 1)

    def _draw_ticker(self, canvas, s):
        cv2.rectangle(canvas, (0, FEED_H), (W, H), (16, 13, 11), -1)
        if s["ticker"]:
            cv2.putText(canvas, f'"{s["ticker"]}"', (24, FEED_H + 50),
                        cv2.FONT_HERSHEY_DUPLEX, 0.95, (180, 230, 255), 1)

    # ---- the loop (must run on the main thread on macOS) ----

    def run(self, frame_source, fullscreen=True, on_key=None):
        """frame_source() -> (frame_id, ndarray|None). Blocks until q/ESC."""
        name = "connect4"
        cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)
        if fullscreen:
            cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        while True:
            _, frame = frame_source()
            cv2.imshow(name, self.compose(frame))
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            if k != 255 and on_key:
                on_key(chr(k) if k < 128 else "")
        cv2.destroyAllWindows()


# ---- phase-3 mock: polish visuals with no hardware ----

if __name__ == "__main__":
    import itertools

    from board import Calibration, empty_board

    # synthetic "camera": a flat photo of a board drawn each frame
    corners = np.array([[340, 120], [940, 120], [940, 640], [340, 640]], np.float32)
    dst = np.array([[0, 0], [700, 0], [700, 600], [0, 600]], np.float32)
    Hm = cv2.getPerspectiveTransform(corners, dst)
    calib = Calibration(Hm.tolist(), 100, 100)

    demo_board = empty_board()
    for c, p in [(3, ROBOT), (3, HUMAN), (2, ROBOT), (4, HUMAN), (2, ROBOT)]:
        from board import landing_row
        demo_board[landing_row(demo_board, c)][c] = p

    def synth_frame():
        f = np.full((FEED_H, FEED_W, 3), (70, 80, 90), np.uint8)
        cv2.rectangle(f, (340, 120), (940, 640), (140, 90, 30), -1)
        dets = []
        for r in range(ROWS):
            for c in range(COLS):
                x, y = calib.cell_center_frame(r, c)
                cell = demo_board[r][c]
                col = (40, 35, 30) if cell == EMPTY else RED if cell == ROBOT else YELLOW
                cv2.circle(f, (int(x), int(y)), 34, col, -1)
                if cell != EMPTY:
                    dets.append((x, y, 70, 70, "red piece" if cell == ROBOT else "yellow piece", 0.93))
        return dets, f

    ov = Overlay(calib=calib)
    states = itertools.cycle(["WAIT_HUMAN", "SCANNING", "THINKING", "ROBOT_MOVING",
                              "VERIFYING", "HUMAN_WIN", "ROBOT_WIN", "ERROR"])

    def key(k):
        if k == " ":
            st = next(states)
            ov.publish(state=st, col=4, eval=180,
                       considered=[(3, 120), (4, 180), (2, -40)],
                       expected="expected red at col 4, saw nothing" if st == "ERROR" else None)
            print("state:", st)
        elif k == "g":
            ov.ghost_drop(4, 2)
        elif k == "v":
            ov.publish(win_cells=[(0, 3), (1, 3), (2, 3), (3, 3)])
        elif k == "n":
            ov.publish(win_cells=None, ghost=None)

    def source():
        dets, f = synth_frame()
        ov.publish(dets=dets, board=demo_board,
                   ticker="column 4. as if you had a choice.")
        return 0, f

    print("mock overlay: SPACE cycles states, g ghost, v win line, n clear, q quit")
    ov.run(source, fullscreen=False, on_key=key)
