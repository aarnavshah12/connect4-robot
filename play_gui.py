"""Click-to-play OpenCV window for the trained PyTorch Connect 4 bot.

Run:  .venv/bin/python play_gui.py

You are yellow and the bot is red. This file opens no camera or arm connection.
Click a board column to play; use the on-screen buttons or N/B/M/Q keys.
"""

from pathlib import Path

import cv2
import numpy as np

import board as B
from ml.player import TorchPlayer


ROOT = Path(__file__).resolve().parent
WINDOW = "Play the PyTorch Connect 4 bot"
WIDTH, HEIGHT = 850, 710
LEFT, TOP, CELL = 34, 150, 80
BOARD_RIGHT, BOARD_BOTTOM = LEFT + 7 * CELL, TOP + 6 * CELL
BUTTONS = {
    "new": (640, 180, 820, 225),
    "bot_first": (640, 237, 820, 282),
    "mode": (640, 294, 820, 339),
}
MODES = (
    ("RL + search", "artifacts/player/best.pt", 64, True),
    ("RL policy only", "artifacts/player/best.pt", 0, False),
    ("Supervised + search", "artifacts/player/supervised.pt", 64, True),
)


def label(image, text, x, y, scale=0.65, color=(235, 240, 245), thickness=1):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


class Demo:
    def __init__(self):
        self.players = {}
        self.mode_index = 0
        self.new_game(bot_first=False)

    @property
    def mode(self):
        return MODES[self.mode_index]

    def new_game(self, bot_first=False):
        self.board = B.empty_board()
        self.last_move = None
        self.finished = False
        self.human_turn = not bot_first
        self.pending_bot = bot_first
        self.status = "Bot is thinking..." if bot_first else "Your turn: click a column."
        self.detail = "RL + search is the trained model with 64 neural search steps."
        self.top_moves = []
        self.value = None

    def change_mode(self):
        self.mode_index = (self.mode_index + 1) % len(MODES)
        self.top_moves = []
        self.value = None
        self.detail = f"Selected: {self.mode[0]}. Applies on the next bot move."

    def _place(self, col, side):
        row = B.landing_row(self.board, col)
        if row is None:
            return False
        self.board[row][col] = side
        self.last_move = (row, col)
        if B.check_win(self.board, side):
            self.finished = True
            self.status = "You win!" if side == B.HUMAN else "Bot wins!"
        elif B.is_full(self.board):
            self.finished = True
            self.status = "Draw!"
        return True

    def human_move(self, col):
        if self.finished or not self.human_turn or col not in range(7):
            return False
        if not self._place(col, B.HUMAN):
            self.status = f"Column {col + 1} is full. Choose another."
            return False
        self.human_turn = False
        if not self.finished:
            self.pending_bot = True
            self.status = "Bot is thinking..."
        return True

    def bot_move(self):
        if not self.pending_bot or self.finished:
            return
        self.pending_bot = False
        name, checkpoint, simulations, guard = self.mode
        try:
            if self.mode_index not in self.players:
                self.players[self.mode_index] = TorchPlayer(
                    ROOT / checkpoint, device="cpu", search_simulations=simulations,
                    tactical_guard=guard)
            choices = []
            col, self.value = self.players[self.mode_index](
                self.board, B.ROBOT, considered=choices)
            if not self._place(col, B.ROBOT):
                raise ValueError(f"model chose full column {col + 1}")
        except Exception as exc:
            self.status = "Bot error. See the terminal; press N to restart."
            self.detail = str(exc)
            print(f"[play_gui] {exc}", flush=True)
            self.finished = True
            return
        self.top_moves = sorted(choices, key=lambda item: item[1], reverse=True)[:3]
        self.detail = f"{name} played column {col + 1}."
        if not self.finished:
            self.human_turn = True
            self.status = "Your turn: click a column."

    def click(self, x, y):
        for action, (x0, y0, x1, y1) in BUTTONS.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                if action == "new":
                    self.new_game()
                elif action == "bot_first":
                    self.new_game(bot_first=True)
                else:
                    self.change_mode()
                return
        if LEFT <= x < BOARD_RIGHT and TOP <= y < BOARD_BOTTOM:
            self.human_move((x - LEFT) // CELL)

    def render(self):
        image = np.full((HEIGHT, WIDTH, 3), (26, 31, 43), dtype=np.uint8)
        label(image, "CONNECT 4  /  PYTORCH BOT", 34, 45, 0.9, (250, 250, 250), 2)
        label(image, "You: yellow    Bot: red", 34, 79, 0.62, (205, 220, 230))
        label(image, self.status, 34, 113, 0.72, (105, 225, 255), 2)

        cv2.rectangle(image, (LEFT - 4, TOP - 4), (BOARD_RIGHT + 4, BOARD_BOTTOM + 4),
                      (160, 93, 35), -1)
        for col in range(7):
            cx = LEFT + col * CELL + CELL // 2
            label(image, str(col + 1), cx - 8, TOP - 15, 0.62)
            for top_row in range(6):
                row = 5 - top_row
                piece = self.board[row][col]
                color = {B.EMPTY: (53, 44, 27), B.ROBOT: (65, 65, 235),
                         B.HUMAN: (65, 218, 255)}[piece]
                center = (cx, TOP + top_row * CELL + CELL // 2)
                cv2.circle(image, center, 31, color, -1, cv2.LINE_AA)
                if self.last_move == (row, col):
                    cv2.circle(image, center, 33, (245, 245, 245), 2, cv2.LINE_AA)

        for action, (x0, y0, x1, y1) in BUTTONS.items():
            cv2.rectangle(image, (x0, y0), (x1, y1), (67, 73, 91), -1)
            cv2.rectangle(image, (x0, y0), (x1, y1), (122, 136, 156), 1)
            caption = {"new": "New game (N)", "bot_first": "Bot first (B)",
                       "mode": "Switch mode (M)"}[action]
            label(image, caption, x0 + 11, y0 + 29, 0.57)

        label(image, "CURRENT MODE", 640, 390, 0.58, (170, 190, 215))
        label(image, self.mode[0], 640, 422, 0.6, (250, 250, 250))
        label(image, "LAST BOT MOVE", 640, 474, 0.58, (170, 190, 215))
        if self.value is not None:
            label(image, f"Advantage: {self.value:+.2f}", 640, 503, 0.55)
            shares = "visits" if self.mode[2] else "policy"
            label(image, f"Top {shares}:", 640, 533, 0.53)
            for index, (col, share) in enumerate(self.top_moves):
                label(image, f"  {col + 1}: {share:.0%}", 640, 559 + 24 * index, 0.5)
        label(image, self.detail[:92], 34, 663, 0.52, (205, 220, 230))
        label(image, "Click a column to play.  N: new  B: bot first  M: mode  Q: quit",
              34, 687, 0.52, (170, 190, 215))
        return image


def main():
    demo = Demo()
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    def mouse(event, x, y, _flags, _user_data):
        if event == cv2.EVENT_LBUTTONDOWN:
            demo.click(x, y)

    cv2.setMouseCallback(WINDOW, mouse)
    try:
        while True:
            cv2.imshow(WINDOW, demo.render())
            key = cv2.waitKey(30) & 0xFF
            try:
                visible = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE)
            except cv2.error:  # the user closed the window
                break
            if key == ord("q") or visible < 1:
                break
            if key == ord("n"):
                demo.new_game()
            elif key == ord("b"):
                demo.new_game(bot_first=True)
            elif key == ord("m"):
                demo.change_mode()
            if demo.pending_bot:
                demo.bot_move()
    finally:
        cv2.destroyWindow(WINDOW)


if __name__ == "__main__":
    main()
