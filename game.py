"""Game state machine (plan §game.py) tying vision, engine, arm, overlay,
and commentary together.

    WAIT_HUMAN -> SCANNING -> THINKING -> ROBOT_MOVING -> VERIFYING -> ...

The machine runs in a worker thread; the overlay's draw loop owns the main
thread (macOS UI rule). Nothing here ever blocks on commentary or TTS —
Commentator.fire() is fire-and-forget by construction.

Run:
    .venv/bin/python game.py            # the real thing (arm + camera)
    .venv/bin/python game.py --no-arm   # rehearsal without the arm connected

After a win, clearing all pieces off the board starts a new game.
"""

import sys
import threading
import time
from pathlib import Path

import yaml

import board as B
from board import Calibration, Debouncer, HUMAN, ROBOT
from engine import WIN, best_move

IMPATIENCE_S = 20
VERIFY_TIMEOUT_S = 8
DRAMA_S = 1.5
GHOST_S = 0.85


def load_config():
    p = Path(__file__).parent / "config.yaml"
    if not p.exists():
        raise SystemExit("config.yaml missing — copy config.example.yaml")
    cfg = yaml.safe_load(p.read_text()) or {}
    if not cfg.get("roboflow_api_key"):
        # the owner keeps the Roboflow key in .claude/settings.local.json
        import json
        import os

        local = Path(__file__).parent / ".claude" / "settings.local.json"
        if local.exists():
            try:
                key = (json.loads(local.read_text()).get("env") or {}).get(
                    "ROBOFLOW_API_KEY")
            except Exception as e:
                print(f"[config] could not read {local.name}: {e}")
                key = None
            if key:
                cfg["roboflow_api_key"] = key
        if not cfg.get("roboflow_api_key"):
            cfg["roboflow_api_key"] = os.environ.get("ROBOFLOW_API_KEY", "")
    if not cfg.get("roboflow_api_key"):
        raise SystemExit("no Roboflow API key: set it in config.yaml, "
                         ".claude/settings.local.json, or $ROBOFLOW_API_KEY")
    return cfg


class Game:
    """Deps are injected so the machine is testable without hardware:
    vision needs .detections(), arm needs .pick_and_drop(col),
    commentator needs .fire(trigger, **ctx)."""

    def __init__(self, vision, arm, overlay, commentator, calib,
                 clock=time.time, sleep=time.sleep):
        self.vision = vision
        self.arm = arm
        self.overlay = overlay
        self.say = commentator
        self.calib = calib
        self.clock = clock
        self.sleep = sleep
        self.debounce = Debouncer()
        self.reset()

    def reset(self):
        self.board = B.empty_board()
        self.turn = 0
        self.robot_picks = 0  # pieces taken off the feeder stack this game
        self.history = []
        self.debounce.reset()

    # ---- vision helpers ----

    def _pieces(self, bd):
        return sum(1 for r in range(B.ROWS) for c in range(B.COLS)
                   if bd[r][c] != B.EMPTY)

    def _poll_stable(self):
        """One debouncer step. Returns a stable parsed board or None.
        Frames where the detection count drops below what's already on the
        board are occlusion (a hand) and reset the streak (plan rule)."""
        _, dets, _ = self.vision.detections()
        parsed = B.parse_detections(dets, self.calib)
        if parsed is not None and self._pieces(parsed) < self._pieces(self.board):
            parsed = None
        return self.debounce.feed(parsed)

    def _await_change(self):
        """WAIT_HUMAN: block until the stable board differs from ours."""
        t0 = self.clock()
        nagged = False
        self.debounce.reset()
        while True:
            stable = self._poll_stable()
            if stable is not None and stable != self.board:
                return stable
            if not nagged and self.clock() - t0 > IMPATIENCE_S:
                nagged = True
                self.say.fire("impatience", seconds=int(self.clock() - t0))
            self.sleep(0.03)

    def _await_board(self, want, timeout):
        """VERIFYING/ERROR recovery: wait until the stable board equals
        `want` (or, if want is None, until any stable board arrives)."""
        t0 = self.clock()
        self.debounce.reset()
        while self.clock() - t0 < timeout:
            stable = self._poll_stable()
            if stable is not None and (want is None or stable == want):
                return stable
            self.sleep(0.03)
        return None

    def _error_until_fixed(self, expected_text, want):
        self.overlay.publish(state="ERROR", expected=expected_text)
        self.say.fire("error")
        while self._await_board(want, timeout=3600) is None:
            pass
        self.overlay.publish(state="WAIT_HUMAN", expected=None)

    # ---- one full human+robot exchange ----

    def play_turn(self):
        self.overlay.publish(state="WAIT_HUMAN", board=self.board, turn=self.turn)
        stable = self._await_change()

        self.overlay.publish(state="SCANNING")
        move = B.legal_human_move(self.board, stable)
        if move is None:
            self._error_until_fixed("that isn't one new yellow piece — undo it", self.board)
            return "error"
        r, c = move

        # eval swing across the human's move decides the jab flavor
        _, before = best_move(self.board, ROBOT)
        best_human, _ = best_move(self.board, HUMAN)
        self.board = stable
        self.turn += 1
        self.history.append(f"{self.turn}. you  col {c}")
        win = B.check_win(self.board, HUMAN)
        if win:
            self.overlay.publish(state="HUMAN_WIN", board=self.board,
                                 win_cells=win, history=self.history)
            self.say.fire("human_win")
            return "human_win"

        self.overlay.publish(state="THINKING", board=self.board,
                             history=self.history)
        considered = []
        col, score = best_move(self.board, ROBOT, considered=considered)
        swing = score - before
        self.overlay.publish(eval=score, considered=considered)
        if swing >= 300:
            self.say.fire("blunder", col=c, swing=swing)
        elif c == best_human:
            self.say.fire("respect", col=c)
        else:
            self.say.fire("jab", col=c)
        self.sleep(DRAMA_S)  # drama hold (plan); engine itself is ~10 ms

        row = B.landing_row(self.board, col)
        self.overlay.publish(state="ROBOT_MOVING", col=col)
        self.overlay.ghost_drop(col, row, duration=GHOST_S)
        self.sleep(GHOST_S)  # announce intent, then move (money shot)
        self.arm.pick_and_drop(col, self.robot_picks)
        self.robot_picks += 1  # the stack is one piece shorter now

        expected = [rw[:] for rw in self.board]
        expected[row][col] = ROBOT
        self.overlay.publish(state="VERIFYING")
        seen = self._await_board(expected, VERIFY_TIMEOUT_S)
        if seen is None:
            self._error_until_fixed(
                f"expected red at column {col} — place it or clear the jam", expected)
        self.board = expected
        self.turn += 1
        self.history.append(f"{self.turn}. bot  col {col}")
        self.overlay.publish(board=self.board, history=self.history, col=None)

        win = B.check_win(self.board, ROBOT)
        if win:
            self.overlay.publish(state="ROBOT_WIN", win_cells=win)
            self.say.fire("robot_win")
            return "robot_win"
        if score >= WIN:
            self.say.fire("foreshadow")  # forced win found; don't spoil the column
        if B.is_full(self.board):
            return "draw"
        return "ok"

    def run(self):
        self.say.fire("game_start")
        while True:
            result = self.play_turn()
            if result in ("human_win", "robot_win", "draw"):
                # board cleared -> new game
                self._await_board(B.empty_board(), timeout=3600 * 24)
                self.reset()
                self.overlay.publish(board=self.board, history=[], win_cells=None,
                                     eval=0, turn=0)
                self.say.fire("game_start")


class NoArm:
    """--no-arm rehearsal: stands in for MaxArm, moves nothing."""

    def pick_and_drop(self, col, pick_index=0):
        print(f"[no-arm] would pick_and_drop({col}); waiting as if moving")
        time.sleep(3)


def main():
    cfg = load_config()
    try:
        calib = Calibration.load()
    except FileNotFoundError:
        raise SystemExit("calibration.json missing — run calibrate.py against "
                         "the rig first (click the 4 board corners)")

    from commentary import Commentator
    from overlay import Overlay
    from vision import Vision

    if "--no-arm" in sys.argv:
        print("*** NO-ARM REHEARSAL MODE: the robot will not move ***")
        arm = NoArm()
    else:
        from maxarm import MaxArm

        arm = MaxArm(port=cfg.get("port"))
        print("waiting for arm...")
        if arm.wait_ready() is None:
            raise SystemExit("arm did not answer — is it powered and plugged in?")
        arm.confirm_workspace()

    from voice import Voice

    vision = Vision(cfg)
    vision.start()
    overlay = Overlay(calib=calib, show_fps=cfg.get("show_fps", True))
    voice = Voice(cfg)

    def on_line(text):  # every spoken line hits the ticker and TTS together
        overlay.publish(ticker=text)
        voice.speak(text)

    commentator = Commentator(cfg, on_line=on_line)
    game = Game(vision, arm, overlay, commentator, calib)

    def pump():  # keep newest detections flowing onto the overlay
        while True:
            _, dets, ifps = vision.detections()
            overlay.publish(dets=dets, infer_fps=ifps)
            time.sleep(0.03)

    threading.Thread(target=game.run, daemon=True).start()
    threading.Thread(target=pump, daemon=True).start()
    try:
        overlay.run(vision.frame)  # blocks on the main thread until q
    finally:
        vision.stop()


if __name__ == "__main__":
    main()
