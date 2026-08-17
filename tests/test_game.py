"""State machine tests with scripted vision, fake arm, recording commentator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import board as B
from board import HUMAN, ROBOT, Calibration
from game import Game

CAL = Calibration([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 100, 100)


def dets_for(bd):
    """Turn a board into perfect synthetic detections."""
    out = []
    for r in range(B.ROWS):
        for c in range(B.COLS):
            if bd[r][c] != B.EMPTY:
                x, y = CAL.cell_center_frame(r, c)
                out.append((x, y, 70, 70, "red" if bd[r][c] == ROBOT else "blue", 0.95))
    return out


class ScriptedVision:
    """Yields each scripted board for enough frames to pass the debouncer,
    then holds the final board forever."""

    def __init__(self, boards, frames_each=6):
        self.frames = []
        for bd in boards:
            self.frames += [bd] * frames_each
        self.i = 0

    def detections(self):
        bd = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return (self.i, dets_for(bd), 20.0)


class FakeArm:
    def __init__(self, vision_script=None):
        self.dropped = []
        self.on_drop = None

    def pick_and_drop(self, col):
        self.dropped.append(col)
        if self.on_drop:
            self.on_drop(col)


class NullOverlay:
    def __init__(self):
        self.published = []

    def publish(self, **kw):
        self.published.append(kw)

    def ghost_drop(self, *a, **kw):
        pass


class RecordingSay:
    def __init__(self):
        self.triggers = []

    def fire(self, trigger, **ctx):
        self.triggers.append(trigger)


def make_game(boards):
    vision = ScriptedVision(boards)
    arm = FakeArm()
    overlay = NullOverlay()
    say = RecordingSay()
    game = Game(vision, arm, overlay, say, CAL, sleep=lambda s: None)
    return game, vision, arm, overlay, say


def with_piece(bd, col, player):
    new = [row[:] for row in bd]
    new[B.landing_row(new, col)][col] = player
    return new


def test_full_turn_human_then_robot():
    b0 = B.empty_board()
    b1 = with_piece(b0, 3, HUMAN)  # human plays center
    game, vision, arm, overlay, say = make_game([b0, b1])

    # after the human's piece appears, the robot must answer with its move
    # and VERIFYING must see the robot piece; script that board once we know col
    def on_drop(col):
        vision.frames += [with_piece(game.board, col, ROBOT)] * 10

    arm.on_drop = on_drop
    result = game.play_turn()
    assert result == "ok"
    assert len(arm.dropped) == 1
    assert game.board[0][3] == HUMAN
    r, c = next((r, c) for r in range(6) for c in range(7)
                if game.board[r][c] == ROBOT)
    assert c == arm.dropped[0]
    states = [p["state"] for p in overlay.published if "state" in p]
    for expected in ["WAIT_HUMAN", "SCANNING", "THINKING", "ROBOT_MOVING", "VERIFYING"]:
        assert expected in states
    assert any(t in ("jab", "respect", "blunder") for t in say.triggers)


def test_illegal_move_enters_error_then_recovers():
    b0 = B.empty_board()
    bad = with_piece(with_piece(b0, 2, HUMAN), 4, HUMAN)  # two new pieces at once
    game, vision, arm, overlay, say = make_game([b0, bad, b0])
    result = game.play_turn()
    assert result == "error"
    assert arm.dropped == []
    states = [p["state"] for p in overlay.published if "state" in p]
    assert "ERROR" in states
    assert states[-1] == "WAIT_HUMAN"  # recovered once the board went back


def test_human_win_detected():
    b = B.empty_board()
    for c in range(3):
        b[0][c] = HUMAN
        b[1][c] = ROBOT
    game, vision, arm, overlay, say = make_game([b, with_piece(b, 3, HUMAN)])
    game.board = [row[:] for row in b]
    result = game.play_turn()
    assert result == "human_win"
    assert "human_win" in say.triggers
    assert arm.dropped == []


def test_robot_wins_and_gloats():
    # robot has three on the bottom row 0..2, human elsewhere; any human move
    # that isn't a block lets the engine finish at col 3
    b = B.empty_board()
    for c in range(3):
        b[0][c] = ROBOT
    b[0][5] = HUMAN
    b[0][6] = HUMAN
    game, vision, arm, overlay, say = make_game([b, with_piece(b, 6, HUMAN)])
    game.board = [row[:] for row in b]

    def on_drop(col):
        vision = game.vision
        vision.frames += [with_piece(game.board, col, ROBOT)] * 10

    arm.on_drop = on_drop
    result = game.play_turn()
    assert result == "robot_win"
    assert arm.dropped == [3]
    assert "robot_win" in say.triggers
