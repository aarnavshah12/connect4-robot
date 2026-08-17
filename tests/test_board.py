import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from board import (
    HUMAN, ROBOT, Calibration, Debouncer, check_win, diff, empty_board,
    landing_row, legal_human_move, parse_detections, physics_ok,
)

# identity homography, 100px cells: frame coords ARE board coords
CAL = Calibration([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 100, 100)


def at(row, col):
    """Frame pixel at the center of (row, col) [row 0 = bottom]."""
    return ((col + 0.5) * 100, (5 - row + 0.5) * 100)


def det(row, col, cls="red", conf=0.9, jitter=(0, 0)):
    x, y = at(row, col)
    return (x + jitter[0], y + jitter[1], cls, conf)


def test_snap_center_and_jitter():
    assert CAL.snap(*at(0, 0)) == (0, 0)
    assert CAL.snap(at(2, 3)[0] + 40, at(2, 3)[1] - 40) == (2, 3)


def test_snap_rejects_far_and_offgrid():
    assert CAL.snap(at(0, 0)[0] + 51, at(0, 0)[1]) == (0, 1)  # next cell over
    assert CAL.snap(-200, -200) is None
    assert CAL.snap(900, 300) is None  # right of the grid


def test_cell_center_roundtrip():
    x, y = CAL.cell_center_frame(4, 6)
    assert CAL.snap(x, y) == (4, 6)


def test_parse_simple():
    b = parse_detections([det(0, 3, "red"), det(0, 4, "blue")], CAL)
    assert b[0][3] == ROBOT and b[0][4] == HUMAN


def test_parse_drops_low_conf_and_unknown_class():
    b = parse_detections([det(0, 3, conf=0.2), det(0, 4, "hand", 0.99)], CAL)
    assert b == empty_board()


def test_parse_rejects_floating_piece():
    assert parse_detections([det(2, 3)], CAL) is None  # nothing under it


def test_parse_rejects_color_conflict():
    assert parse_detections([det(0, 3, "red"), det(0, 3, "blue")], CAL) is None


def test_debounce_needs_five_identical():
    d = Debouncer()
    b = parse_detections([det(0, 0)], CAL)
    for _ in range(4):
        assert d.feed(b) is None
    assert d.feed(b) == b
    d.feed(None)  # unusable frame resets the streak
    assert d.feed(b) is None


def test_diff_and_legal_human_move():
    old = empty_board()
    old[0][3] = ROBOT
    new = [row[:] for row in old]
    new[0][2] = HUMAN
    assert diff(old, new) == [(0, 2, HUMAN)]
    assert legal_human_move(old, new) == (0, 2)


def test_legal_rejects_two_new_or_floating_or_wrong_color():
    old = empty_board()
    two = [row[:] for row in old]
    two[0][1] = HUMAN
    two[0][2] = HUMAN
    assert legal_human_move(old, two) is None
    rob = [row[:] for row in old]
    rob[0][1] = ROBOT
    assert legal_human_move(old, rob) is None


def test_diff_detects_vanished_piece():
    old = empty_board()
    old[0][3] = ROBOT
    assert diff(old, empty_board()) is None


def test_landing_row_and_win():
    b = empty_board()
    for c in range(4):
        b[0][c] = ROBOT
    assert check_win(b, ROBOT) == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert check_win(b, HUMAN) is None
    assert landing_row(b, 0) == 1
    diag = empty_board()
    for i in range(4):
        for r in range(i):
            diag[r][i] = HUMAN
        diag[i][i] = ROBOT
    assert physics_ok(diag)
    assert check_win(diag, ROBOT) == [(0, 0), (1, 1), (2, 2), (3, 3)]
