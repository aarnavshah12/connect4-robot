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


def det(row, col, cls="red piece", conf=0.9, jitter=(0, 0)):
    x, y = at(row, col)
    return (x + jitter[0], y + jitter[1], 70, 70, cls, conf)


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
    b = parse_detections([det(0, 3, "red piece"), det(0, 4, "yellow piece")], CAL)
    assert b[0][3] == ROBOT and b[0][4] == HUMAN


def test_parse_drops_low_conf_and_unknown_class():
    # 0.05 is under even red's eval-derived 0.12 threshold
    b = parse_detections([det(0, 3, conf=0.05), det(0, 4, "hand", 0.99)], CAL)
    assert b == empty_board()


def test_per_class_thresholds():
    # red at 0.45 passes (threshold 0.35); yellow at 0.45 is dropped (0.55 —
    # raised after live phantom-yellow hallucinations)
    b = parse_detections([det(0, 3, "red piece", 0.45), det(0, 4, "yellow piece", 0.45)], CAL)
    assert b[0][3] == ROBOT and b[0][4] == 0
    # model's Board / No Piece classes are ignored even at high confidence
    b2 = parse_detections([det(0, 3, "board", 0.99), det(0, 4, "no piece", 0.99)], CAL)
    assert b2 == empty_board()


def test_parse_rejects_floating_piece():
    assert parse_detections([det(2, 3)], CAL) is None  # nothing under it


def test_parse_rejects_color_conflict():
    assert parse_detections([det(0, 3, "red piece"), det(0, 3, "yellow piece")], CAL) is None


def test_debounce_needs_five_identical():
    d = Debouncer()
    b = parse_detections([det(0, 0)], CAL)
    for _ in range(4):
        assert d.feed(b) is None
    assert d.feed(b) == b
    d.reset()
    # unusable frames pause the streak, they don't reset it
    assert d.feed(b) is None
    assert d.feed(b) is None
    d.feed(None)
    assert d.feed(b) is None
    assert d.feed(b) is None
    assert d.feed(b) == b  # 5 identical valid reads total
    # a different board DOES reset
    other = parse_detections([det(0, 1)], CAL)
    d.feed(other)
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
