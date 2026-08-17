"""Engine gate (plan §build-order 2): across 20+ games vs a random opponent,
the engine never misses a win-in-1 and never fails to block a lone block-in-1.
Plus legality and speed checks."""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from engine import DEPTH, WIN, _to_bitboards, _won, best_move


def drop(board, col, p):
    for r in range(6):
        if board[r][col] == 0:
            board[r][col] = p
            return True
    return False


def undo(board, col):
    for r in reversed(range(6)):
        if board[r][col]:
            board[r][col] = 0
            return


def won(board, p):
    me, _, _ = _to_bitboards(board, p)
    return _won(me)


def legal(board):
    return [c for c in range(7) if board[5][c] == 0]


def wins_in_1(board, p):
    out = []
    for c in legal(board):
        drop(board, c, p)
        if won(board, p):
            out.append(c)
        undo(board, c)
    return out


def test_gate_20_games_no_missed_win_or_block():
    rng = random.Random(42)
    for game in range(20):
        board = [[0] * 7 for _ in range(6)]
        turn = rng.choice([1, 2])  # engine is player 2
        while True:
            if not legal(board):
                break
            if turn == 1:
                drop(board, rng.choice(legal(board)), 1)
            else:
                my_wins = wins_in_1(board, 2)
                threats = wins_in_1(board, 1)
                col, score = best_move(board, 2)
                assert board[5][col] == 0, f"illegal move in game {game}"
                if my_wins:
                    assert col in my_wins, (
                        f"game {game}: missed win-in-1 {my_wins}, played {col}"
                    )
                elif len(threats) == 1:
                    assert col == threats[0], (
                        f"game {game}: missed block-in-1 {threats[0]}, played {col}"
                    )
                drop(board, col, 2)
            if won(board, turn):
                break
            turn = 3 - turn


def test_forced_win_reported_in_eval():
    # three in a row on the bottom: engine to move must win at col 3 or 0...
    # here 0,1,2 filled -> winning drop is col 3
    board = [[0] * 7 for _ in range(6)]
    for c in range(3):
        board[0][c] = 2
        board[1][c] = 1
    col, score = best_move(board, 2)
    assert col == 3
    assert score >= WIN


def test_block_simple_threat():
    board = [[0] * 7 for _ in range(6)]
    for c in range(3):
        board[0][c] = 1  # human threatens col 3
    col, _ = best_move(board, 2)
    assert col == 3


def test_full_column_never_chosen():
    board = [[0] * 7 for _ in range(6)]
    for r in range(6):
        board[r][3] = 1 if r % 2 else 2  # center full
    col, _ = best_move(board, 2)
    assert col != 3


def test_speed_under_target():
    board = [[0] * 7 for _ in range(6)]
    t0 = time.perf_counter()
    best_move(board, 2)  # empty board = worst case breadth
    dt = time.perf_counter() - t0
    # plan target is 100 ms; allow slack for slow CI but flag regressions
    assert dt < 1.0, f"depth-{DEPTH} search took {dt * 1000:.0f} ms"
    print(f"empty-board search: {dt * 1000:.0f} ms")
