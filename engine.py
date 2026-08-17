"""Connect 4 move engine: negamax + alpha-beta, depth 8, bitboards.

Public API (the only thing game.py may rely on):

    best_move(board, player) -> (column, eval_score)

`board` is 6 rows x 7 cols, board[row][col], row 0 = BOTTOM. Cells are
0 empty, 1 and 2 the two players; `player` (1 or 2) is the side to move.
eval_score is from `player`'s perspective and drives the overlay's
advantage bar. Never returns an illegal move by construction.

Optional kwarg `considered`: a list that receives (column, score) per root
candidate as the search finishes it — feeds the overlay's THINKING flash.

Bitboard layout (Pascal Pons style): bit index = col*7 + row, row 0 bottom,
bit 6 of each column is a sentinel kept empty so shift-based win checks
don't wrap between columns.
"""

DEPTH = 8
WIN = 1_000_000
ORDER = (3, 2, 4, 1, 5, 0, 6)  # center first: best pruning, best play
CELL_SCORE = {1: 1, 2: 10, 3: 60}
CENTER_W = 6

TOP_SENTINELS = sum(1 << (c * 7 + 6) for c in range(7))
CENTER_MASK = sum(1 << (3 * 7 + r) for r in range(6))
FULL_TOP = sum(1 << (c * 7 + 5) for c in range(7))


def _windows():
    idx = lambda c, r: c * 7 + r
    wins = []
    for c in range(7):
        for r in range(6):
            if c <= 3:
                wins.append([idx(c + i, r) for i in range(4)])
            if r <= 2:
                wins.append([idx(c, r + i) for i in range(4)])
            if c <= 3 and r <= 2:
                wins.append([idx(c + i, r + i) for i in range(4)])
            if c <= 3 and r >= 3:
                wins.append([idx(c + i, r - i) for i in range(4)])
    return tuple(sum(1 << b for b in w) for w in wins)


WINDOWS = _windows()


def _won(bb):
    for shift in (1, 7, 6, 8):
        m = bb & (bb >> shift)
        if m & (m >> (2 * shift)):
            return True
    return False


def _evaluate(me, opp):
    score = CENTER_W * ((me & CENTER_MASK).bit_count() - (opp & CENTER_MASK).bit_count())
    for m in WINDOWS:
        mc = (me & m).bit_count()
        if mc:
            if not opp & m:
                score += CELL_SCORE[mc]
        else:
            oc = (opp & m).bit_count()
            if oc:
                score -= CELL_SCORE[oc]
    return score


def _heights(mask):
    return [(mask >> (c * 7)) & 0x7F for c in range(7)]


def _drop(bb, mask, col):
    """Return (new_bb, new_mask) after the side `bb` plays `col`."""
    new = mask | (mask + (1 << (col * 7)))
    return bb | (new ^ mask), new


def _playable(mask, col):
    return not mask & (1 << (col * 7 + 5))


def _negamax(me, mask, depth, alpha, beta, table):
    key = (me, mask, depth)
    hit = table.get(key)
    if hit is not None:
        flag, val = hit
        if (flag == 0 or (flag == 1 and val >= beta) or (flag == 2 and val <= alpha)):
            return val

    alpha0 = alpha
    best = None
    for col in ORDER:
        if not _playable(mask, col):
            continue
        new_me, new_mask = _drop(me, mask, col)
        if _won(new_me):
            score = WIN + depth  # prefer faster wins
        elif depth == 1:
            score = _evaluate(new_me, new_mask ^ new_me)
        else:
            score = -_negamax(new_mask ^ new_me, new_mask, depth - 1, -beta, -alpha, table)
        if best is None or score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break

    if best is None:  # no legal moves: draw
        best = 0
    # 0 exact, 1 lower bound (beta cutoff), 2 upper bound (failed low)
    flag = 1 if best >= beta else 2 if best <= alpha0 else 0
    table[key] = (flag, best)
    return best


def _to_bitboards(board, player):
    me = opp = mask = 0
    for r in range(6):
        for c in range(7):
            cell = board[r][c]
            if cell:
                bit = 1 << (c * 7 + r)
                mask |= bit
                if cell == player:
                    me |= bit
                else:
                    opp |= bit
    return me, opp, mask


def best_move(board, player, considered=None):
    """(column, eval_score) for `player` on `board`. See module docstring."""
    me, opp, mask = _to_bitboards(board, player)
    table = {}
    best_col, best_score, alpha = None, None, -WIN * 2
    for col in ORDER:
        if not _playable(mask, col):
            continue
        new_me, new_mask = _drop(me, mask, col)
        if _won(new_me):
            score = WIN + DEPTH
        else:
            score = -_negamax(new_mask ^ new_me, new_mask, DEPTH - 1, -WIN * 2, -alpha, table)
        if considered is not None:
            considered.append((col, score))
        if best_score is None or score > best_score:
            best_col, best_score = col, score
        if best_score > alpha:
            alpha = best_score
    if best_col is None:
        raise ValueError("no legal moves: board is full")
    return best_col, best_score


if __name__ == "__main__":
    # terminal self-test: play against the engine (you are player 1, X)
    import time

    board = [[0] * 7 for _ in range(6)]

    def show():
        for r in reversed(range(6)):
            print(" ".join(".XO"[board[r][c]] for c in range(7)))
        print("0 1 2 3 4 5 6")

    def drop2d(col, p):
        for r in range(6):
            if board[r][col] == 0:
                board[r][col] = p
                return r
        return None

    def won2d(p):
        me, _, _ = _to_bitboards(board, p)
        return _won(me)

    turn = 1
    while True:
        show()
        if turn == 1:
            col = int(input("your column (0-6): "))
            if drop2d(col, 1) is None:
                print("column full")
                continue
        else:
            t0 = time.perf_counter()
            col, score = best_move(board, 2)
            dt = (time.perf_counter() - t0) * 1000
            drop2d(col, 2)
            print(f"engine: column {col}  (eval {score:+d}, {dt:.0f} ms)")
        if won2d(turn):
            show()
            print("you win" if turn == 1 else "engine wins")
            break
        if all(board[5][c] for c in range(7)):
            show()
            print("draw")
            break
        turn = 3 - turn
