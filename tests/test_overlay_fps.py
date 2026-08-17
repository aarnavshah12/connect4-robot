"""Headless proxy for the plan's 30 fps overlay gate: compose() must stay
well under the frame budget with a full worst-case state published (42 boxes,
ghost animation, considered flash, win line, ticker). The real gate — live
camera + inference — is owner-run; this catches compose-side regressions."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from board import Calibration, empty_board
from overlay import FEED_H, FEED_W, Overlay

# compose must leave imshow/waitKey plenty of room inside 33 ms
BUDGET_MEAN_MS = 14.0
BUDGET_P95_MS = 24.0


def test_compose_time_supports_30fps():
    import cv2

    corners = np.float32([[340, 120], [940, 120], [940, 640], [340, 640]])
    dst = np.float32([[0, 0], [700, 0], [700, 600], [0, 600]])
    calib = Calibration(cv2.getPerspectiveTransform(corners, dst).tolist(), 100, 100)

    ov = Overlay(calib=calib)
    board = empty_board()
    dets = []
    for r in range(6):
        for c in range(7):
            board[r][c] = 1 if (r + c) % 2 else 2
            x, y = calib.cell_center_frame(r, c)
            dets.append((x, y, 70, 70, "red piece" if board[r][c] == 1 else "yellow piece", 0.9))
    ov.publish(state="THINKING", board=board, dets=dets, eval=250,
               considered=[(3, 120), (4, 90), (2, -40), (5, 10)],
               history=[f"{i}. you c{i % 7} / bot c{(i + 3) % 7}" for i in range(8)],
               ticker="column four. obviously.", turn=9,
               win_cells=[(0, 0), (1, 1), (2, 2), (3, 3)])
    ov.ghost_drop(3, 2, duration=999)  # keep the ghost mid-flight every frame

    frame = np.random.randint(0, 255, (FEED_H, FEED_W, 3), dtype=np.uint8)
    for _ in range(10):  # warmup
        ov.compose(frame)
    times = []
    for _ in range(120):
        t0 = time.perf_counter()
        ov.compose(frame)
        times.append((time.perf_counter() - t0) * 1000)
    mean = sum(times) / len(times)
    p95 = sorted(times)[int(len(times) * 0.95)]
    print(f"compose: mean {mean:.1f} ms, p95 {p95:.1f} ms")
    assert mean < BUDGET_MEAN_MS, f"compose mean {mean:.1f} ms blows the 30fps budget"
    assert p95 < BUDGET_P95_MS, f"compose p95 {p95:.1f} ms blows the 30fps budget"
