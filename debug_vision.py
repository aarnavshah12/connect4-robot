"""Vision pipeline X-ray: shows, for every inference result, exactly what the
game's board parser does with each detection and why frames get rejected.

    .venv/bin/python debug_vision.py     (Ctrl+C to stop)

Drop a piece in while it runs, then read the trail:
  - "OFF-GRID" = the centroid warps outside every cell -> recalibrate corners
  - pieces landing in row 5 with empties below = calibration is upside-down
  - "FLOATING" = physics check veto (usually the row-flip above)
  - board flip-flopping between reads = detection jitter resetting the debounce
"""

import time

from board import (COLOR_TO_PLAYER, DEFAULT_MIN_CONFIDENCE, MIN_CONFIDENCE,
                   Calibration, Debouncer, empty_board, physics_ok)
from game import load_config
from vision import Vision


def explain(dets, calib):
    board = empty_board()
    kept, rejected = [], []
    for x, y, w, h, cls, conf in dets:
        thr = MIN_CONFIDENCE.get(cls, DEFAULT_MIN_CONFIDENCE)
        if cls not in COLOR_TO_PLAYER:
            rejected.append(f"{cls}({conf:.2f}) ignored-class")
            continue
        if conf < thr:
            rejected.append(f"{cls}({conf:.2f}) below-thr-{thr}")
            continue
        hit = calib.snap(x, y)
        if hit is None:
            bx, by = calib.warp(x, y)
            rejected.append(f"{cls}({conf:.2f}) OFF-GRID at board-space ({bx:.0f},{by:.0f})")
            continue
        r, c = hit
        player = COLOR_TO_PLAYER[cls]
        if board[r][c] not in (0, player):
            return None, kept, rejected, "CELL CONFLICT (red+yellow in one cell)"
        board[r][c] = player
        kept.append(f"{cls}({conf:.2f})->row{r},col{c}")
    if not physics_ok(board):
        return None, kept, rejected, "FLOATING piece (physics veto — row flip?)"
    return board, kept, rejected, None


def main():
    import cv2

    cfg = load_config()
    calib = Calibration.load()
    vision = Vision(cfg)
    vision.start()
    deb = Debouncer()
    last_fid = 0
    print("watching... drop a piece in. Ctrl+C or press q in the window to stop.")
    try:
        while True:
            # live preview window: camera + grid + piece boxes
            _, frame = vision.frame()
            if frame is not None:
                _, dets_now, _ = vision.detections()
                for x, y, w, h, cls, conf in dets_now:
                    if cls in COLOR_TO_PLAYER:
                        color = (60, 60, 230) if "red" in cls else (0, 205, 235)
                        cv2.rectangle(frame, (int(x - w/2), int(y - h/2)),
                                      (int(x + w/2), int(y + h/2)), color, 2)
                        hit = calib.snap(x, y)
                        cv2.putText(frame, f"{cls} {conf:.2f} -> {hit}",
                                    (int(x - w/2), int(y - h/2) - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                for r in range(7):
                    a = calib.cell_center_frame(r - 0.5, -0.5)
                    b = calib.cell_center_frame(r - 0.5, 6.5)
                    cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (90, 200, 90), 1)
                for c in range(8):
                    a = calib.cell_center_frame(-0.5, c - 0.5)
                    b = calib.cell_center_frame(5.5, c - 0.5)
                    cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (90, 200, 90), 1)
                cv2.imshow("debug (q quits)", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            fid, dets, ifps = vision.detections()
            if fid == last_fid:
                time.sleep(0.05)
                continue
            last_fid = fid
            board, kept, rejected, veto = explain(dets, calib)
            stable = deb.feed(board)
            bits = [f"[{ifps:.0f}fps] {len(dets)} dets"]
            if kept:
                bits.append("kept: " + " ".join(kept))
            if rejected:
                bits.append("rej: " + " ".join(rejected))
            if veto:
                bits.append(f"VETO: {veto}")
            bits.append(f"debounce {deb._count}/{deb.n}" + (" STABLE" if stable else ""))
            print(" | ".join(bits))
    except KeyboardInterrupt:
        pass
    finally:
        vision.stop()


if __name__ == "__main__":
    main()
