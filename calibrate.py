"""Camera calibration: click the 4 corners of the play area, save the
homography + cell geometry + inference crop box to calibration.json.

OWNER-RUN against the physical rig (plan rule). Usage:

    .venv/bin/python calibrate.py

Click the OUTER corners of the 7x6 play area in this order:
top-left, top-right, bottom-right, bottom-left (as seen on screen).
After the 4th click a grid preview is drawn over the feed — check that the
lines land on the board's cell boundaries.

Keys:  u undo last click   r reset   x rotate grid (if 7x6 shows as 6x7)
       s save   q quit without saving

Also locks exposure/white balance where the backend allows it, so
detections don't flicker (plan §Vision).
"""

import json
from pathlib import Path

import cv2
import numpy as np

from board import COLS, ROWS

CELL = 100  # rectified board space: 100 px per cell -> 700x600 board
CROP_MARGIN = 60  # px margin around the corners for the inference crop
OUT = Path(__file__).parent / "calibration.json"


def load_config():
    import yaml

    p = Path(__file__).parent / "config.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def lock_camera(cap):
    """Best effort: kill auto-exposure/WB so detections don't flicker."""
    for prop, val, name in [
        (cv2.CAP_PROP_AUTO_EXPOSURE, 0.25, "auto-exposure off"),
        (cv2.CAP_PROP_AUTO_WB, 0, "auto white balance off"),
    ]:
        ok = cap.set(prop, val)
        print(f"  {name}: {'ok' if ok else 'not supported by this backend'}")


def main():
    cfg = load_config()
    cam_index = cfg.get("camera_index", 0)
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit(f"camera {cam_index} did not open")
    print("locking camera controls:")
    lock_camera(cap)

    clicks = []
    rot = 0  # x key: which clicked corner counts as top-left (fixes rotated grids)

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))
            print(f"corner {len(clicks)}: ({x}, {y})")

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_mouse)

    H = None
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        vis = frame.copy()
        for i, (x, y) in enumerate(clicks):
            cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(vis, str(i + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if len(clicks) == 4:
            ordered = clicks[rot:] + clicks[:rot]
            src = np.array(ordered, dtype=np.float32)
            dst = np.array([[0, 0], [COLS * CELL, 0],
                            [COLS * CELL, ROWS * CELL], [0, ROWS * CELL]],
                           dtype=np.float32)
            H = cv2.getPerspectiveTransform(src, dst)
            Hinv = np.linalg.inv(H)
            for c in range(COLS + 1):  # grid preview via the homography
                a = Hinv @ np.array([c * CELL, 0, 1])
                b = Hinv @ np.array([c * CELL, ROWS * CELL, 1])
                cv2.line(vis, (int(a[0] / a[2]), int(a[1] / a[2])),
                         (int(b[0] / b[2]), int(b[1] / b[2])), (0, 200, 255), 2)
            for r in range(ROWS + 1):
                a = Hinv @ np.array([0, r * CELL, 1])
                b = Hinv @ np.array([COLS * CELL, r * CELL, 1])
                cv2.line(vis, (int(a[0] / a[2]), int(a[1] / a[2])),
                         (int(b[0] / b[2]), int(b[1] / b[2])), (0, 200, 255), 2)
            cv2.putText(vis, "7 wide x 6 tall? press s to save. wrong way round? press x",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(vis, f"click corner {len(clicks) + 1}/4 "
                        "(TL, TR, BR, BL of play area)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow("calibrate", vis)

        k = cv2.waitKey(16) & 0xFF
        if k == ord("u") and clicks:
            clicks.pop()
            H = None
        elif k == ord("r"):
            clicks.clear()
            H = None
        elif k == ord("x") and len(clicks) == 4:
            rot = (rot + 1) % 4
            print(f"grid rotated (corner order shift {rot})")
        elif k == ord("s") and H is not None:
            xs = [p[0] for p in clicks]
            ys = [p[1] for p in clicks]
            h, w = frame.shape[:2]
            crop = [max(0, min(xs) - CROP_MARGIN), max(0, min(ys) - CROP_MARGIN),
                    min(w, max(xs) + CROP_MARGIN), min(h, max(ys) + CROP_MARGIN)]
            OUT.write_text(json.dumps({
                "homography": H.tolist(),
                "cell_w": CELL, "cell_h": CELL,
                "corners": clicks, "crop": crop,
                "frame_size": [w, h],
            }, indent=2) + "\n")
            print(f"saved {OUT} (crop box {crop})")
            break
        elif k == ord("q"):
            print("quit without saving")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
