"""Try the PyTorch cell classifier on one board photo; no camera or arm.

Run:  .venv/bin/python vision_photo.py
Choose a JPG/PNG, then click the OUTER board corners in this order:
top-left, top-right, bottom-right, bottom-left. Press R to retry or Q to quit.
For a photo from the already calibrated camera, use --use-calibration instead.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

import board as B
from ml.cells import CellReader, cell_crops, image_tensor, observation


ROOT = Path(__file__).resolve().parent
RECTIFIED_SIZE = (700, 600)
CORNER_NAMES = ("top-left", "top-right", "bottom-right", "bottom-left")


def choose_photo():
    """Use macOS's native file picker without adding a GUI dependency."""
    if sys.platform != "darwin":
        raise SystemExit("Pass a photo path with --image on this operating system.")
    result = subprocess.run(
        ["osascript", "-e", 'POSIX path of (choose file with prompt "Choose a Connect 4 board photo")'],
        capture_output=True, text=True, check=False)
    if result.returncode:
        raise SystemExit("No photo selected. Pass --image /path/to/photo.jpg to try again.")
    return Path(result.stdout.strip())


def click_corners(photo):
    """Collect four outer board corners, scaled back to source-image pixels."""
    height, width = photo.shape[:2]
    scale = min(1.0, 1100 / width, 750 / height)
    shown = cv2.resize(photo, (max(1, round(width * scale)),
                               max(1, round(height * scale))))
    points = []
    window = "Mark board corners (R retry, Q quit)"

    def mouse(event, x, y, _flags, _data):
        if event == cv2.EVENT_LBUTTONDOWN and y >= 44 and len(points) < 4:
            points.append((x / scale, (y - 44) / scale))

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, mouse)
    try:
        while len(points) < 4:
            display = cv2.copyMakeBorder(shown, 44, 0, 0, 0,
                                         cv2.BORDER_CONSTANT, value=(24, 30, 42))
            for i, (x, y) in enumerate(points):
                at = (round(x * scale), 44 + round(y * scale))
                cv2.circle(display, at, 7, (0, 230, 255), -1)
                cv2.putText(display, str(i + 1), (at[0] + 9, at[1] - 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2)
            for i in range(1, len(points)):
                a = (round(points[i - 1][0] * scale), 44 + round(points[i - 1][1] * scale))
                b = (round(points[i][0] * scale), 44 + round(points[i][1] * scale))
                cv2.line(display, a, b, (0, 230, 255), 2)
            instruction = f"Click OUTER {CORNER_NAMES[len(points)]} board corner ({len(points) + 1}/4)"
            cv2.putText(display, instruction, (12, 27), cv2.FONT_HERSHEY_SIMPLEX,
                        0.63, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                raise SystemExit("Corner selection cancelled.")
            if key == ord("r"):
                points.clear()
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    raise SystemExit("Corner selection cancelled.")
            except cv2.error:
                raise SystemExit("Corner selection cancelled.") from None
    finally:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass
    return points


def calibration_from_corners(points):
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(points).all() or abs(cv2.contourArea(points)) < 100:
        raise ValueError("the four board corners must form a nonempty quadrilateral")
    target = np.float32([[0, 0], [700, 0], [700, 600], [0, 600]])
    homography = cv2.getPerspectiveTransform(points, target)
    return B.Calibration(homography, cell_w=100, cell_h=100)


def analyze(photo, calibration, checkpoint, threshold=0.90):
    """Return the annotated board preview and readable prediction details."""
    reader = CellReader(checkpoint, calibration, device="cpu", threshold=threshold)
    images = torch.stack([image_tensor(crop) for crop in cell_crops(photo, calibration)])
    with torch.inference_mode():
        probabilities = reader.model(images).softmax(dim=1).cpu().numpy()
    if probabilities.shape != (42, 4) or not np.isfinite(probabilities).all():
        raise ValueError("model produced invalid cell probabilities")

    labels = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    uncertain = (labels == 3) | (confidence < threshold)
    detections = observation(probabilities, calibration, threshold)
    board = None if detections is None else B.parse_detections(
        detections, calibration, min_confidence=0.0)
    rows = []
    for row in reversed(range(6)):
        rows.append(" ".join("?" if uncertain[row * 7 + col] else ".RY?"[labels[row * 7 + col]]
                             for col in range(7)))
    low = int((confidence < threshold).sum())
    occluded = int((labels == 3).sum())
    if detections is None:
        status = f"REJECTED: {low} low-confidence and {occluded} occluded cells"
    elif board is None:
        status = "REJECTED: predicted board violates gravity/board rules"
    else:
        status = "ACCEPTED: all 42 cells confident and board is legal"

    preview = cv2.warpPerspective(photo, calibration.H, RECTIFIED_SIZE)
    for row in range(6):
        for col in range(7):
            index = row * 7 + col
            top = 5 - row
            x0, y0 = col * 100, top * 100
            color = (0, 150, 255) if uncertain[index] else (65, 220, 70)
            cv2.rectangle(preview, (x0 + 2, y0 + 2), (x0 + 98, y0 + 98), color, 2)
            text = f"{'.RY?'[labels[index]]} {confidence[index]:.0%}"
            cv2.rectangle(preview, (x0 + 5, y0 + 6), (x0 + 95, y0 + 31),
                          (25, 25, 30), -1)
            cv2.putText(preview, text, (x0 + 9, y0 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.49, color, 1, cv2.LINE_AA)
    return preview, rows, status, float(confidence.min())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="JPG/PNG path; opens a picker if omitted on macOS")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/vision/best.pt")
    parser.add_argument("--threshold", type=float, default=0.90)
    geometry = parser.add_mutually_exclusive_group()
    geometry.add_argument("--use-calibration", action="store_true",
                          help="use this rig's saved calibration for a same-camera photo")
    geometry.add_argument("--corners", nargs=8, type=float,
                          metavar=("TL_X", "TL_Y", "TR_X", "TR_Y", "BR_X", "BR_Y", "BL_X", "BL_Y"),
                          help="four outer corners, for scripting without clicking")
    parser.add_argument("--calibration", type=Path, default=ROOT / "calibration.json")
    parser.add_argument("--output", type=Path, help="optional annotated PNG to save")
    parser.add_argument("--no-preview", action="store_true", help="print results without a result window")
    args = parser.parse_args()

    path = args.image or choose_photo()
    photo = cv2.imread(str(path))
    if photo is None:
        raise SystemExit(f"Could not read {path}. Use a JPG or PNG photo.")
    if args.use_calibration:
        calibration = B.Calibration.load(args.calibration)
    else:
        points = np.reshape(args.corners, (4, 2)) if args.corners else click_corners(photo)
        calibration = calibration_from_corners(points)
    preview, rows, status, minimum = analyze(photo, calibration, args.checkpoint,
                                             args.threshold)
    print("Predicted board (top row first; . empty, R red, Y yellow, ? uncertain):")
    print("\n".join(rows))
    print(status)
    print(f"Lowest cell confidence: {minimum:.1%}")
    if args.output:
        if not cv2.imwrite(str(args.output), preview):
            raise SystemExit(f"Could not save preview to {args.output}")
        print(f"Saved annotated preview: {args.output}")
    if not args.no_preview:
        cv2.imshow("PyTorch vision: orange cells are uncertain (press any key)", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
