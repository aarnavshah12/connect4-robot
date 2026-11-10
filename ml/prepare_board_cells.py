"""Prepare full calibrated-cell crops from a COCO export.

Unlike tight object boxes, these crops match the 42-cell runtime reader. The
export already labels each hole, so annotation centers can locate the grid in
each photo. Images with missing or inconsistent cell geometry are skipped.
Corrections are explicit and reproducible in label_fixes.json; raw data is not
modified. Existing train/valid/test image assignments are preserved.
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ml.prepare_cells import LABELS

BOARD_SIZE = (700, 600)
CELL = 100


def _grid(annotations):
    """Return (homography, annotation->row/col) for a complete 7x6 grid."""
    if len(annotations) != 42:
        return None
    points = np.float32([
        (ann["bbox"][0] + ann["bbox"][2] / 2,
         ann["bbox"][1] + ann["bbox"][3] / 2) for ann in annotations
    ])
    x, y = points[:, 0], points[:, 1]
    corners = [int(np.argmin(x + y)), int(np.argmax(x - y)),
               int(np.argmax(x + y)), int(np.argmin(x - y))]
    if len(set(corners)) != 4:
        return None
    targets = np.float32([[50, 50], [650, 50], [650, 550], [50, 550]])
    first = cv2.getPerspectiveTransform(points[corners], targets)
    projected = cv2.perspectiveTransform(points[:, None, :], first)[:, 0, :]
    columns = np.rint(projected[:, 0] / CELL - 0.5).astype(int)
    rows = np.rint(projected[:, 1] / CELL - 0.5).astype(int)
    cells = list(zip(rows.tolist(), columns.tolist()))
    if (len(set(cells)) != 42 or any(not (0 <= row < 6 and 0 <= col < 7)
                                    for row, col in cells)):
        return None
    ideal = np.float32([((col + 0.5) * CELL, (row + 0.5) * CELL)
                        for row, col in cells])
    refined, _ = cv2.findHomography(points, ideal)
    if refined is None:
        return None
    error = np.linalg.norm(cv2.perspectiveTransform(points[:, None, :], refined)[:, 0, :] - ideal,
                           axis=1)
    if float(error.max()) > 20:
        return None
    return refined, cells


def prepare(source, output, fixes_path=None, object_crops=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f"output exists: {output}; choose a new experiment directory")
    fixes = json.loads(Path(fixes_path).read_text()) if fixes_path else {}
    used_fixes = set()
    counts, skipped, tasks = defaultdict(int), [], []
    for raw_split, split in (("train", "train"), ("valid", "val"), ("test", "test")):
        folder = source / raw_split
        coco = json.loads((folder / "_annotations.coco.json").read_text())
        categories = {cat["id"]: cat["name"].strip().lower() for cat in coco["categories"]}
        grouped = defaultdict(list)
        for ann in coco["annotations"]:
            if LABELS.get(categories[ann["category_id"]]) is not None:
                grouped[ann["image_id"]].append(ann)
        for info in coco["images"]:
            annotations = grouped[info["id"]]
            if not annotations:
                skipped.append(f"{raw_split}/{info['file_name']}: no labeled cells")
                continue
            areas = np.asarray([ann["bbox"][2] * ann["bbox"][3] for ann in annotations])
            median = float(np.median(areas))
            reasonable = [ann for ann, area in zip(annotations, areas)
                          if 0.25 * median < area < 4 * median]
            found = _grid(reasonable)
            if found is None:
                skipped.append(f"{raw_split}/{info['file_name']}: incomplete/invalid grid "
                               f"({len(reasonable)} cell boxes)")
                continue
            path = (folder / info["file_name"]).resolve()
            if not path.is_relative_to(folder.resolve()):
                raise ValueError(f"image path escapes its split: {path}")
            tasks.append((path, split, categories, reasonable, found))
    if not tasks:
        raise ValueError("no complete labeled boards in this export")
    output.mkdir(parents=True)
    for path, split, categories, annotations, (homography, cells) in tasks:
        original = cv2.imread(str(path))
        if original is None:
            raise ValueError(f"cannot read {path}")
        aligned = cv2.warpPerspective(original, homography, BOARD_SIZE)
        image_id = annotations[0]["image_id"]
        for ann, (row, col) in zip(annotations, cells):
            label = LABELS[categories[ann["category_id"]]]
            key = f"{split}/{label}/{image_id}_{ann['id']}.png"
            if key in fixes:
                used_fixes.add(key)
                label = fixes[key]
            if label not in ("empty", "red", "yellow", "occluded"):
                raise ValueError(f"invalid corrected label: {label}")
            destination = output / split / label / f"board_{image_id}_{row}_{col}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            crop = aligned[row * CELL:(row + 1) * CELL,
                           col * CELL:(col + 1) * CELL]
            if not cv2.imwrite(str(destination), crop):
                raise OSError(f"could not write {destination}")
            counts[f"{split}/{label}"] += 1
    if object_crops:
        for label in ("empty", "red", "yellow", "occluded"):
            for path in (Path(object_crops) / "train" / label).glob("*.png"):
                key = f"train/{label}/{path.name}"
                if key in fixes:
                    used_fixes.add(key)
                actual = fixes.get(key, label)
                destination = output / "train" / actual / f"box_{path.name}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
                counts[f"train/{actual}"] += 1
    unknown = set(fixes) - used_fixes
    if unknown:
        raise ValueError(f"corrections did not match source crops: {sorted(unknown)}")
    report = {"counts": dict(counts), "boards": len(tasks), "skipped": skipped,
              "corrections": sorted(used_fixes), "object_crops_included": bool(object_crops)}
    (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="data/cells-v2")
    parser.add_argument("--object-crops", help="optional original train object crops")
    parser.add_argument("--fixes", help="optional JSON of visually reviewed corrections")
    args = parser.parse_args()
    prepare(args.source, args.output, args.fixes, args.object_crops)


if __name__ == "__main__":
    main()
