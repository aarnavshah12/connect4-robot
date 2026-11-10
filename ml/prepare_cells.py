"""Turn a Roboflow COCO export into labeled cell crops.

python -m ml.prepare_cells --source data/raw/connect4 --output data/cells

Existing train/valid/test splits are preserved. Re-export with session-based
splits first if adjacent video frames currently occur in different splits.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2

from ml.cells import CLASS_NAMES

LABELS = {
    "no piece": "empty", "empty": "empty",
    "red piece": "red", "red": "red",
    "yellow piece": "yellow", "yellow": "yellow",
    "occluded": "occluded", "blocked": "occluded",
}


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f"output already exists; choose a new directory: {output}")
    crops, counts, seen = [], {}, {}
    for raw_split, split in (("train", "train"), ("valid", "val"), ("test", "test")):
        folder = source / raw_split
        if not folder.exists() and raw_split == "valid":
            folder = source / "val"
        annotation = folder / "_annotations.coco.json"
        if not annotation.exists():
            raise ValueError(f"missing {annotation}; export COCO JSON with all three splits")
        coco = json.loads(annotation.read_text())
        categories = {c["id"]: c["name"].strip().lower() for c in coco["categories"]}
        images = {i["id"]: i for i in coco["images"]}
        by_image = {}
        for item in coco["annotations"]:
            label = LABELS.get(categories[item["category_id"]])
            if label is not None:
                by_image.setdefault(item["image_id"], []).append((item, label))
        for image_id, items in by_image.items():
            path = (folder / images[image_id]["file_name"]).resolve()
            if not path.is_relative_to(folder.resolve()):
                raise ValueError(f"image path escapes its split: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen and seen[digest] != split:
                raise ValueError(f"identical source image in {seen[digest]} and {split}: {path.name}")
            seen[digest] = split
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"cannot read {path}")
            height, width = image.shape[:2]
            for item, label in items:
                x, y, w, h = item["bbox"]
                x0, y0 = max(0, int(x)), max(0, int(y))
                x1, y1 = min(width, round(x + w)), min(height, round(y + h))
                if x1 <= x0 or y1 <= y0:
                    raise ValueError(f"empty bounding box in {path.name}: {item['bbox']}")
                destination = output / split / label / f"{image_id}_{item['id']}.png"
                crops.append((path, (x0, y0, x1, y1), destination, split, label))
                counts[f"{split}/{label}"] = counts.get(f"{split}/{label}", 0) + 1
    for split in ("train", "val", "test"):
        for label in CLASS_NAMES[:3]:
            if not counts.get(f"{split}/{label}"):
                raise ValueError(f"{split} has no {label} examples; empty cells need explicit annotations")
    # Validate everything before writing. Reload each image once while exporting.
    output.mkdir(parents=True)
    current, image = None, None
    manifest = []
    for path, (x0, y0, x1, y1), destination, split, label in crops:
        if path != current:
            current, image = path, cv2.imread(str(path))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), image[y0:y1, x0:x1]):
            raise OSError(f"could not write {destination}")
        manifest.append({"crop": str(destination.relative_to(output)),
                         "source": str(path), "split": split, "label": label})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(counts, indent=2))
    if not counts.get("train/occluded"):
        print("No occluded examples: add labeled hand/arm occlusions before live use.")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", default=Path("data/cells"), type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)


if __name__ == "__main__":
    main()
