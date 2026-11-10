"""Vision model, image preprocessing, and a labeled-cell dataset.

Class IDs 0/1/2 match board.py. Class 3 means a cell is occluded, so a
confident prediction of that class still cannot become a game observation.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

CLASS_NAMES = ("empty", "red", "yellow", "occluded")
IMAGE_SIZE = 64


class CellClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.classifier = nn.Linear(64, len(CLASS_NAMES))

    def forward(self, images):
        # [batch, 3, 64, 64] -> [batch, 4] raw class scores (logits).
        return self.classifier(self.features(images))


def image_tensor(bgr):
    """OpenCV BGR uint8 -> RGB float32 channels-first tensor in [0, 1]."""
    rgb = cv2.cvtColor(cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().div(255)


class CellDataset(Dataset):
    def __init__(self, root, split, augment=False):
        self.samples = []
        self.augment = augment
        for label, name in enumerate(CLASS_NAMES):
            for path in sorted((Path(root) / split / name).glob("*")):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((path, label))
        if not self.samples:
            raise ValueError(f"no labeled cells in {Path(root) / split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"cannot read image: {path}")
        x = image_tensor(image)
        if self.augment:
            # Brightness changes preserve red/yellow labels. Hue swaps do not.
            x = (x * (0.8 + 0.4 * torch.rand(()))).clamp(0, 1)
        return x, label


def cell_crops(frame, calibration):
    """Rectify the play area, then return crops in row-0-at-bottom order."""
    width = round(7 * calibration.cell_w)
    height = round(6 * calibration.cell_h)
    rectified = cv2.warpPerspective(frame, calibration.H, (width, height))
    crops = []
    for row in range(6):
        top = 5 - row
        for col in range(7):
            x0, x1 = round(col * width / 7), round((col + 1) * width / 7)
            y0, y1 = round(top * height / 6), round((top + 1) * height / 6)
            crops.append(rectified[y0:y1, x0:x1])
    return crops


def observation(probabilities, calibration, threshold):
    """[42, 4] probabilities -> existing full-frame detections, or None.

    None means unknown, while [] means confidently empty. Keeping those
    distinct prevents an occluded board from appearing to have been cleared.
    """
    p = np.asarray(probabilities)
    if p.shape != (42, len(CLASS_NAMES)) or not np.isfinite(p).all():
        return None
    labels, confidence = p.argmax(axis=1), p.max(axis=1)
    if (labels == 3).any() or (confidence < threshold).any():
        return None
    detections = []
    for index, (label, conf) in enumerate(zip(labels, confidence)):
        if label == 0:
            continue
        row, col = divmod(index, 7)
        x, y = calibration.cell_center_frame(row, col)
        # Boxes describe calibrated cells; they are not object-localization outputs.
        left = calibration.cell_center_frame(row, col - 0.4)
        right = calibration.cell_center_frame(row, col + 0.4)
        top = calibration.cell_center_frame(row + 0.4, col)
        bottom = calibration.cell_center_frame(row - 0.4, col)
        w, h = abs(right[0] - left[0]), abs(bottom[1] - top[1])
        detections.append((x, y, w, h, CLASS_NAMES[label], float(conf)))
    return detections


class CellReader:
    def __init__(self, checkpoint, calibration, device="cpu", threshold=0.9):
        from ml.utils import load_checkpoint

        if not 0 < threshold <= 1:
            raise ValueError("vision_min_confidence must be in (0, 1]")
        saved = load_checkpoint(checkpoint, "cells")
        if saved["classes"] != list(CLASS_NAMES) or saved["image_size"] != IMAGE_SIZE:
            raise ValueError("checkpoint classes or preprocessing do not match this reader")
        self.device = torch.device(device)
        self.model = CellClassifier().to(self.device)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        self.calibration, self.threshold = calibration, threshold

    def predict(self, frame):
        images = torch.stack([image_tensor(c) for c in cell_crops(frame, self.calibration)])
        with torch.inference_mode():
            probabilities = self.model(images.to(self.device)).softmax(dim=1).cpu().numpy()
        return observation(probabilities, self.calibration, self.threshold)
