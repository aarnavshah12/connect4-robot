"""Evaluate the selected checkpoint on a held-out cell split."""

import argparse
import json
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from ml.cells import CellDataset, CellReader
from ml.train_vision import metrics
from ml.utils import setup


def stress_report(reader, dataset, device, threshold=0.9):
    """Controlled lighting probes plus complete-board counts when available."""
    scenarios = {"clean": 1.0, "dim_55pct": 0.55, "bright_140pct": 1.4}
    images = torch.stack([dataset[i][0] for i in range(len(dataset))])
    labels = torch.tensor([label for _, label in dataset.samples])
    report = {}
    for name, brightness in scenarios.items():
        with torch.inference_mode():
            predictions = []
            for batch in images.split(64):
                predictions.append(reader.model(batch.to(device).mul(brightness).clamp(0, 1))
                                   .softmax(dim=1).cpu())
        probabilities = torch.cat(predictions)
        chosen = probabilities.argmax(dim=1)
        confident = probabilities.max(dim=1).values >= threshold
        groups = defaultdict(list)
        for path, ok, enough in zip((p for p, _ in dataset.samples),
                                    (chosen == labels).tolist(), confident.tolist()):
            if path.name.startswith("board_"):
                groups[path.name.split("_")[1]].append((ok, enough))
        full_boards = [values for values in groups.values() if len(values) == 42]
        report[name] = {
            "correct_cells": int((chosen == labels).sum()), "total_cells": len(labels),
            "exact_boards": sum(all(ok for ok, _ in board) for board in full_boards),
            "accepted_boards": sum(all(enough for _, enough in board) for board in full_boards),
            "accepted_correct_boards": sum(all(ok and enough for ok, enough in board)
                                           for board in full_boards),
            "accepted_incorrect_boards": sum(all(enough for _, enough in board)
                                             and not all(ok for ok, _ in board)
                                             for board in full_boards),
            "total_boards": len(full_boards),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/cells")
    parser.add_argument("--checkpoint", default="artifacts/vision/best.pt")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--stress", action="store_true", help="also test dim and bright cells and exact boards")
    args = parser.parse_args()
    device = setup(device=args.device)
    reader = CellReader(args.checkpoint, calibration=None, device=device)
    dataset = CellDataset(args.data, args.split)
    result = metrics(reader.model, DataLoader(dataset, batch_size=64), device)
    if args.stress:
        result["stress_and_boards"] = stress_report(reader, dataset, device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
