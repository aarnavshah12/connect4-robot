"""Evaluate the selected checkpoint on a held-out cell split."""

import argparse
import json

from torch.utils.data import DataLoader

from ml.cells import CellDataset, CellReader
from ml.train_vision import metrics
from ml.utils import setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/cells")
    parser.add_argument("--checkpoint", default="artifacts/vision/best.pt")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()
    device = setup(device=args.device)
    reader = CellReader(args.checkpoint, calibration=None, device=device)
    result = metrics(reader.model, DataLoader(CellDataset(args.data, args.split), batch_size=64), device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
