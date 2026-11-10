"""The few setup steps shared by the two training scripts."""

import random
from pathlib import Path

import numpy as np
import torch


def choose_device(device="auto"):
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available() else
                  "mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(device)


def setup(seed=42, device="auto"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # These networks are small; many CPU threads add overhead.
    torch.set_num_threads(2)
    return choose_device(device)


def load_checkpoint(path, kind):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("kind") != kind or checkpoint.get("format_version") != 1:
        raise ValueError(f"expected a version-1 {kind} checkpoint: {path}")
    return checkpoint
