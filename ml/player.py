"""A policy/value network and an adapter for game.py's move interface."""

import numpy as np
import torch
from torch import nn

from ml.positions import encode, legal_mask
from ml.utils import choose_device, load_checkpoint


class PolicyValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 6 * 7, 64), nn.ReLU(),
        )
        self.policy = nn.Linear(64, 7)
        self.value = nn.Linear(64, 1)

    def forward(self, positions):
        features = self.features(positions)
        # Policy logits are move preferences. Value is an expected game outcome.
        return self.policy(features), self.value(features).tanh().squeeze(-1)


class TorchPlayer:
    score_kind = "expected_outcome"

    def __init__(self, checkpoint, device="cpu"):
        self.device = choose_device(device)
        saved = load_checkpoint(checkpoint, "player")
        if saved.get("encoding") != "current-opponent-bottom-first":
            raise ValueError("unsupported player checkpoint encoding")
        self.model = PolicyValueNet().to(self.device)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()

    def __call__(self, board, player, considered=None):
        allowed = legal_mask(board)
        if not allowed.any():
            raise ValueError("no legal moves: board is full")
        x = torch.from_numpy(encode(board, player)).unsqueeze(0).to(self.device)
        mask = torch.as_tensor(allowed, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            logits, value = self.model(x)
            probabilities = logits.masked_fill(~mask, -torch.inf).softmax(dim=1)[0].cpu().numpy()
        if not np.isfinite(probabilities).all() or not torch.isfinite(value).all():
            raise ValueError("player produced non-finite predictions")
        if considered is not None:
            considered.extend((int(c), float(probabilities[c])) for c in np.flatnonzero(allowed))
        return int(probabilities.argmax()), float(value.item())
