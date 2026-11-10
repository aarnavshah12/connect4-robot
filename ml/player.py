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
        # Policy logits are move preferences. Value approximates the teacher's search score.
        return self.policy(features), self.value(features).tanh().squeeze(-1)


class TorchPlayer:
    score_kind = "learned_value"

    def __init__(self, checkpoint, device="cpu", tactical_guard=False):
        self.device = choose_device(device)
        saved = load_checkpoint(checkpoint, "player")
        if saved.get("encoding") != "current-opponent-bottom-first":
            raise ValueError("unsupported player checkpoint encoding")
        self.model = PolicyValueNet().to(self.device)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        self.tactical_guard = tactical_guard

    @staticmethod
    def _winning_columns(board, player):
        from board import check_win, landing_row

        winning = []
        for col in range(7):
            row = landing_row(board, col)
            if row is None:
                continue
            candidate = [line[:] for line in board]
            candidate[row][col] = player
            if check_win(candidate, player):
                winning.append(col)
        return winning

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
        column = int(probabilities.argmax())
        if self.tactical_guard:
            wins = self._winning_columns(board, player)
            if wins:
                column = max(wins, key=lambda c: probabilities[c])
            else:
                threats = self._winning_columns(board, 3 - player)
                if len(threats) == 1:
                    column = threats[0]
        return int(column), float(value.item())
