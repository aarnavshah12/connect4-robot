"""Learn minimax moves and search values with PyTorch."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from engine import _drop, _playable, _to_bitboards, _won
from ml.player import PolicyValueNet
from ml.positions import encode
from ml.utils import setup


class PositionDataset(Dataset):
    def __init__(self, path, split, augment=False, tactical_weight=1.0):
        with np.load(path, allow_pickle=False) as data:
            boards, players = data[f"{split}_boards"], data[f"{split}_players"]
            self.positions = torch.from_numpy(np.stack([encode(b, p) for b, p in zip(boards, players)]))
            self.moves = torch.from_numpy(data[f"{split}_moves"].copy()).long()
            self.values = torch.from_numpy(data[f"{split}_values"].copy()).float()
            self.allowed = torch.from_numpy((boards[:, 5, :] == 0).copy())
        if len(self.moves) != len(boards) or not self.allowed[torch.arange(len(boards)), self.moves].all():
            raise ValueError("dataset contains an illegal teacher move")
        self.augment = augment
        self.weights = torch.ones(len(boards), dtype=torch.float32)
        if tactical_weight != 1:
            for i, (board, player) in enumerate(zip(boards, players)):
                me, opp, mask = _to_bitboards(board, int(player))
                my_wins = [col for col in range(7) if _playable(mask, col)
                           and _won(_drop(me, mask, col)[0])]
                opp_wins = [col for col in range(7) if _playable(mask, col)
                            and _won(_drop(opp, mask, col)[0])]
                if my_wins or len(opp_wins) == 1:
                    self.weights[i] = tactical_weight

    def __len__(self):
        return len(self.moves)

    def __getitem__(self, index):
        position, move = self.positions[index], self.moves[index]
        allowed, value = self.allowed[index], self.values[index]
        if self.augment and torch.rand(()) < 0.5:
            position, move, allowed = position.flip(-1), 6 - move, allowed.flip(-1)
        return position, move, value, allowed, self.weights[index]


def metrics(model, loader, device):
    model.eval()
    correct = count = 0
    squared_error = 0.0
    with torch.inference_mode():
        for positions, moves, values, allowed, _weights in loader:
            logits, predicted_value = model(positions.to(device))
            predictions = logits.masked_fill(~allowed.to(device), -torch.inf).argmax(dim=1)
            correct += int((predictions.cpu() == moves).sum())
            squared_error += float(((predicted_value.cpu() - values) ** 2).sum())
            count += len(moves)
    return {"move_agreement": correct / count, "value_mse": squared_error / count, "positions": count}


def train(data, output, epochs=20, batch_size=128, lr=1e-3, seed=42, device="auto",
          tactical_weight=4.0, value_weight=0.25):
    device, output = setup(seed, device), Path(output)
    if (output / "best.pt").exists():
        raise ValueError(f"checkpoint exists; choose a new output directory: {output}")
    if epochs < 1 or batch_size < 1 or lr <= 0 or tactical_weight < 1 or value_weight < 0:
        raise ValueError("invalid training settings")
    training = PositionDataset(data, "train", augment=True, tactical_weight=tactical_weight)
    validation = PositionDataset(data, "val")
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation, batch_size=batch_size)
    with np.load(data, allow_pickle=False) as dataset:
        metadata = json.loads(str(dataset["metadata"]))
    model = PolicyValueNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    policy_loss, value_loss = nn.CrossEntropyLoss(reduction="none"), nn.MSELoss()
    output.mkdir(parents=True, exist_ok=True)
    best, history = -1.0, []
    print(f"device={device}; train={len(training)}; val={len(validation)}", flush=True)
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for positions, moves, values, allowed, weights in train_loader:
            positions, moves = positions.to(device), moves.to(device)
            values, allowed, weights = values.to(device), allowed.to(device), weights.to(device)
            optimizer.zero_grad()
            logits, predicted_value = model(positions)
            # Full columns must be excluded during both training and inference.
            logits = logits.masked_fill(~allowed, -1e9)
            loss = (policy_loss(logits, moves) * weights).mean()
            loss += value_weight * value_loss(predicted_value, values)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(moves)
        validation_metrics = metrics(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": total / len(training), **validation_metrics}
        history.append(row)
        print(f"epoch={epoch} loss={row['train_loss']:.4f} "
              f"val_move_agreement={row['move_agreement']:.3f} val_value_mse={row['value_mse']:.3f}",
              flush=True)
        selection_score = row["move_agreement"] - 0.1 * row["value_mse"]
        if selection_score > best:
            best = selection_score
            stale_epochs = 0
            torch.save({"format_version": 1, "kind": "player", "state_dict": model.state_dict(),
                        "encoding": "current-opponent-bottom-first", "epoch": epoch,
                        "validation": validation_metrics, "seed": seed, "dataset": metadata,
                        "tactical_weight": tactical_weight, "value_weight": value_weight},
                       output / "best.pt")
        else:
            stale_epochs += 1
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if stale_epochs >= 6:
            print("stopping after six epochs without validation improvement", flush=True)
            break
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/games.npz")
    parser.add_argument("--output", default="artifacts/player")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tactical-weight", type=float, default=4.0)
    parser.add_argument("--value-weight", type=float, default=0.25)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()
    train(args.data, args.output, args.epochs, args.batch_size, args.lr, args.seed, args.device,
          args.tactical_weight, args.value_weight)


if __name__ == "__main__":
    main()
