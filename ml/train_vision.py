"""Train the cell CNN. Read the five learning steps inside the batch loop."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.cells import CLASS_NAMES, IMAGE_SIZE, CellClassifier, CellDataset
from ml.utils import setup


def metrics(model, loader, device, brightness=1.0):
    model.eval()
    confusion = torch.zeros((4, 4), dtype=torch.long)
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device).mul(brightness).clamp(0, 1)
            predictions = model(images).argmax(dim=1).cpu()
            confusion += torch.bincount(labels * 4 + predictions, minlength=16).reshape(4, 4)
    true_positive = confusion.diag().float()
    precision = true_positive / confusion.sum(dim=0).clamp(min=1)
    recall = true_positive / confusion.sum(dim=1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    present = confusion.sum(dim=1) > 0
    return {"accuracy": float(true_positive.sum() / confusion.sum()),
            "macro_f1": float(f1[present].mean()),
            "confusion_rows_true_columns_predicted": confusion.tolist(),
            "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                                 "count": int(confusion[i].sum())}
                          for i, name in enumerate(CLASS_NAMES)}}


def train(data, output, epochs=20, batch_size=64, lr=1e-3, seed=42, device="auto"):
    device = setup(seed, device)
    output = Path(output)
    if (output / "best.pt").exists():
        raise ValueError(f"checkpoint exists; choose a new output directory: {output}")
    train_data = CellDataset(data, "train", augment=True)
    validation = CellDataset(data, "val")
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("epochs, batch_size, and lr must be positive")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation, batch_size=batch_size)
    model = CellClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    class_counts = torch.bincount(torch.tensor([label for _, label in train_data.samples]),
                                  minlength=len(CLASS_NAMES)).float()
    class_weights = (class_counts.sum() / (3 * class_counts.clamp(min=1))).sqrt()
    class_weights[class_counts == 0] = 0
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    output.mkdir(parents=True, exist_ok=True)
    best, history, stale_epochs = -1.0, [], 0
    print(f"device={device}; train={len(train_data)}; val={len(validation)}", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()                  # 1. Clear the previous gradients.
            logits = model(images)                 # 2. Predict with current weights.
            loss = criterion(logits, labels)        # 3. Compare scores to true labels.
            loss.backward()                        # 4. Differentiate loss w.r.t. weights.
            optimizer.step()                       # 5. Move weights to reduce the loss.
            total += loss.item() * len(labels)
        validation_metrics = metrics(model, val_loader, device)
        dim_metrics = metrics(model, val_loader, device, brightness=0.55)
        bright_metrics = metrics(model, val_loader, device, brightness=1.4)
        row = {"epoch": epoch, "train_loss": total / len(train_data), **validation_metrics,
               "dim_macro_f1": dim_metrics["macro_f1"],
               "bright_macro_f1": bright_metrics["macro_f1"]}
        history.append(row)
        print(f"epoch={epoch} loss={row['train_loss']:.4f} "
              f"val_f1={row['macro_f1']:.3f} dim_f1={row['dim_macro_f1']:.3f} "
              f"bright_f1={row['bright_macro_f1']:.3f}", flush=True)
        selection_score = (row["macro_f1"] + row["dim_macro_f1"] +
                           row["bright_macro_f1"]) / 3
        if selection_score > best:
            best, stale_epochs = selection_score, 0
            torch.save({"format_version": 1, "kind": "cells", "state_dict": model.state_dict(),
                        "classes": list(CLASS_NAMES), "image_size": IMAGE_SIZE,
                        "epoch": epoch, "validation": validation_metrics,
                        "selection_score": selection_score, "dim_validation": dim_metrics,
                        "bright_validation": bright_metrics,
                        "seed": seed, "training_samples": len(train_data)}, output / "best.pt")
        else:
            stale_epochs += 1
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if stale_epochs >= 6:
            print("stopping after six epochs without validation improvement", flush=True)
            break
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/cells")
    parser.add_argument("--output", default="artifacts/vision")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()
    train(args.data, args.output, args.epochs, args.batch_size, args.lr, args.seed, args.device)


if __name__ == "__main__":
    main()
