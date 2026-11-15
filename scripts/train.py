import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from aerialseg.dataset import IGNORE_INDEX, LoveDADataset
from aerialseg.model import UNet

DATA = Path("data/processed")
RUN_DIR = Path("runs/UNet")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            loss = loss_fn(model(images), masks)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
    return total / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = get_device()
    print("device:", device)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = LoveDADataset(DATA / "train", augment=True)
    val_ds = LoveDADataset(DATA / "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = UNet().to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = float("inf")
    with open (RUN_DIR / "log.csv", "w", newline= "") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss",  "val_loss", "seconds"])
        for epoch in range(1, args.epochs +1):
            start = time.time()
            train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer)
            val_loss = run_epoch(model, val_loader, loss_fn, device)
            seconds = time.time() - start

            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{seconds:.0f}"])
            f.flush()

            note = ""
            if val_loss <best:
                best = val_loss
                torch.save(model.state_dict(), RUN_DIR / "best.pt")
                note = "  (saved)"
            print(f"epoch {epoch:2d} train {train_loss:.3f} val {val_loss:.3f} {seconds:.0f}s{note}")


if __name__ == "__main__":
    main()


