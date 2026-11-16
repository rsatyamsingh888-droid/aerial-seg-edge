import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = Path("runs/UNet/log.csv")
OUT = Path("docs")



def main():
    with open(LOG) as f:
        rows = list(csv.DictReader(f))
    epochs = [int(r["epoch"]) for r in rows]
    train = [float(r["train_loss"]) for r in rows]
    val = [float(r["val_loss"]) for r in rows]
    best = val.index(min(val))

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train, label="train")
    plt.plot(epochs, val, label="validation")
    plt.scatter([epochs[best]], [val[best]], zorder=3, label=f"best (epoch {epochs[best]})")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.title("U-Net on LoveDA, 512 px")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    OUT.mkdir(exist_ok=True)
    plt.savefig(OUT / "loss_curve.png", dpi=150)
    (OUT / "train_log.csv").write_text(LOG.read_text())
    print("wrote docs/loss_curve.png and docs/train_log.csv")



if __name__=="__main__":
    main()

