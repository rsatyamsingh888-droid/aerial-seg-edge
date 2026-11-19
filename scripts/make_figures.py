from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from aerialseg.dataset import IGNORE_INDEX, LoveDADataset

CLASSES = ["background", "building", "road", "water",
           "barren", "forest", "agriculture"]

COLOURS = ["#d9d9d9", "#e41a1c", "#ffd92f", "#377eb8",
           "#a65628", "#1b7837", "#a6d854", "#000000"]

INDICES = [100, 500, 1100, 1500]
MODELS = {"fp32": "runs/unet/model.onnx", "int8": "runs/unet/model_int8.onnx"}

def main():
    ds = LoveDADataset("data/processed/test")
    sessions = {name: ort.InferenceSession(path, providers=["CPUExecutionProvider"]) for name, path in MODELS.items()}
    cmap = ListedColormap(COLOURS)

    fig, axes = plt.subplots(len(INDICES), 4, figsize=(12, 3 * len(INDICES)))
    titles = ["images", "label", "prediction fp32", "prediction int8"]

    for row, index in enumerate(INDICES):
        image, mask = ds[index]
        x = image.unsqueeze(0).numpy()

        label = mask.numpy().copy()
        label[label == IGNORE_INDEX] = len(CLASSES)

        panels = [image.permute(1, 2, 0).numpy(), label]
        for session in sessions.values():
            name = session.get_inputs()[0].name
            logits = session.run(None, {name: x})[0]
            panels.append(logits.argmax(axis=1)[0])

        for col, panel in enumerate(panels):
            ax = axes[row, col]
            if col == 0:
                ax.imshow(panel)
            else:
                ax.imshow(panel, cmap=cmap, vmin=0, vmax=len(CLASSES), interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(titles[col])
                    axes[row, 0].set_ylabel(ds.images[index].stem, fontsize=10)

    legend = [Patch(color=c, label=n) for c, n in zip(COLOURS, CLASSES + ["no-data"])]
    fig.legend(handles=legend, loc="lower center", ncol=8, fontsize=9)
    fig.tight_layout(rect=(0, 0.3, 1, 1))


    out = Path("docs/predictions.png")
    fig.savefig(out, dpi=100)
    print("wrote", out)


if __name__ == "__main__":
    main()