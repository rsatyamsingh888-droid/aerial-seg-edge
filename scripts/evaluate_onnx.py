import argparse
import json
from pathlib import Path

import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

from aerialseg.dataset import IGNORE_INDEX, LoveDADataset
from aerialseg.metrics import confusion_matrix, iou_per_class


CLASSES = ["background", "building", "road", "water", "barren", "forest", "agriculture"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/unet/model.onnx")
    parser.add_argument("--name", default="onnx")
    args = parser.parse_args()

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    ds = LoveDADataset("data/processed/test")
    loader = DataLoader(ds, batch_size=1)

    matrix = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long)
    for images, masks in loader:
        logits = session.run(None, {input_name: images.numpy()})[0]
        pred = torch.from_numpy(logits).argmax(dim=1)
        matrix += confusion_matrix(pred, masks, len(CLASSES), IGNORE_INDEX)


    iou = iou_per_class(matrix)
    results = {name: round(v.item(), 4)for name, v in zip(CLASSES, iou)}
    results["mIoU"] = round(iou.mean().item(), 4)

    print(f"model: {args.model} images: {len(ds)}")
    for name, value in results.items():
        print(f"{name:12s} {100 * value:5.1f} %")


    out = Path("docs") / f"results_{args.name}_test.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()