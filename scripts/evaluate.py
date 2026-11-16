import argparse
import json
from pathlib import Path


import torch
from torch.utils.data import DataLoader

from aerialseg.dataset import IGNORE_INDEX, LoveDADataset, LoveDADataset
from aerialseg.metrics import confusion_matrix, iou_per_class
from aerialseg.model import UNet

CLASSES =["background", "building", "road", "water", "barren", "forest", "agriculture"]

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--weights", default="runs/unet/best.pt")
    args = parser.parse_args()

    device = get_device()
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    ds = LoveDADataset(Path("data/processed") / args.split)
    loader = DataLoader(ds, batch_size=8)

    matrix = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long)
    with torch.no_grad():
        for images, masks in loader:
            pred = model(images.to(device)).argmax(dim=1).cpu()
            matrix += confusion_matrix(pred, masks, len(CLASSES), IGNORE_INDEX)


    iou = iou_per_class(matrix)
    results = {name: round(v.item(), 4) for name, v in zip(CLASSES, iou)}
    results["mIoU"] = round(iou.mean().item(), 4)

    print(f"split: {args.split}, images: {len(ds)}")
    for name, value in results.items():
        print(f" {name:12s} {100 * value:5.1f} %")


    out = Path("docs") / f"results_pytorch_{args.split}.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
