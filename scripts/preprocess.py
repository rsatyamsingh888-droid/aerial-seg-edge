import random

from pathlib import Path

import numpy as np
from PIL import Image

RAW = Path("data/raw")
OUT = Path("data/processed")
SIZE = 512
VAL_FRACTION = 0.1
SEED = 42


def resize_pair(img_path, mask_path):
    image = Image.open(img_path).convert("RGB")
    image = image.resize((SIZE, SIZE), Image.BILINEAR)
    mask = Image.open(mask_path).resize((SIZE, SIZE), Image.NEAREST)
    return image, mask


def find_pairs(split):
    pairs = []
    for scene in ["Urban", "Rural"]:
        img_dir = RAW / split / scene/ "images_png"
        mask_dir = RAW / split / scene / "masks_png"
        for img in sorted(img_dir.glob("*.png")):
            mask = mask_dir / img.name
            if mask.exists():
                pairs.append((scene, img, mask))
    return pairs


def save_set(name, pairs):
    img_out = OUT / name / "images"
    mask_out = OUT / name / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)
    for scene, img_path, mask_path in pairs:
        image, mask = resize_pair(img_path, mask_path)
        new_name = f"{scene.lower()}_{img_path.name}"
        image.save(img_out / new_name)
        mask.save(mask_out / new_name)
    print(f"{name}: {len(pairs)} pairs")
    
        

def main():
    train_pairs = find_pairs("Train")
    random.Random(SEED).shuffle(train_pairs)
    n_val = int(len(train_pairs) * VAL_FRACTION)

    save_set("val", train_pairs[:n_val])
    save_set("train", train_pairs[n_val:])
    save_set("test", find_pairs("Val"))

if __name__ == "__main__":
    main()
