from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IGNORE_INDEX = 255

class LoveDADataset(Dataset):
    def __init__(self, root, augment=False):
        self.images = sorted((Path(root) / "images").glob("*.png"))
        self.mask_dir = Path(root) / "masks"
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        image = np.array(Image.open(img_path))
        mask = np.array(Image.open(self.mask_dir / img_path.name))


        if self.augment:
            if torch.rand(1).item() < 0.5:
                image = image[:, ::-1]
                mask = mask[:, ::-1]
            if torch.rand(1).item() < 0.5:
                image = image[::-1]
                mask = mask[::-1]

        image = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0


        mask = mask.astype(np.int64) - 1
        mask[mask == -1] = IGNORE_INDEX
        return image, torch.from_numpy(mask)

if __name__ == "__main__":
        ds = LoveDADataset("data/processed/train", augment=True)
        image, mask = ds[0]
        print("items:", len(ds))
        print("image:", image.shape, image.dtype, image.min().item(), image.max().item())
        print("mask:", mask.shape, mask.dtype, torch.unique(mask))

