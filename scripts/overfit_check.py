import torch
from torch import nn
from torch.utils.data import DataLoader

from aerialseg.dataset import IGNORE_INDEX, LoveDADataset
from aerialseg.model import UNet



def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    device = get_device()
    print("device:", device)

    ds = LoveDADataset("data/processed/train")
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    images, masks = next(iter(loader))
    images, masks = images.to(device), masks.to(device)


    model = UNet().to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(1, 101):
        logits = model(images)
        loss = loss_fn(logits, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"step {step:3d}  loss {loss.item():.3f}")


if __name__ == "__main__":
    main()