from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from aerialseg.model import UNet

WEIGHTS =Path("runs/unet/best.pt")
ONNX_PATH = Path("runs/unet/model.onnx")


def main():
    model = UNet()
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()

    example = torch.randn(1, 3, 512, 512)
    torch.onnx.export(
        model,
        (example,),
        str(ONNX_PATH),
        input_names=["image"],
        output_names=["logits"],
        dynamo=True,
        external_data=False,
    )
    print(f"exported: {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1e6:.2f} MB)")

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    with torch.no_grad():
        expected = model(example).numpy()
    actual = session.run(None, {input_name: example.numpy()})[0]

    diff = np.abs(expected - actual).max()
    same = (expected.argmax(axis=1) == actual.argmax(axis=1)).mean()
    print(f"largest score difference: {diff:.6f}")
    print(f"pixels with the same predicted class: {100 * same:.3f} %")


if __name__ == "__main__":
    main()