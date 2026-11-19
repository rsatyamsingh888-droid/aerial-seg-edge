from pathlib import Path

import numpy as np
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from aerialseg.dataset import LoveDADataset

FP32_PATH = Path("runs/unet/model.onnx")
PREP_PATH = Path("runs/unet/model_prep.onnx")
INT8_PATH = Path("runs/unet/model_int8.onnx")
N_CALIB = 100


class CalibrationReader(CalibrationDataReader):
    def __init__(self, input_name):
        ds = LoveDADataset("data/processed/train")
        step = len(ds) // N_CALIB
        self.item = [ds[i][0].unsqueeze(0).numpy() for i in range(0, len(ds), step)][:N_CALIB]
        self.input_name = input_name
        self.position = 0

    def get_next(self):
        if self.position >= len(self.item):
            return None
        batch = {self.input_name: self.item[self.position]}
        self.position += 1
        return batch


def main():
    quant_pre_process(str(FP32_PATH), str(PREP_PATH))

    quantize_static(
        str(PREP_PATH),
        str(INT8_PATH),
        CalibrationReader("image"),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )

    for path in [FP32_PATH, INT8_PATH]:
        print(f"{path.name:18s} {path.stat().st_size / 1e6:.2} MB")


if __name__ == "__main__":
    main()