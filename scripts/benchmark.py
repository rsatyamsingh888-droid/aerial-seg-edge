import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from aerialseg.model import UNet

WEIGHTS =Path("runs/unet/best.pt")
ONNX_PATH = Path("runs/unet/model.onnx")
WARMUP = 10
RUNS = 50


def measure(fn):
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return round(float(np.median(times)), 1), round(float(np.percentile(times, 90)), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    x = torch.randn(1, 3, 512, 512)

    model = UNet()
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()

    def run_pytorch():
        with torch.no_grad():
            model(x)

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(ONNX_PATH), options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    x_np = x.numpy()

    def run_onnx():
        session.run(None, {input_name: x_np})


    results = {
        "machine": platform.platform(),
        "threads": args.threads,
        "input": "1 x 3 x 512 x 512",
        "runs": RUNS,
        
    }
    for name, fn, path in [("pytorch", run_pytorch, WEIGHTS), ("onnx", run_onnx, ONNX_PATH)]:
        median, p90 = measure(fn)
        size = round(path.stat().st_size / 1e6, 2)
        results[name] = {"size_mb": size, "median_ms": median, "p90_ms": p90}
        print(f"{name:8s} {size:6.2f} MB median {median:6.1f} ms p90 {p90:6.1f} ms")



    out = Path("docs") / f"benchmark_{args.threads}threads.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()