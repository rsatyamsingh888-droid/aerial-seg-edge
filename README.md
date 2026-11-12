# aerial-seg-edge

Train a small semantic segmentation model on aerial images in PyTorch,
export it to ONNX, and measure what the export changes:
accuracy, model size and CPU inference time.

**Status: work in progress.** The results section is filled in only with
numbers measured by the scripts in this repository.

## Progress

- [x] Data download and statistics
- [ ] Preprocessing and PyTorch dataset
- [ ] U-Net model and training loop
- [ ] Evaluation (mean IoU per class)
- [ ] ONNX export and parity check
- [ ] Benchmark: model size and CPU latency
- [ ] Results table and prediction images

## Data

[LoveDA](https://zenodo.org/records/5706578): aerial images of urban and
rural scenes, 1024 x 1024 px, with 7 land-cover classes.
Measured statistics: [docs/data_stats.md](docs/data_stats.md).
Licence: see [DATA_LICENSE.md](DATA_LICENSE.md).

The official Test split has no public labels. The official Val split is
therefore used as the held-out test set, and a validation set is split
off from Train.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py    # about 6.4 GB
python scripts/explore_data.py
```

## Results

Not measured yet.

## Licence

Code: MIT. Data: see [DATA_LICENSE.md](DATA_LICENSE.md).