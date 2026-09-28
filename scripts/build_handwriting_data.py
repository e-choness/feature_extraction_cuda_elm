#!/usr/bin/env python3
"""Build the training/test CSVs for the Space's hand-drawn digit classifier.

    python3 scripts/build_handwriting_data.py            # writes build/handwriting/{train,test}.csv
    docker compose run --rm dev-gpu build/<dir>/felm-train \
        --train build/handwriting/train.csv --test build/handwriting/test.csv \
        --inputs 64 --classes 10 --hidden 4096 --activation relu --ridge 1 --input-scale 64 \
        --out data/models/handwriting_8x8.felm

Training data: MNIST train (60k handwritten digits, drawn into 8x8 features with the same code the
Space uses on sketches) plus every UCI 8x8 digit and its +-1 pixel shifts. Test data: MNIST test.
A classifier trained on UCI alone scores ~47% on MNIST handwriting through this pipeline; with MNIST
added it scores ~96% (see docs/demos.md#hand-drawn-digits).

Needs numpy and pillow. MNIST is downloaded once into .cache/mnist.
"""

import gzip
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy" / "huggingface" / "zerogpu"))
import digitprep  # noqa: E402  (shared with the Space's app.py)

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}
SHIFTS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]


def load_idx(path: Path) -> np.ndarray:
    data = gzip.decompress(path.read_bytes())
    if int.from_bytes(data[:4], "big") == 2051:
        n, rows, cols = (int.from_bytes(data[i : i + 4], "big") for i in (4, 8, 12))
        return np.frombuffer(data, np.uint8, offset=16).reshape(n, rows, cols)
    return np.frombuffer(data, np.uint8, offset=8)


def mnist() -> dict[str, np.ndarray]:
    cache = ROOT / ".cache" / "mnist"
    cache.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for key, name in MNIST_FILES.items():
        path = cache / name
        if not path.exists():
            print(f"downloading {name}", flush=True)
            urllib.request.urlretrieve(MNIST_URL + name, path)
        arrays[key] = load_idx(path)
    return arrays


def mnist_features(images: np.ndarray) -> np.ndarray:
    # MNIST is white ink on black at 28x28; upscale like a sketchpad canvas before featurising.
    out = np.empty((len(images), 64), np.float32)
    for i, img in enumerate(images):
        ink = np.asarray(Image.fromarray(img).resize((224, 224), Image.Resampling.BILINEAR), np.float32) / 255.0
        out[i] = digitprep.features(ink)
    return out


def uci_with_shifts() -> tuple[np.ndarray, np.ndarray]:
    uci = np.loadtxt(ROOT / "data" / "datasets" / "digits_8x8.csv", delimiter=",", skiprows=1)
    labels, pixels = uci[:, 0].astype(int), uci[:, 1:].reshape(-1, 8, 8)
    xs, ys = [], []
    for dx, dy in SHIFTS:
        # UCI digits have blank borders, so a +-1 roll never wraps ink around.
        xs.append(np.roll(np.roll(pixels, dy, axis=1), dx, axis=2).reshape(-1, 64))
        ys.append(labels)
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys)


def write_csv(path: Path, labels: np.ndarray, feats: np.ndarray) -> None:
    header = "label," + ",".join(f"f{i}" for i in range(64))
    rows = np.column_stack([labels, feats]).astype(np.int32)  # block counts are integers 0..16
    np.savetxt(path, rows, fmt="%d", delimiter=",", header=header, comments="")
    print(f"wrote {path} ({len(labels)} rows)", flush=True)


def main() -> None:
    out = ROOT / "build" / "handwriting"
    out.mkdir(parents=True, exist_ok=True)
    m = mnist()
    uci_x, uci_y = uci_with_shifts()
    train_x = np.concatenate([mnist_features(m["train_x"]), uci_x])
    train_y = np.concatenate([m["train_y"].astype(int), uci_y])
    write_csv(out / "train.csv", train_y, train_x)
    write_csv(out / "test.csv", m["test_y"].astype(int), mnist_features(m["test_x"]))


if __name__ == "__main__":
    main()
