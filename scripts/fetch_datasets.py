#!/usr/bin/env python3
"""Download MNIST and Fashion-MNIST into .cache/datasets/<name>/ as uncompressed IDX files.

    python3 scripts/fetch_datasets.py            # both datasets
    python3 scripts/fetch_datasets.py mnist      # just one

The files feed bench/bench_datasets.cpp (see docs/benchmarks.md) and feature_elm::loadIdx. Neither
dataset is stored in the repository:
  * MNIST (Yann LeCun, Corinna Cortes, Christopher J. C. Burges), commonly distributed under
    CC BY-SA 3.0.
  * Fashion-MNIST (Zalando Research), MIT license.

Standard library only.
"""

import gzip
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "mnist": "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "fashion-mnist": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
}
FILES = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
]


def fetch(name: str) -> None:
    target = ROOT / ".cache" / "datasets" / name
    target.mkdir(parents=True, exist_ok=True)
    for stem in FILES:
        out = target / stem
        if out.exists():
            continue
        url = DATASETS[name] + stem + ".gz"
        print(f"{name}: downloading {stem}.gz", flush=True)
        with urllib.request.urlopen(url) as response, gzip.GzipFile(fileobj=response) as src:
            with open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
    print(f"{name}: ready in {target.relative_to(ROOT)}", flush=True)


def main() -> None:
    names = sys.argv[1:] or list(DATASETS)
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        sys.exit(f"unknown dataset(s): {', '.join(unknown)}; choose from {', '.join(DATASETS)}")
    for name in names:
        fetch(name)


if __name__ == "__main__":
    main()
