# Models

## handwriting_8x8.felm

The classifier that the Hugging Face Space uses for hand-drawn digits.

| Property | Value |
|---|---|
| Model | `BatchElm<float>`: 64 inputs, 4,096 ReLU hidden units, 10 outputs, ridge 1.0 |
| Input | 8×8 block counts (0–16) produced by `deploy/huggingface/zerogpu/digitprep.py` |
| Training data | MNIST train (60,000 digits) through `digitprep`, plus the 1,797 UCI 8×8 digits and their ±1 pixel shifts (76,173 rows) |
| Held-out accuracy | 96.9% on the 10,000 MNIST test digits (every digit 94–99%) |
| Size | 1.2 MB; format described in [docs/api.md](../../docs/api.md#model-files) |
| Training time | 3.5 s on an RTX 4080 with `felm-train` |

### Rebuild

```bash
python3 scripts/build_handwriting_data.py      # needs numpy + pillow; downloads MNIST to .cache/
docker compose run --rm dev-gpu bash -c "cmake --build build/docker --target felm_train && \
  build/docker/felm-train --train build/handwriting/train.csv --test build/handwriting/test.csv \
  --inputs 64 --classes 10 --hidden 4096 --activation relu --ridge 1 --input-scale 64 \
  --out data/models/handwriting_8x8.felm"
```

The seed is fixed, so a rebuild on the same backend produces a byte-identical file.

### Why not the UCI digits alone?

The UCI set has 1,797 digits from 43 writers. A classifier trained only on it scored 46.5% on MNIST
handwriting through the same drawing pipeline (digit 6: 27%). Adding MNIST raises that to ~97%.
The full comparison is in [docs/demos.md](../../docs/demos.md#hand-drawn-digits).

### Attribution

The weights are derived from the **MNIST** database of handwritten digits (Yann LeCun, Corinna
Cortes and Christopher J. C. Burges), which is itself built from NIST Special Databases 1 and 3.
MNIST is commonly distributed under the Creative Commons Attribution-Share Alike 3.0 license, so
treat this model file as a derivative of MNIST under those terms. MNIST itself is not stored in this
repository; the build script downloads it. The UCI digits are covered in
[`data/datasets/SOURCE.md`](../datasets/SOURCE.md).
