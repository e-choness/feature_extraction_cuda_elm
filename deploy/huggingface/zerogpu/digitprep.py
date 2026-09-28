"""Drawing -> 8x8 feature pipeline shared by the Space (inference) and the model build (training).

Mirrors how the UCI optical-digits set was made: the digit is fitted into a 32x32 binary box
(aspect ratio preserved), then "on" pixels are counted in 4x4 blocks, giving 64 values in 0..16.
Training data (MNIST handwriting and UCI digits) and live drawings go through this same code, so
the classifier never sees a preprocessing mismatch.
"""

import numpy as np
from PIL import Image

INK_THRESHOLD = 0.08  # fraction of full ink that counts as part of the digit when cropping
BINARY_THRESHOLD = 0.35  # after resizing into the 32x32 box


def ink_from_rgba(rgba: np.ndarray) -> np.ndarray:
    """Dark strokes on a light or transparent background -> ink intensity in 0..1."""
    rgba = np.asarray(rgba, dtype=np.float32)
    if rgba.ndim == 2:
        return 1.0 - rgba / 255.0
    if rgba.shape[2] == 3:
        return (255.0 - rgba.mean(axis=2)) / 255.0
    gray = rgba[..., :3].mean(axis=2)
    return (255.0 - gray) / 255.0 * (rgba[..., 3] / 255.0)


def features(ink: np.ndarray) -> np.ndarray | None:
    """Ink image (any resolution, 0..1) -> 64 block counts in 0..16, or None if blank."""
    rows = np.where(ink.max(axis=1) > INK_THRESHOLD)[0]
    cols = np.where(ink.max(axis=0) > INK_THRESHOLD)[0]
    if len(rows) == 0:
        return None
    crop = ink[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    h, w = crop.shape
    scale = 32.0 / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8)).resize(
        (nw, nh), Image.Resampling.BILINEAR
    )
    box = np.zeros((32, 32), dtype=bool)
    top, left = (32 - nh) // 2, (32 - nw) // 2
    box[top : top + nh, left : left + nw] = np.asarray(resized, dtype=np.float32) / 255.0 > BINARY_THRESHOLD
    return box.reshape(8, 4, 8, 4).sum(axis=(1, 3)).astype(np.float32).ravel()


def preview(feats: np.ndarray, size: int = 160) -> Image.Image:
    """8x8 features -> an upscaled grayscale image of what the model sees."""
    pixels = (np.asarray(feats).reshape(8, 8) / 16.0 * 255).astype(np.uint8)
    return Image.fromarray(pixels).resize((size, size), Image.Resampling.NEAREST)
