"""Quantizes a trained weights.pt into the weights.npz that ships with agent.py.

Two different reasons to quantize, for two different tensors:

- `feature_transformer.weight` is 40,961 x 256 float32, ~42MB — uncomfortably close to the whole
  zip's 50MB cap on its own. int16 halves that, leaving real headroom for the rest of the zip.
  This one is about file size.
- `head.1.weight` (512 x 32) is a few KB either way, so quantizing it buys nothing on disk. It's
  quantized so agent.py can run it as a genuine int8 SIMD dot product at inference time instead of
  a float32 matmul — see agent.py's `_int8_matvec` and `evaluate_head`. It's also the only head
  layer worth bothering with: at 512x32 multiply-adds it does ~94% of the head's arithmetic,
  versus 32x32 and 32x1 for the two layers after it, so those stay plain float32.

Both int16 and int8 targets use the same per-tensor symmetric scheme: scale = max(|w|) / qmax,
round(w / scale) clamped to the integer range. Feature transformer weights get summed over up to
32 active rows before anything reads them, so int8 rounding error would compound too far — hence
int16 there specifically.

Output is a plain numpy .npz, not a torch file: agent.py never imports torch, so weights.npz is
the only format it can load. torch stays here on the training side only, to read the checkpoint
train.py wrote.

Usage: python quantize.py [--in weights.pt] [--out weights.npz]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from train import WEIGHTS_PATH

DEFAULT_OUTPUT = Path(__file__).parent / "weights.npz"

FEATURE_TRANSFORMER_BITS = 16  # summed over up to 32 rows; int8 rounding error compounds too much
HEAD_BITS = 8  # one matmul, not a sum of many rows — int8 rounds cleanly, and enables SIMD

QUANTIZED_KEYS = {
    "feature_transformer.weight": FEATURE_TRANSFORMER_BITS,
    "head.1.weight": HEAD_BITS,
}


def quantize_tensor(weight: np.ndarray, bits: int) -> tuple[np.ndarray, float]:
    qmax = 2 ** (bits - 1) - 1
    max_abs = float(np.abs(weight).max())
    scale = max_abs / qmax if max_abs > 0 else 1.0
    dtype = np.int16 if bits == 16 else np.int8
    quantized = np.clip(np.round(weight / scale), -qmax - 1, qmax).astype(dtype)
    return quantized, scale


def quantize(input_path: Path, output_path: Path) -> None:
    state = torch.load(input_path, map_location="cpu")
    arrays: dict[str, np.ndarray] = {}
    original_bytes = 0
    for key, tensor in state.items():
        weight = tensor.numpy()
        original_bytes += weight.nbytes
        if key in QUANTIZED_KEYS:
            quantized, scale = quantize_tensor(weight, QUANTIZED_KEYS[key])
            arrays[key] = quantized
            arrays[f"{key}.scale"] = np.float32(scale)
        else:
            # biases and the two small head layers are cheap either way; quantizing them buys
            # nothing worth the added error.
            arrays[key] = weight.astype(np.float32)

    np.savez(output_path, **arrays)

    quantized_bytes = sum(array.nbytes for array in arrays.values())
    print(
        f"{input_path} ({original_bytes / 1e6:.1f} MB) -> "
        f"{output_path} ({quantized_bytes / 1e6:.1f} MB)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", type=Path, default=WEIGHTS_PATH)
    parser.add_argument("--out", dest="output_path", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    quantize(arguments.input_path, arguments.output_path)


if __name__ == "__main__":
    main()
