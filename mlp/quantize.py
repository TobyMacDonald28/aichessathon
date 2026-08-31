"""Quantizes a trained weights.pt to int16 (feature transformer) / int8 (head) for on-disk size.

Be clear about what this does and doesn't buy in this codebase: Stockfish's NNUE gets a real
*speed* win from quantization because its C++ inference does hand-written int8/int16 SIMD
arithmetic. We don't have that — nnue.py dequantizes straight back to float32 at load time (see
`load_weights`), so the accumulator and head both still run in plain numpy float32. What
quantizing here actually buys is file size: the feature transformer alone is 40,961 x 256 float32
weights, ~42MB — uncomfortably close to the whole zip's 50MB cap on its own. int16 halves that,
int8 quarters it, leaving real headroom for the rest of the zip.

Usage: python quantize.py [--in weights.pt] [--out weights.int8.pt]
"""

import argparse
from pathlib import Path

import torch
from train import WEIGHTS_PATH

DEFAULT_OUTPUT = Path(__file__).parent / "weights.int8.pt"

FEATURE_TRANSFORMER_BITS = 16  # summed over up to 32 rows; int8 rounding error compounds too much
HEAD_BITS = 8  # each head layer is one matmul, not a sum of many rows — int8 rounds cleanly

QUANTIZED_KEYS = {
    "feature_transformer.weight": FEATURE_TRANSFORMER_BITS,
    "head.1.weight": HEAD_BITS,
    "head.3.weight": HEAD_BITS,
    "head.5.weight": HEAD_BITS,
}


def _quantize(weight: torch.Tensor, bits: int) -> tuple[torch.Tensor, float]:
    qmax = 2 ** (bits - 1) - 1
    max_abs = weight.abs().max().item()
    scale = max_abs / qmax if max_abs > 0 else 1.0
    dtype = torch.int16 if bits == 16 else torch.int8
    quantized = torch.clamp(torch.round(weight / scale), -qmax - 1, qmax).to(dtype)
    return quantized, scale


def quantize(input_path: Path, output_path: Path) -> None:
    state = torch.load(input_path, map_location="cpu")
    bundle: dict[str, tuple[torch.Tensor, float]] = {}
    for key, tensor in state.items():
        if key in QUANTIZED_KEYS:
            bundle[key] = _quantize(tensor, QUANTIZED_KEYS[key])
        else:
            # biases are small and cheap; quantizing them buys nothing worth the added error
            bundle[key] = (tensor, 1.0)
    torch.save(bundle, output_path)

    original_bytes = sum(t.numel() * t.element_size() for t in state.values())
    quantized_bytes = sum(t.numel() * t.element_size() for t, _ in bundle.values())
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
