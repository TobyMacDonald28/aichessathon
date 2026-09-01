"""Runtime NNUE inference: the incrementally-updated accumulator that makes evaluating a
position during search cheap, instead of the from-scratch forward pass train.py uses.

The key idea, and the actual point of NNUE: `Accumulator.push`/`pop` mirror `board.push`/`pop`
move for move. Instead of recomputing "sum the embedding rows for every piece on the board" after
every move (train.py's `NNUE.forward` does exactly that), this only touches the embedding rows
for the few squares a single move actually changes — one or two rows, not up to 30. The one
exception is a king move: since the king's square is the anchor every other feature is expressed
relative to, moving it changes what all ~30 of that side's features *mean*, so that side gets a
full recompute (still only for that one side, not both).

This is also where the quantized half of NNUE happens (see quantize.py for why these two tensors
specifically): the feature transformer weight stays int16 end to end, so `Accumulator`'s push/pop
are plain integer add/sub — no rescaling, no rounding, exact. The head's first layer (512 -> 32,
~94% of the head's multiply-adds) runs as an int8 x int8 -> int32 dot product in `_int8_matvec`, a
numba-jitted loop. Checked against the compiled assembly on this machine, it's a scalar
multiply-accumulate loop, not real SIMD — LLVM's auto-vectorizer doesn't kick in for this widening
int8->int32 reduction pattern. It's still faster than the numpy alternatives (no BLAS dispatch or
temporary-array allocation for a matrix this small), just not for the reason that name suggests.
The last two head layers (32 -> 32 -> 1) are cheap enough that plain float32 is fine and simpler.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import chess
import numba
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from quantize import HEAD_BITS, QUANTIZED_KEYS, quantize_tensor
from train import (
    ACCUMULATOR_DIM,
    PIECE_INDEX_NO_KING,
    WEIGHTS_PATH,
    _orient,
    halfkp_indices,
)

Update = tuple[int, int, bool, int]  # (square, piece_type, color, sign)

# The accumulator sums up to 32 int16 weight rows (a legal position has at most 30 non-king
# pieces). int16 alone could overflow (32 * 32767 > 32767); int32 has all the headroom needed.
_ACC_DTYPE = np.int32


@dataclass(slots=True)
class NNUEWeights:
    ft_weight: np.ndarray  # int16, (FEATURE_DIM + 1, ACCUMULATOR_DIM)
    ft_scale: float
    head1_weight: np.ndarray  # int8, (32, ACCUMULATOR_DIM * 2)
    head1_scale: float
    head1_bias: np.ndarray  # float32, (32,)
    head3_weight: np.ndarray  # float32, (32, 32)
    head3_bias: np.ndarray
    head5_weight: np.ndarray  # float32, (1, 32)
    head5_bias: np.ndarray


# ---------------------------------------------------------------------------
# Loading weights: quantize.py's bundle format is {key: (tensor, scale)}; a plain unquantized
# weights.pt straight out of train.py is {key: tensor}. Either way, the feature transformer and
# head.1 come out as integer arrays here — if the file wasn't already quantized, this quantizes
# them on the spot, so the runtime path is always integer regardless of which file gets loaded.
# ---------------------------------------------------------------------------


StateDict = dict[str, torch.Tensor | tuple[torch.Tensor, float]]


def _load_integer(state: StateDict, key: str, bits: int, dtype: type) -> tuple[np.ndarray, float]:
    value = state[key]
    tensor, scale = value if isinstance(value, tuple) else quantize_tensor(value, bits)
    return tensor.numpy().astype(dtype), scale


def _load_float(state: StateDict, key: str) -> np.ndarray:
    value = state[key]
    tensor, scale = value if isinstance(value, tuple) else (value, 1.0)
    return tensor.numpy().astype(np.float32) * scale


def load_weights(path: Path = WEIGHTS_PATH) -> NNUEWeights:
    state = torch.load(path, map_location="cpu")

    ft_weight, ft_scale = _load_integer(
        state, "feature_transformer.weight", QUANTIZED_KEYS["feature_transformer.weight"], np.int16
    )
    head1_weight, head1_scale = _load_integer(state, "head.1.weight", HEAD_BITS, np.int8)

    return NNUEWeights(
        ft_weight=ft_weight,
        ft_scale=ft_scale,
        head1_weight=head1_weight,
        head1_scale=head1_scale,
        head1_bias=_load_float(state, "head.1.bias"),
        head3_weight=_load_float(state, "head.3.weight"),
        head3_bias=_load_float(state, "head.3.bias"),
        head5_weight=_load_float(state, "head.5.weight"),
        head5_bias=_load_float(state, "head.5.bias"),
    )


# ---------------------------------------------------------------------------
# The int8 matvec kernel: compiled, not vectorized (see the module docstring) — still faster than
# numpy here because it's one allocation-free loop instead of a BLAS call or numpy's generic
# integer matmul path. An explicit signature makes numba compile eagerly, right here at import
# time, instead of lazily on the first call during search — the 60s import budget is exactly
# where that cost belongs. No `cache=True`: the platform's filesystem is read-only and each game
# is a fresh process anyway, so an on-disk cache would only risk a failed write for no benefit.
# ---------------------------------------------------------------------------


@numba.njit("int32[:](int8[:], int8[:,:])")
def _int8_matvec(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    out_features, in_features = weight.shape
    out = np.zeros(out_features, dtype=np.int32)
    for i in range(out_features):
        acc = np.int32(0)
        for j in range(in_features):
            acc += np.int32(x[j]) * np.int32(weight[i, j])
        out[i] = acc
    return out


def evaluate_head(x: np.ndarray, weights: NNUEWeights) -> float:
    """x: the concatenated (stm, nstm) accumulator, int32, shape (ACCUMULATOR_DIM * 2,)."""
    relu_x = np.maximum(x, 0)

    # Requantize the int32 accumulator down to int8 for the SIMD matvec. There's no fixed
    # activation range to calibrate against (train.py doesn't bound this layer's output), so the
    # scale is derived per call from the actual max — cheap, since it's one reduction over 512
    # ints, and exact for whatever range this position's activations happen to occupy.
    act_max = max(int(relu_x.max()), 1)
    act_scale = act_max / 127.0
    x_i8 = np.minimum(np.round(relu_x / act_scale), 127).astype(np.int8)

    h1_i32 = _int8_matvec(x_i8, weights.head1_weight)
    combined_scale = np.float32(act_scale * weights.ft_scale * weights.head1_scale)
    h1 = h1_i32.astype(np.float32) * combined_scale + weights.head1_bias
    h1 = np.maximum(h1, 0.0)

    h2 = h1 @ weights.head3_weight.T + weights.head3_bias
    h2 = np.maximum(h2, 0.0)

    out = h2 @ weights.head5_weight.T + weights.head5_bias
    return float(out[0])


# ---------------------------------------------------------------------------
# What changed: given a move about to be played on `board` (not yet pushed), work out which
# HalfKP (square, piece_type, color) facts appeared or disappeared, and whether either king moved
# (which forces a full accumulator refresh for that side instead of an incremental update).
# ---------------------------------------------------------------------------


def move_updates(board: chess.Board, move: chess.Move) -> tuple[list[Update], bool | None]:
    mover = board.piece_at(move.from_square)
    assert mover is not None
    updates: list[Update] = []

    if board.is_castling(move):
        kingside = chess.square_file(move.to_square) == chess.square_file(chess.G1)
        rank = chess.square_rank(move.from_square)
        rook_from = chess.square(7 if kingside else 0, rank)
        rook_to = chess.square(5 if kingside else 3, rank)
        updates.append((rook_from, chess.ROOK, mover.color, -1))
        updates.append((rook_to, chess.ROOK, mover.color, +1))
        return updates, mover.color

    if board.is_en_passant(move):
        captured_square = move.to_square + (-8 if mover.color == chess.WHITE else 8)
        captured = board.piece_at(captured_square)
        assert captured is not None
        updates.append((captured_square, captured.piece_type, captured.color, -1))
    else:
        captured = board.piece_at(move.to_square)
        if captured is not None:
            updates.append((move.to_square, captured.piece_type, captured.color, -1))

    if mover.piece_type == chess.KING:
        # The king itself is never a feature (that's the "half" in HalfKP) — only the anchor
        # shift matters, handled by a full refresh of the mover's own side after the push.
        return updates, mover.color

    updates.append((move.from_square, mover.piece_type, mover.color, -1))
    new_type = move.promotion if move.promotion else mover.piece_type
    updates.append((move.to_square, new_type, mover.color, +1))
    return updates, None


def _apply(acc: np.ndarray, weight: np.ndarray, perspective: bool, king_square: int,
           updates: list[Update]) -> None:
    king_sq = _orient(king_square, perspective)
    for square, piece_type, color, sign in updates:
        piece_square = _orient(square, perspective)
        relative_color = 0 if color == perspective else 1
        combined_type = relative_color * 5 + PIECE_INDEX_NO_KING[piece_type]
        index = king_sq * 640 + piece_square * 10 + combined_type
        if sign > 0:
            acc += weight[index]
        else:
            acc -= weight[index]


class Accumulator:
    """Owns both perspectives' running accumulator and the board itself, so the two can never
    drift out of sync: every push/pop goes through here instead of `board.push`/`board.pop`."""

    def __init__(self, weight: np.ndarray) -> None:
        self.weight = weight  # int16
        self.white_acc = np.zeros(ACCUMULATOR_DIM, dtype=_ACC_DTYPE)
        self.black_acc = np.zeros(ACCUMULATOR_DIM, dtype=_ACC_DTYPE)
        self._stack: list[tuple[np.ndarray, np.ndarray]] = []

    def set_position(self, board: chess.Board) -> None:
        self.white_acc = self._full(board, chess.WHITE)
        self.black_acc = self._full(board, chess.BLACK)
        self._stack.clear()

    def _full(self, board: chess.Board, perspective: bool) -> np.ndarray:
        indices = halfkp_indices(board, perspective)
        if not indices:
            return np.zeros(ACCUMULATOR_DIM, dtype=_ACC_DTYPE)
        return self.weight[indices].sum(axis=0, dtype=_ACC_DTYPE)

    def accumulators_for(self, side_to_move: bool) -> np.ndarray:
        stm, nstm = (
            (self.white_acc, self.black_acc)
            if side_to_move == chess.WHITE
            else (self.black_acc, self.white_acc)
        )
        return np.concatenate([stm, nstm])

    def push(self, board: chess.Board, move: chess.Move) -> None:
        self._stack.append((self.white_acc.copy(), self.black_acc.copy()))
        if move == chess.Move.null():
            board.push(move)
            return

        updates, king_moved_color = move_updates(board, move)
        for perspective, acc in ((chess.WHITE, self.white_acc), (chess.BLACK, self.black_acc)):
            if perspective == king_moved_color:
                continue
            _apply(acc, self.weight, perspective, board.king(perspective), updates)

        board.push(move)

        if king_moved_color == chess.WHITE:
            self.white_acc = self._full(board, chess.WHITE)
        elif king_moved_color == chess.BLACK:
            self.black_acc = self._full(board, chess.BLACK)

    def pop(self, board: chess.Board) -> None:
        board.pop()
        self.white_acc, self.black_acc = self._stack.pop()
