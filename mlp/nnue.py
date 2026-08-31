"""Runtime NNUE inference: the incrementally-updated accumulator that makes evaluating a
position during search cheap, instead of the from-scratch forward pass train.py uses.

The key idea, and the actual point of NNUE: `Accumulator.push`/`pop` mirror `board.push`/`pop`
move for move. Instead of recomputing "sum the embedding rows for every piece on the board" after
every move (train.py's `NNUE.forward` does exactly that), this only touches the embedding rows
for the few squares a single move actually changes — one or two rows, not up to 30. The one
exception is a king move: since the king's square is the anchor every other feature is expressed
relative to, moving it changes what all ~30 of that side's features *mean*, so that side gets a
full recompute (still only for that one side, not both).

This module also runs the small head network in plain numpy rather than torch — once the
accumulator gives you the 512-wide vector, the rest is two tiny matmuls, and skipping torch
avoids per-call tensor-construction overhead in the search's hot path.
"""

import sys
from pathlib import Path

import chess
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train import (
    ACCUMULATOR_DIM,
    PIECE_INDEX_NO_KING,
    WEIGHTS_PATH,
    _orient,
    halfkp_indices,
)

Update = tuple[int, int, bool, int]  # (square, piece_type, color, sign)


# ---------------------------------------------------------------------------
# Loading weights: dequantized (if quantized) straight to float32 numpy arrays.
# The accumulator and head both run in float32 regardless of how the weights were
# stored on disk — see quantize.py for why quantization only saves file size here,
# not runtime arithmetic cost.
# ---------------------------------------------------------------------------


def load_weights(path: Path = WEIGHTS_PATH) -> dict[str, np.ndarray]:
    state = torch.load(path, map_location="cpu")
    if all(isinstance(v, torch.Tensor) for v in state.values()):
        return {k: v.numpy().astype(np.float32) for k, v in state.items()}
    # quantize.py's bundle format: {key: (tensor, scale)}
    return {k: (tensor.numpy().astype(np.float32) * scale) for k, (tensor, scale) in state.items()}


def head_forward(x: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    x = np.maximum(x, 0.0)
    x = x @ weights["head.1.weight"].T + weights["head.1.bias"]
    x = np.maximum(x, 0.0)
    x = x @ weights["head.3.weight"].T + weights["head.3.bias"]
    x = np.maximum(x, 0.0)
    x = x @ weights["head.5.weight"].T + weights["head.5.bias"]
    return x


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
        self.weight = weight
        self.white_acc = np.zeros(ACCUMULATOR_DIM, dtype=np.float32)
        self.black_acc = np.zeros(ACCUMULATOR_DIM, dtype=np.float32)
        self._stack: list[tuple[np.ndarray, np.ndarray]] = []

    def set_position(self, board: chess.Board) -> None:
        self.white_acc = self._full(board, chess.WHITE)
        self.black_acc = self._full(board, chess.BLACK)
        self._stack.clear()

    def _full(self, board: chess.Board, perspective: bool) -> np.ndarray:
        indices = halfkp_indices(board, perspective)
        if not indices:
            return np.zeros(ACCUMULATOR_DIM, dtype=np.float32)
        return self.weight[indices].sum(axis=0)

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
