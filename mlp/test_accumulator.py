"""Checks Accumulator's incremental push/pop against a from-scratch recompute at every ply.

Run directly: `python test_accumulator.py`. Not part of the training pipeline — this exists
because an incremental-update bug (a missed en passant capture, a castling rook, a king move
that should have triggered a refresh) would otherwise just look like a slightly-off evaluation
forever, not a crash. The accumulator is integer (int16 weight rows summed into an int32
running total), so incremental and from-scratch must match exactly — no tolerance needed.
"""

import random
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nnue import Accumulator
from train import ACCUMULATOR_DIM, FEATURE_DIM

# Positions chosen to force specific move types soon after the game starts: castling (both
# sides), en passant, promotion, and a capture, rather than hoping random play stumbles into them.
STRESS_FENS = [
    chess.STARTING_FEN,
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",  # both sides can castle either way
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",  # black can play c5, enabling en passant
    "8/P7/8/8/8/8/8/k6K w - - 0 1",  # white pawn one step from promoting
]


def assert_matches(acc: Accumulator, board: chess.Board, context: str) -> None:
    expected_white = acc._full(board, chess.WHITE)
    expected_black = acc._full(board, chess.BLACK)
    if not np.array_equal(acc.white_acc, expected_white):
        raise AssertionError(f"white accumulator mismatch {context}\nfen={board.fen()}")
    if not np.array_equal(acc.black_acc, expected_black):
        raise AssertionError(f"black accumulator mismatch {context}\nfen={board.fen()}")


def play_random_game(weight: np.ndarray, start_fen: str, plies: int, rng: random.Random) -> None:
    board = chess.Board(start_fen)
    acc = Accumulator(weight)
    acc.set_position(board)
    assert_matches(acc, board, "at start")

    played: list[chess.Move] = []
    for ply in range(plies):
        legal = list(board.legal_moves)
        if not legal:
            break
        move = rng.choice(legal)
        acc.push(board, move)
        played.append(move)
        assert_matches(acc, board, f"after ply {ply} ({move.uci()})")

    for ply in reversed(range(len(played))):
        acc.pop(board)
        assert_matches(acc, board, f"after popping back to ply {ply}")


def main() -> None:
    rng = random.Random(0)
    weight = rng_weight(rng)

    for fen in STRESS_FENS:
        for game in range(20):
            play_random_game(weight, fen, plies=40, rng=random.Random(game))
        print(f"ok: {fen}")

    print(f"all checks passed ({len(STRESS_FENS)} start positions x 20 games x 40 plies)")


def rng_weight(rng: random.Random) -> np.ndarray:
    np_rng = np.random.default_rng(rng.randint(0, 2**31))
    return np_rng.integers(
        -1000, 1000, size=(FEATURE_DIM + 1, ACCUMULATOR_DIM), dtype=np.int16
    )


if __name__ == "__main__":
    main()
