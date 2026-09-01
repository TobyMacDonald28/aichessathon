"""NNUE agent: evaluate() is backed by an incrementally-updated accumulator instead of a
from-scratch forward pass. Every `board.push`/`board.pop` in the search below goes through
`acc.push`/`acc.pop` instead, so the two never drift apart.

Self-contained on purpose: the competition only ships one file, `agent.py`, so everything this
needs — HalfKP feature encoding, weight loading, and the accumulator — lives here rather than
being imported from mlp/train.py or mlp/quantize.py (those stay as the training-side tools that
produced weights.npz in the first place).
"""

import time
from collections import Counter
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import chess
import chess.polyglot
import numba
import numpy as np

MATE_SCORE = 100000

PIECE_VALUE = {
    chess.PAWN: 100.0,
    chess.KNIGHT: 320.0,
    chess.BISHOP: 330.0,
    chess.ROOK: 500.0,
    chess.QUEEN: 900.0,
}

# ---------------------------------------------------------------------------
# HalfKP feature encoding: a feature is "there is a <piece type, friend-or-foe> on <square>, given
# my king is on <king square>", computed once per side per position. Kings aren't features
# themselves (that's the "half" in HalfKP) — the king square is the anchor the other 40,960
# features are relative to.
# ---------------------------------------------------------------------------

ACCUMULATOR_DIM = 256

PIECE_INDEX_NO_KING = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}


def _orient(square: int, perspective: bool) -> int:
    """Square as seen by `perspective`: mirrored vertically when that side is Black, so the board
    always "looks like" it's being viewed from the bottom."""
    return square if perspective == chess.WHITE else chess.square_mirror(square)


def halfkp_indices(board: chess.Board, perspective: bool) -> list[int]:
    king = board.king(perspective)
    assert king is not None
    king_square = _orient(king, perspective)
    indices = []
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        piece_square = _orient(square, perspective)
        relative_color = 0 if piece.color == perspective else 1
        combined_type = relative_color * 5 + PIECE_INDEX_NO_KING[piece.piece_type]
        indices.append(king_square * 640 + piece_square * 10 + combined_type)
    return indices


# ---------------------------------------------------------------------------
# Weight loading. mlp/quantize.py is what produces weights.npz: it reads the float32 checkpoint
# training writes and saves plain numpy arrays, quantizing the two tensors below on the way. Numpy
# arrays, not a torch state_dict, so this file never needs `import torch` — nothing in the search
# loop below touches it either, and torch is a slow, ~40MB import to pay for on every game.
#
# Two different reasons to quantize, for two different tensors:
# - feature_transformer.weight is 40,961 x 256 float32, ~42MB on its own — uncomfortably close to
#   the zip's 50MB cap. int16 halves that. This one is about file size.
# - head.1.weight (512 x 32) is a few KB either way; quantizing it buys nothing on disk. It's
#   quantized so the matvec below can run as a genuine int8 SIMD-shaped dot product instead of a
#   float32 matmul. It's also the only head layer worth bothering with: at 512x32 multiply-adds it
#   does ~94% of the head's arithmetic, versus 32x32 and 32x1 for the two layers after it, so those
#   stay plain float32.
# ---------------------------------------------------------------------------

WEIGHTS_PATH = Path(__file__).parent / "weights.npz"
FEATURE_TRANSFORMER_BITS = 16  # summed over up to 32 rows; int8 rounding error compounds too much
HEAD_BITS = 8  # one matmul, not a sum of many rows — int8 rounds cleanly, and enables SIMD


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


def _quantize(weight: np.ndarray, bits: int) -> tuple[np.ndarray, float]:
    qmax = 2 ** (bits - 1) - 1
    max_abs = float(np.abs(weight).max())
    scale = max_abs / qmax if max_abs > 0 else 1.0
    dtype = np.int16 if bits == 16 else np.int8
    quantized = np.clip(np.round(weight / scale), -qmax - 1, qmax).astype(dtype)
    return quantized, scale


def _load_integer(npz: np.lib.npyio.NpzFile, key: str, bits: int) -> tuple[np.ndarray, float]:
    weight = npz[key]
    if np.issubdtype(weight.dtype, np.floating):
        return _quantize(weight, bits)
    return weight, float(npz[f"{key}.scale"])


def _load_float(npz: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    weight = npz[key].astype(np.float32)
    scale_key = f"{key}.scale"
    return weight * float(npz[scale_key]) if scale_key in npz.files else weight


def load_weights(path: Path = WEIGHTS_PATH) -> NNUEWeights:
    with np.load(path) as npz:
        ft_weight, ft_scale = _load_integer(
            npz, "feature_transformer.weight", FEATURE_TRANSFORMER_BITS
        )
        head1_weight, head1_scale = _load_integer(npz, "head.1.weight", HEAD_BITS)

        return NNUEWeights(
            ft_weight=ft_weight,
            ft_scale=ft_scale,
            head1_weight=head1_weight,
            head1_scale=head1_scale,
            head1_bias=_load_float(npz, "head.1.bias"),
            head3_weight=_load_float(npz, "head.3.weight"),
            head3_bias=_load_float(npz, "head.3.bias"),
            head5_weight=_load_float(npz, "head.5.weight"),
            head5_bias=_load_float(npz, "head.5.bias"),
        )


_WEIGHTS = load_weights()

# ---------------------------------------------------------------------------
# The int8 matvec kernel: compiled, not vectorized — a scalar multiply-accumulate loop, since
# LLVM's auto-vectorizer doesn't kick in for this widening int8->int32 reduction pattern. Still
# faster than the numpy alternatives here (no BLAS dispatch or temporary-array allocation for a
# matrix this small). An explicit signature makes numba compile eagerly, right here at import time,
# instead of lazily on the first call during search — the 60s import budget is exactly where that
# cost belongs.
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

    # Requantize the int32 accumulator down to int8 for the matvec. There's no fixed activation
    # range to calibrate against, so the scale is derived per call from the actual max — cheap
    # (one reduction over 512 ints) and exact for whatever range this position's activations
    # happen to occupy.
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
# The accumulator: instead of recomputing "sum the embedding rows for every piece on the board"
# after every move, push/pop only touch the embedding rows for the few squares a single move
# actually changes — one or two rows, not up to 30. The one exception is a king move: since the
# king's square is the anchor every other feature is expressed relative to, moving it changes what
# all ~30 of that side's features *mean*, so that side gets a full recompute (still only for that
# one side, not both).
# ---------------------------------------------------------------------------

Update = tuple[int, int, bool, int]  # (square, piece_type, color, sign)

# The accumulator sums up to 32 int16 weight rows (a legal position has at most 30 non-king
# pieces). int16 alone could overflow (32 * 32767 > 32767); int32 has all the headroom needed.
_ACC_DTYPE = np.int32


def move_updates(board: chess.Board, move: chess.Move) -> tuple[list[Update], bool | None]:
    """Given a move about to be played on `board` (not yet pushed), work out which HalfKP
    (square, piece_type, color) facts appeared or disappeared, and whether either king moved
    (which forces a full accumulator refresh for that side instead of an incremental update)."""
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
        return updates, mover.color

    updates.append((move.from_square, mover.piece_type, mover.color, -1))
    new_type = move.promotion if move.promotion else mover.piece_type
    updates.append((move.to_square, new_type, mover.color, +1))
    return updates, None


def _apply(
    acc: np.ndarray, weight: np.ndarray, perspective: bool, king_square: int, updates: list[Update]
) -> None:
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
            king = board.king(perspective)
            assert king is not None
            _apply(acc, self.weight, perspective, king, updates)

        board.push(move)

        if king_moved_color == chess.WHITE:
            self.white_acc = self._full(board, chess.WHITE)
        elif king_moved_color == chess.BLACK:
            self.black_acc = self._full(board, chess.BLACK)

    def pop(self, board: chess.Board) -> None:
        board.pop()
        self.white_acc, self.black_acc = self._stack.pop()


_ACC = Accumulator(_WEIGHTS.ft_weight)

# ---------------------------------------------------------------------------
# Search: alpha-beta negamax with a transposition table, quiescence search on captures, null-move
# pruning, and killer-move ordering. evaluate() is the only thing that differs from a classical
# engine — everything else here is standard.
# ---------------------------------------------------------------------------

class TTEntry(TypedDict):
    depth: int
    score: float
    flag: str
    best_move: str | None


GAME_HISTORY: Counter[Hashable] = Counter()
TT: dict[Hashable, TTEntry] = {}
KILLERS_PER_DEPTH = 2
KILLERS: dict[int, list[chess.Move]] = {}

SAFETY_MARGIN_MS = 50.0
MIN_TIME_LIMIT_SEC = 0.05

# Futility pruning: at a frontier node (depth 1, one ply from qsearch), a quiet move that doesn't
# give check can't plausibly swing the score by more than a couple of pieces in one move, so if
# the static eval already trails alpha by more than that, skip searching it instead of proving it.
# Always searches at least the first (best-ordered) move regardless, so best_score/best_move_for_tt
# never come out empty. Captures, promotions, and checks are exempt — those are exactly the moves
# that can swing eval by more than the margin allows.
FUTILITY_MARGIN = 200.0


class SearchTimeout(Exception):
    pass


def evaluate(board: chess.Board) -> float:
    """Score relative to the side to move: positive means the mover is better. Reads whatever
    _ACC's accumulator currently holds — the caller is responsible for keeping it in sync with
    `board` via push()/pop(), not this function."""
    if board.is_checkmate():
        return -MATE_SCORE + len(board.move_stack)
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    x = _ACC.accumulators_for(board.turn)
    return evaluate_head(x, _WEIGHTS)


def time_budget_sec(time_left_ms: int, board: chess.Board) -> float:
    expected_moves_left = max(20, 60 - board.fullmove_number)
    usable_ms = max(time_left_ms - SAFETY_MARGIN_MS, 0.0)
    budget_sec = max(usable_ms / 1000.0 / expected_moves_left, MIN_TIME_LIMIT_SEC)
    return min(budget_sec, time_left_ms / 1000.0 * 0.5)


def store_killer(depth: int, move: chess.Move) -> None:
    slots = KILLERS.setdefault(depth, [])
    if move in slots:
        return
    slots.insert(0, move)
    del slots[KILLERS_PER_DEPTH:]


def score_move(
    board: chess.Board,
    move: chess.Move,
    tt_move_uci: str | None = None,
    killers: list[chess.Move] | None = None,
) -> float:
    if tt_move_uci and move.uci() == tt_move_uci:
        return 1000000.0

    score = 0.0
    if move.promotion:
        score += 90000.0 + PIECE_VALUE.get(move.promotion, 0)

    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        victim_val = PIECE_VALUE.get(victim.piece_type, 100.0) if victim else 100.0
        attacker_val = PIECE_VALUE.get(attacker.piece_type, 100.0) if attacker else 100.0
        score += 10000.0 + victim_val - (attacker_val / 100.0)
    elif killers and move in killers:
        score += 9000.0

    return score


def qsearch(
    board: chess.Board, alpha: float, beta: float, start_time: float, time_limit: float
) -> float:
    if time.time() - start_time > time_limit:
        raise SearchTimeout()

    stand_pat = evaluate(board)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    captures = list(board.generate_legal_captures())
    captures.sort(key=lambda m: score_move(board, m), reverse=True)

    for move in captures:
        _ACC.push(board, move)
        score = -qsearch(board, -beta, -alpha, start_time, time_limit)
        _ACC.pop(board)

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def negamax(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    start_time: float,
    time_limit: float,
    path_keys: set[Hashable],
) -> float:
    if time.time() - start_time > time_limit:
        raise SearchTimeout()

    key = board._transposition_key()

    if GAME_HISTORY[key] >= 2 or key in path_keys:
        return 0.0

    alpha_orig = alpha
    tt_move_uci = None

    if key in TT:
        tt_entry = TT[key]
        tt_move_uci = tt_entry.get("best_move")
        if tt_entry["depth"] >= depth:
            if tt_entry["flag"] == "EXACT":
                return tt_entry["score"]
            elif tt_entry["flag"] == "LOWER":
                alpha = max(alpha, tt_entry["score"])
            elif tt_entry["flag"] == "UPPER":
                beta = min(beta, tt_entry["score"])
            if alpha >= beta:
                return tt_entry["score"]

    if depth == 0:
        return qsearch(board, alpha, beta, start_time, time_limit)

    in_check = board.is_check()

    if depth >= 3 and beta < MATE_SCORE and not in_check and len(board.piece_map()) > 10:
        _ACC.push(board, chess.Move.null())
        null_score = -negamax(
            board, depth - 1 - 2, -beta, -beta + 1, start_time, time_limit, path_keys
        )
        _ACC.pop(board)
        if null_score >= beta:
            return beta

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        if in_check:
            return -MATE_SCORE + len(board.move_stack)
        return 0.0

    killers = KILLERS.get(depth)
    legal_moves.sort(key=lambda m: score_move(board, m, tt_move_uci, killers), reverse=True)

    static_eval = evaluate(board) if depth == 1 and not in_check and beta < MATE_SCORE else None

    best_score = -float("inf")
    best_move_for_tt = None
    path_keys.add(key)

    for move_index, move in enumerate(legal_moves):
        is_tactical = board.is_capture(move) or move.promotion is not None
        gives_check = board.gives_check(move)

        if (
            static_eval is not None
            and not is_tactical
            and not gives_check
            and move_index > 0
            and static_eval + FUTILITY_MARGIN <= alpha
        ):
            continue

        _ACC.push(board, move)
        # Check extension: a move that gives check doesn't cost depth, so a forcing sequence of
        # checks gets resolved instead of getting cut off right at the search horizon.
        child_depth = depth if gives_check else depth - 1
        score = -negamax(board, child_depth, -beta, -alpha, start_time, time_limit, path_keys)
        _ACC.pop(board)

        if score > best_score:
            best_score = score
            best_move_for_tt = move.uci()

        if best_score > alpha:
            alpha = best_score
        if alpha >= beta:
            if not is_tactical:
                store_killer(depth, move)
            break

    path_keys.remove(key)

    # A score of exactly 0.0 here can only come from a terminal draw (repetition, stalemate, or
    # insufficient material) propagating up through best_score — evaluate()'s NNUE output is a
    # continuous float that never lands on exactly 0.0. Repetition specifically is path-dependent:
    # the same board key can be a draw via one move-order history and not a draw via a different
    # one that reaches the identical position, so caching a repetition-tainted 0.0 as EXACT would
    # feed a stale "it's a draw" verdict to a later search that reaches this key a different way.
    if best_score != 0.0:
        flag = "EXACT"
        if best_score <= alpha_orig:
            flag = "UPPER"
        elif best_score >= beta:
            flag = "LOWER"

        TT[key] = {
            "depth": depth,
            "score": best_score,
            "flag": flag,
            "best_move": best_move_for_tt,
        }

    return best_score


def get_move(fen: str, time_left_ms: int) -> str:
    global GAME_HISTORY, TT, KILLERS

    board = chess.Board(fen)
    _ACC.set_position(board)
    start_time = time.time()
    try:
        with chess.polyglot.open_reader("book.bin") as reader:
            book_entry = reader.choice(board)
            return book_entry.move.uci()
    except (FileNotFoundError, IndexError):
        pass

    if board.halfmove_clock == 0:
        GAME_HISTORY.clear()

    key = board._transposition_key()
    GAME_HISTORY[key] += 1

    base_path_keys = set(GAME_HISTORY.keys())

    if len(TT) > 1000000:
        TT = {}

    KILLERS = {}

    time_limit_sec = time_budget_sec(time_left_ms, board)

    moves = list(board.legal_moves)
    if not moves:
        return ""

    best_move = moves[0]

    try:
        for depth in range(1, 20):
            alpha = -float("inf")
            beta = float("inf")
            best_score = -float("inf")

            moves.sort(key=lambda m: score_move(board, m, best_move.uci()), reverse=True)
            current_best_move = moves[0]

            for move in moves:
                _ACC.push(board, move)
                path_keys = base_path_keys.copy()
                score = -negamax(
                    board, depth - 1, -beta, -alpha, start_time, time_limit_sec, path_keys
                )
                _ACC.pop(board)

                if score > best_score:
                    best_score = score
                    current_best_move = move
                if best_score > alpha:
                    alpha = best_score

            best_move = current_best_move

    except SearchTimeout:
        pass

    return best_move.uci()
