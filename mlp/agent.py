"""NNUE agent variant: same search shell as the root agent.py, but evaluate() is backed by an
incrementally-updated accumulator instead of a from-scratch forward pass. Every
`board.push`/`board.pop` in the search below goes through `acc.push`/`acc.pop` instead, so the
two never drift apart.

Self-contained on purpose: the competition only ships one file, `agent.py`, so everything this
needs — HalfKP feature encoding, weight loading/quantization, and the accumulator — lives here
rather than being imported from train.py/quantize.py (those stay as the training-side tools that
produced weights.pt in the first place). Promoting this file to the zip root is just a copy plus
shipping weights.pt (or a quantized weights.int8.pt) alongside it.

Not the competition submission yet — test with `make play --white mlp --black .` before ever
promoting it to the zip root.
"""

import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.polyglot
import numba
import numpy as np
import torch

MATE = 1e6
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
# themselves (that's the "half" in HalfKP) — the king square is the anchor the 40,960 piece
# features are relative to. 4 more features (see castling_indices below) round FEATURE_DIM out to
# 40,964: each side's castling rights, which unlike every piece feature are deliberately NOT
# multiplied by king_square — "I can still castle kingside" is one fact, true or false, not 64
# different facts depending on exactly which square my king happens to occupy.
# ---------------------------------------------------------------------------

ACCUMULATOR_DIM = 256

PIECE_INDEX_NO_KING = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}
PIECE_FEATURE_DIM = 64 * 64 * 10
CASTLING_FEATURE_DIM = 4
FEATURE_DIM = PIECE_FEATURE_DIM + CASTLING_FEATURE_DIM


def _orient(square: int, perspective: bool) -> int:
    """Square as seen by `perspective`: mirrored vertically when that side is Black, so the board
    always "looks like" it's being viewed from the bottom."""
    return square if perspective == chess.WHITE else chess.square_mirror(square)


def castling_indices(board: chess.Board, perspective: bool) -> list[int]:
    indices = []
    for relative_color, color in ((0, perspective), (1, not perspective)):
        if board.has_kingside_castling_rights(color):
            indices.append(PIECE_FEATURE_DIM + relative_color * 2 + 0)
        if board.has_queenside_castling_rights(color):
            indices.append(PIECE_FEATURE_DIM + relative_color * 2 + 1)
    return indices


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
    indices.extend(castling_indices(board, perspective))
    return indices


# ---------------------------------------------------------------------------
# Weight loading. weights.pt straight out of train.py is a plain {key: tensor} state_dict; running
# it through quantize.py turns it into {key: (tensor, scale)} for the two tensors worth
# quantizing. This loader accepts either — a plain tensor gets quantized here on the spot, so the
# runtime path below is always integer regardless of which file was shipped.
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

WEIGHTS_PATH = Path(__file__).parent / "weights.pt"
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


StateDict = dict[str, torch.Tensor | tuple[torch.Tensor, float]]


def _quantize_tensor(weight: torch.Tensor, bits: int) -> tuple[torch.Tensor, float]:
    qmax = 2 ** (bits - 1) - 1
    max_abs = weight.abs().max().item()
    scale = max_abs / qmax if max_abs > 0 else 1.0
    dtype = torch.int16 if bits == 16 else torch.int8
    quantized = torch.clamp(torch.round(weight / scale), -qmax - 1, qmax).to(dtype)
    return quantized, scale


def _load_integer(state: StateDict, key: str, bits: int, dtype: type) -> tuple[np.ndarray, float]:
    value = state[key]
    tensor, scale = value if isinstance(value, tuple) else _quantize_tensor(value, bits)
    return tensor.numpy().astype(dtype), scale


def _load_float(state: StateDict, key: str) -> np.ndarray:
    value = state[key]
    tensor, scale = value if isinstance(value, tuple) else (value, 1.0)
    return tensor.numpy().astype(np.float32) * scale


def load_weights(path: Path = WEIGHTS_PATH) -> NNUEWeights:
    state = torch.load(path, map_location="cpu")

    ft_weight, ft_scale = _load_integer(
        state, "feature_transformer.weight", FEATURE_TRANSFORMER_BITS, np.int16
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


CastlingRights = tuple[bool, bool, bool, bool]  # (white KS, white QS, black KS, black QS)
_CASTLING_LABELS: tuple[tuple[bool, int], ...] = (
    (chess.WHITE, 0),
    (chess.WHITE, 1),
    (chess.BLACK, 0),
    (chess.BLACK, 1),
)


def _castling_rights(board: chess.Board) -> CastlingRights:
    return (
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    )


def _lost_castling_rights(before: CastlingRights, after: CastlingRights) -> list[tuple[bool, int]]:
    """(color, right) pairs — right 0=kingside, 1=queenside — present in `before` but gone in
    `after`. This diffs python-chess's own has_*_castling_rights() across the push rather than
    hand-coding "king moved" / "rook moved off its home square" / "rook got captured on its home
    square" ourselves — that reuses logic python-chess already gets right (including the
    easy-to-miss case of a rook lost by capture without ever having moved) instead of duplicating
    it, and it's the only way a right can ever change: castling rights never come back once lost,
    so there's no symmetric "gained" case to handle."""
    return [
        label
        for label, was, now in zip(_CASTLING_LABELS, before, after, strict=True)
        if was and not now
    ]


def _apply_castling_losses(
    acc: np.ndarray, weight: np.ndarray, perspective: bool, lost: list[tuple[bool, int]]
) -> None:
    for color, right in lost:
        relative_color = 0 if color == perspective else 1
        acc -= weight[PIECE_FEATURE_DIM + relative_color * 2 + right]


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
        rights_before = _castling_rights(board)
        for perspective, acc in ((chess.WHITE, self.white_acc), (chess.BLACK, self.black_acc)):
            if perspective == king_moved_color:
                continue
            king = board.king(perspective)
            assert king is not None
            _apply(acc, self.weight, perspective, king, updates)

        board.push(move)

        lost_rights = _lost_castling_rights(rights_before, _castling_rights(board))
        if lost_rights:
            for perspective, acc in (
                (chess.WHITE, self.white_acc),
                (chess.BLACK, self.black_acc),
            ):
                if perspective == king_moved_color:
                    continue  # gets a full refresh below, which reflects the new rights anyway
                _apply_castling_losses(acc, self.weight, perspective, lost_rights)

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

GAME_HISTORY = Counter()
TT: dict = {}
KILLERS_PER_DEPTH = 2
KILLERS: dict[int, list[chess.Move]] = {}

SAFETY_MARGIN_MS = 50.0
MIN_TIME_LIMIT_SEC = 0.05


class SearchTimeout(Exception):
    pass


def evaluate(board: chess.Board) -> float:
    """Score relative to the side to move: positive means the mover is better. Reads whatever
    _ACC's accumulator currently holds — the caller is responsible for keeping it in sync with
    `board` via push()/pop(), not this function."""
    if board.is_checkmate():
        return -MATE
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
    path_keys: set,
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

    if depth >= 3 and beta < MATE_SCORE and not board.is_check() and len(board.piece_map()) > 10:
        _ACC.push(board, chess.Move.null())
        null_score = -negamax(
            board, depth - 1 - 2, -beta, -beta + 1, start_time, time_limit, path_keys
        )
        _ACC.pop(board)
        if null_score >= beta:
            return beta

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        if board.is_check():
            return -MATE_SCORE + len(board.move_stack)
        return 0.0

    killers = KILLERS.get(depth)
    legal_moves.sort(key=lambda m: score_move(board, m, tt_move_uci, killers), reverse=True)

    best_score = -float("inf")
    best_move_for_tt = None
    path_keys.add(key)

    for move in legal_moves:
        _ACC.push(board, move)
        score = -negamax(board, depth - 1, -beta, -alpha, start_time, time_limit, path_keys)
        _ACC.pop(board)

        if score > best_score:
            best_score = score
            best_move_for_tt = move.uci()

        if best_score > alpha:
            alpha = best_score
        if alpha >= beta:
            if not board.is_capture(move) and move.promotion is None:
                store_killer(depth, move)
            break

    path_keys.remove(key)

    flag = "EXACT"
    if best_score <= alpha_orig:
        flag = "UPPER"
    elif best_score >= beta:
        flag = "LOWER"

    TT[key] = {"depth": depth, "score": best_score, "flag": flag, "best_move": best_move_for_tt}

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

            moves.sort(key=lambda m: (board.is_capture(m), m.promotion is not None), reverse=True)
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
