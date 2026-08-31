"""NNUE agent variant: same search shell as the root agent.py, but evaluate() is backed by the
incrementally-updated Accumulator in nnue.py instead of a from-scratch forward pass. Every
`board.push`/`board.pop` in the search below goes through `acc.push`/`acc.pop` instead, so the
two never drift apart. Not the competition submission — test with
`make play --white mlp --black .` before ever promoting it to the zip root.
"""

import sys
import time
from collections import Counter
from pathlib import Path

import chess
import chess.polyglot
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nnue import Accumulator, head_forward, load_weights

MATE = 1e6
MATE_SCORE = 100000

_WEIGHTS = load_weights()
_ACC = Accumulator(_WEIGHTS["feature_transformer.weight"])

PIECE_VALUE = {
    chess.PAWN: 100.0,
    chess.KNIGHT: 320.0,
    chess.BISHOP: 330.0,
    chess.ROOK: 500.0,
    chess.QUEEN: 900.0,
}

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
    x = _ACC.accumulators_for(board.turn)[np.newaxis, :]
    return float(head_forward(x, _WEIGHTS)[0, 0])


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
