"""The submission entrypoint. The platform imports this file and calls get_move."""

import random
import time
from collections import Counter

import chess

PIECE_VALUE = {
    chess.PAWN: 100.0,
    chess.KNIGHT: 320.0,
    chess.BISHOP: 330.0,
    chess.ROOK: 500.0,
    chess.QUEEN: 900.0,
}

MOBILITY_WEIGHT = 4.0
MATE = 1e6
MATE_SCORE = 100000

PAWN_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_PST = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

BISHOP_PST = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

ROOK_PST = [
     0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
]

KING_PST = [
     20, 30, 10,  0,  0, 10, 30, 20,
     20, 20,  0,  0,  0,  0, 20, 20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30
]

PST_MAP = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.KING: KING_PST,
    chess.QUEEN: [0] * 64 
}

MASK_64 = 0xFFFFFFFFFFFFFFFF
NOT_A_FILE = ~chess.BB_FILE_A & MASK_64
NOT_H_FILE = ~chess.BB_FILE_H & MASK_64

PASSED_PAWN_BONUS = [0, 40, 50, 50, 75, 120, 200, 0]

OPENING_BOOK = {
    chess.STARTING_FEN: ["e2e4", "d2d4", "c2c4", "g1f3"],
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1": [
        "c7c5",
        "e7e5",
        "e7e6",
        "c7c6",
    ],
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1": ["d7d5", "g8f6", "e7e6"],
}

GAME_HISTORY = Counter()
TT = {} 

class SearchTimeout(Exception):
    pass

def evaluate(board: chess.Board, mobility: int) -> float:
    if board.is_checkmate():
        return -MATE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    score = 0.0

    white_pawns = board.pieces_mask(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces_mask(chess.PAWN, chess.BLACK)

    b_span = black_pawns
    b_span |= (b_span >> 8)
    b_span |= (b_span >> 16)
    b_span |= (b_span >> 32)
    b_stop_zone = b_span | ((b_span & NOT_A_FILE) >> 1) | ((b_span & NOT_H_FILE) << 1)
    white_passed = white_pawns & ~b_stop_zone
    
    w_span = white_pawns
    w_span = (w_span | (w_span << 8)) & MASK_64
    w_span = (w_span | (w_span << 16)) & MASK_64
    w_span = (w_span | (w_span << 32)) & MASK_64
    w_stop_zone = w_span | ((w_span & NOT_A_FILE) >> 1) | ((w_span & NOT_H_FILE) << 1)
    black_passed = black_pawns & ~w_stop_zone

    for sq in chess.SquareSet(white_passed):
        score += PASSED_PAWN_BONUS[chess.square_rank(sq)]
    for sq in chess.SquareSet(black_passed):
        score -= PASSED_PAWN_BONUS[7 - chess.square_rank(sq)]

    for piece_type in PIECE_VALUE:
        for sq in board.pieces(piece_type, chess.WHITE):
            score += PIECE_VALUE[piece_type] + PST_MAP[piece_type][sq]
        for sq in board.pieces(piece_type, chess.BLACK):
            score -= PIECE_VALUE[piece_type] + PST_MAP[piece_type][chess.square_mirror(sq)]

    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += 50
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= 50

    if board.turn == chess.BLACK:
        score = -score

    score += MOBILITY_WEIGHT * mobility
    return score

def qsearch(
    board: chess.Board, alpha: float, beta: float, start_time: float, time_limit: float
) -> float:
    if time.time() - start_time > time_limit:
        raise SearchTimeout()

    stand_pat = evaluate(board, 0)
    
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    captures = list(board.generate_legal_captures())
    captures.sort(key=lambda m: m.promotion is not None, reverse=True)
    
    for move in captures:
        board.push(move)
        score = -qsearch(board, -beta, -alpha, start_time, time_limit)
        board.pop()

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
    if key in TT and TT[key]['depth'] >= depth:
        tt_entry = TT[key]
        if tt_entry['flag'] == 'EXACT':
            return tt_entry['score']
        elif tt_entry['flag'] == 'LOWER':
            alpha = max(alpha, tt_entry['score'])
        elif tt_entry['flag'] == 'UPPER':
            beta = min(beta, tt_entry['score'])
        if alpha >= beta:
            return tt_entry['score']

    if depth == 0:
        return qsearch(board, alpha, beta, start_time, time_limit)

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        if board.is_check():
            return -MATE_SCORE + len(board.move_stack)
        return 0.0

    legal_moves.sort(key=lambda m: (board.is_capture(m), m.promotion is not None), reverse=True)

    best_score = -float('inf')

    path_keys.add(key)

    for move in legal_moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, start_time, time_limit, path_keys)
        board.pop()

        if score > best_score:
            best_score = score
        if best_score > alpha:
            alpha = best_score
        if alpha >= beta:
            break

    path_keys.remove(key)

    flag = 'EXACT'
    if best_score <= alpha_orig:
        flag = 'UPPER'
    elif best_score >= beta:
        flag = 'LOWER'
        
    TT[key] = {'depth': depth, 'score': best_score, 'flag': flag}

    return best_score

def get_move(fen: str, time_left_ms: int) -> str:
    global GAME_HISTORY, TT
    
    if fen in OPENING_BOOK:
        return random.choice(OPENING_BOOK[fen])

    board = chess.Board(fen)
    start_time = time.time()

    if board.halfmove_clock == 0:
        GAME_HISTORY.clear()
        
    key = board._transposition_key()
    GAME_HISTORY[key] += 1

    base_path_keys = set(GAME_HISTORY.keys())
    
    TT.clear() 
    
    time_limit_sec = (time_left_ms / 1000.0) / 25.0
    if time_limit_sec < 0.1:
        time_limit_sec = 0.1 

    moves = list(board.legal_moves)
    if not moves:
        return ""
    
    best_move = moves[0]

    try:
        for depth in range(1, 20):
            alpha = -float('inf')
            beta = float('inf')
            best_score = -float('inf')
            
            moves.sort(key=lambda m: (board.is_capture(m), m.promotion is not None), reverse=True)
            current_best_move = moves[0]

            for move in moves:
                board.push(move)
                
                path_keys = base_path_keys.copy()

                score = -negamax(
                    board, depth - 1, -beta, -alpha, start_time, time_limit_sec, path_keys
                )
                board.pop()

                if score > best_score:
                    best_score = score
                    current_best_move = move
                if best_score > alpha:
                    alpha = best_score

            best_move = current_best_move
            
    except SearchTimeout:
        pass

    return best_move.uci()