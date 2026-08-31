"""The submission entrypoint. The platform imports this file and calls get_move."""

import random
import sys
import time
import chess
from collections import Counter
import chess.polyglot

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

# Endgame-only tables: the king wants to be active/central once the queens and
# rooks are off, and passed pawns matter far more than opening-square nuance.
KING_PST_EG = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
]

PAWN_PST_EG = [
     0,  0,  0,  0,  0,  0,  0,  0,
    10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10,
    20, 20, 20, 20, 20, 20, 20, 20,
    30, 30, 30, 30, 30, 30, 30, 30,
    50, 50, 50, 50, 50, 50, 50, 50,
    80, 80, 80, 80, 80, 80, 80, 80,
     0,  0,  0,  0,  0,  0,  0,  0
]

PST_MG = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.KING: KING_PST,
    chess.QUEEN: [0] * 64
}

PST_EG = {
    chess.PAWN: PAWN_PST_EG,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.KING: KING_PST_EG,
    chess.QUEEN: [0] * 64
}

# Tapered-eval phase weights (classic PeSTO scheme): sum of these over every
# knight/bishop/rook/queen still on the board, per side, maxes out at 24 when
# both armies are at full strength and falls to 0 in a bare-king endgame.
PHASE_WEIGHTS = {
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
}
MAX_PHASE = 24

MASK_64 = 0xFFFFFFFFFFFFFFFF
NOT_A_FILE = ~chess.BB_FILE_A & MASK_64
NOT_H_FILE = ~chess.BB_FILE_H & MASK_64

PASSED_PAWN_BONUS = [0, 40, 50, 50, 75, 120, 200, 0]


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

    mg_score = 0.0
    eg_score = 0.0
    phase = 0

    for piece_type in PIECE_VALUE:
        for sq in board.pieces(piece_type, chess.WHITE):
            mg_score += PIECE_VALUE[piece_type] + PST_MG[piece_type][sq]
            eg_score += PIECE_VALUE[piece_type] + PST_EG[piece_type][sq]
            phase += PHASE_WEIGHTS.get(piece_type, 0)
        for sq in board.pieces(piece_type, chess.BLACK):
            mg_score -= PIECE_VALUE[piece_type] + PST_MG[piece_type][chess.square_mirror(sq)]
            eg_score -= PIECE_VALUE[piece_type] + PST_EG[piece_type][chess.square_mirror(sq)]
            phase += PHASE_WEIGHTS.get(piece_type, 0)

    for sq in board.pieces(chess.KING, chess.WHITE):
        mg_score += PST_MG[chess.KING][sq]
        eg_score += PST_EG[chess.KING][sq]
    for sq in board.pieces(chess.KING, chess.BLACK):
        mg_score -= PST_MG[chess.KING][chess.square_mirror(sq)]
        eg_score -= PST_EG[chess.KING][chess.square_mirror(sq)]

    mg_phase = min(phase, MAX_PHASE)
    eg_phase = MAX_PHASE - mg_phase
    score += (mg_score * mg_phase + eg_score * eg_phase) / MAX_PHASE

    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2: score += 50
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2: score -= 50

    if board.turn == chess.BLACK:
        score = -score

    score += MOBILITY_WEIGHT * mobility
    return score

def score_move(board: chess.Board, move: chess.Move, tt_move_uci: str = None) -> float:
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
        
    return score

def qsearch(board: chess.Board, alpha: float, beta: float, start_time: float, time_limit: float) -> float:
    if time.time() - start_time > time_limit:
        raise SearchTimeout()

    stand_pat = evaluate(board, 0)
    
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    captures = list(board.generate_legal_captures())
    captures.sort(key=lambda m: score_move(board, m), reverse=True)
    
    for move in captures:
        board.push(move)
        score = -qsearch(board, -beta, -alpha, start_time, time_limit)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha

def negamax(board: chess.Board, depth: int, alpha: float, beta: float, start_time: float, time_limit: float, path_keys: set) -> float:
    if time.time() - start_time > time_limit:
        raise SearchTimeout()

    key = board._transposition_key()
    
    # 1. Draw detection
    if GAME_HISTORY[key] >= 2 or key in path_keys:
        return 0.0

    alpha_orig = alpha
    tt_move_uci = None
    
    # 2. Transposition Table Probe
    if key in TT:
        tt_entry = TT[key]
        tt_move_uci = tt_entry.get('best_move') # Extract the best move to sort it first later
        
        if tt_entry['depth'] >= depth:
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

    # 3. Null Move Pruning (NMP)
    # If we have depth, aren't in check, and it's not a late endgame (avoiding zugzwang)
    if depth >= 3 and beta < MATE_SCORE and not board.is_check() and len(board.piece_map()) > 10:
        board.push(chess.Move.null())
        # Search with reduced depth (R=2) and a zero-window
        null_score = -negamax(board, depth - 1 - 2, -beta, -beta + 1, start_time, time_limit, path_keys)
        board.pop()
        
        if null_score >= beta:
            return beta # Prune this branch immediately

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        if board.is_check():
            return -MATE_SCORE + len(board.move_stack)
        return 0.0

    legal_moves.sort(key=lambda m: score_move(board, m, tt_move_uci), reverse=True)

    best_score = -float('inf')
    best_move_for_tt = None
    path_keys.add(key)

    for move in legal_moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, start_time, time_limit, path_keys)
        board.pop()

        if score > best_score:
            best_score = score
            best_move_for_tt = move.uci() # Remember the move that gave us the best score
            
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
        
    TT[key] = {'depth': depth, 'score': best_score, 'flag': flag, 'best_move': best_move_for_tt}

    return best_score

def get_move(fen: str, time_left_ms: int) -> str:
    global GAME_HISTORY, TT

    board = chess.Board(fen)
    start_time = time.time()
    try:
        with chess.polyglot.open_reader("book.bin") as reader:
            book_entry = reader.choice(board)
            move_uci = book_entry.move.uci()
            return move_uci
    except (FileNotFoundError, IndexError):
        pass

    if board.halfmove_clock == 0:
        GAME_HISTORY.clear()
        
    key = board._transposition_key()
    GAME_HISTORY[key] += 1

    base_path_keys = set(GAME_HISTORY.keys())
    
    if len(TT) > 1000000:
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
                
                score = -negamax(board, depth - 1, -beta, -alpha, start_time, time_limit_sec, path_keys)
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