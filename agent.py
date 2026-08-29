"""The submission entrypoint. The platform imports this file and calls get_move."""

import random

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

MOBILITY_WEIGHT = 4.0
MATE = 1e6

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

# Import time runs once per game, inside a 60 second budget, before your clock starts.
# Load weights and build tables out here, not inside get_move.

def evaluate(board: chess.Board, mobility: int) -> float:
    def evaluate(board: chess.Board, mobility: int) -> float:
    
        if board.is_checkmate():
            return -MATE
        if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition():
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
            white_pieces = board.pieces(piece_type, chess.WHITE)
            for sq in white_pieces:
                score += PIECE_VALUE[piece_type]
                score += PST_MAP[piece_type][sq]
                
            black_pieces = board.pieces(piece_type, chess.BLACK)
            for sq in black_pieces:
                score -= PIECE_VALUE[piece_type]
                score -= PST_MAP[piece_type][chess.square_mirror(sq)]

        if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
            score += 50
        if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
            score -= 50

        if board.turn == chess.BLACK:
            score = -score

     
        score += MOBILITY_WEIGHT * mobility

        return score



def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in; your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    returns       "e2e4", or "e7e8q" for a promotion

    The process stays alive between your moves, so state you keep on a module or in a
    closure survives to the next call. It does not survive to the next game.

    print() is safe. Your stdout is redirected away from the protocol stream, discarded
    during rated games and shown back to you in the validation log.
    """
    board = chess.Board(fen)

    # Everything from here down is yours to replace. baselines/greedy searches one ply,
    # baselines/minimax searches two. Neither is strong. Reading them is the fastest way
    # to see the shape of a search, and beating them is the first real milestone.
    return random.choice(list(board.legal_moves)).uci()
