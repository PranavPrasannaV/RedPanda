
"""
Strategic theme definitions and annotation for chess positions.

This module defines a vocabulary of strategic themes that the model can predict.
These themes help bridge the gap between short-term tactics and long-term planning.
"""

import chess
from typing import List, Set
from enum import IntEnum

class StrategicTheme(IntEnum):
    """
    Strategic themes that can be active in a chess position.
    Multi-label: A position can have multiple active themes.
    """
    # Attacking themes
    KINGSIDE_ATTACK = 0
    QUEENSIDE_ATTACK = 1
    CENTER_CONTROL = 2
    PIECE_ACTIVITY = 3
    
    # Structural themes
    PAWN_BREAK = 4
    WEAK_SQUARES = 5
    OPEN_FILE = 6
    OUTPOST = 7
    
    # Endgame themes
    KING_ACTIVITY = 8
    PASSED_PAWN = 9
    ROOK_ACTIVITY = 10
    
    # Defensive themes
    PROPHYLAXIS = 11
    FORTRESS = 12
    CONSOLIDATION = 13
    
    # Positional themes
    MINORITY_ATTACK = 14
    SPACE_ADVANTAGE = 15
    BISHOP_PAIR = 16
    EXCHANGE_SACRIFICE = 17
    
    # Piece-specific
    KNIGHT_MANEUVER = 18
    ROOK_LIFT = 19
    BATTERY = 20
    
    # Tactical preparation
    PIN_EXPLOITATION = 21
    DISCOVERED_ATTACK_PREP = 22
    SACRIFICE_PREP = 23

NUM_THEMES = len(StrategicTheme)

def detect_themes(board: chess.Board) -> Set[StrategicTheme]:
    """
    Heuristically detect active strategic themes in a position.
    
    This is used for training data generation. The neural network will
    learn to predict these themes more accurately.
    
    Returns:
        Set of detected strategic themes
    """
    themes = set()
    turn = board.turn
    
    # 1. Piece activity
    if _has_active_pieces(board, turn):
        themes.add(StrategicTheme.PIECE_ACTIVITY)
    
    # 2. King activity (endgame)
    piece_count = len(board.piece_map())
    if piece_count <= 12:
        if _is_king_active(board, turn):
            themes.add(StrategicTheme.KING_ACTIVITY)
    
    # 3. Passed pawns
    if _has_passed_pawn(board, turn):
        themes.add(StrategicTheme.PASSED_PAWN)
    
    # 4. Open files (rook activity potential)
    if _has_rook_on_open_file(board, turn):
        themes.add(StrategicTheme.OPEN_FILE)
        themes.add(StrategicTheme.ROOK_ACTIVITY)
    
    # 5. Bishop pair
    if _has_bishop_pair(board, turn):
        themes.add(StrategicTheme.BISHOP_PAIR)
    
    # 6. Kingside attack potential
    if _has_kingside_attack_potential(board, turn):
        themes.add(StrategicTheme.KINGSIDE_ATTACK)
    
    # 7. Queenside attack potential
    if _has_queenside_attack_potential(board, turn):
        themes.add(StrategicTheme.QUEENSIDE_ATTACK)
    
    # 8. Space advantage
    if _has_space_advantage(board, turn):
        themes.add(StrategicTheme.SPACE_ADVANTAGE)
    
    # 9. Center control
    if _controls_center(board, turn):
        themes.add(StrategicTheme.CENTER_CONTROL)
    
    # 10. Outposts
    if _has_outpost(board, turn):
        themes.add(StrategicTheme.OUTPOST)
    
    return themes

def themes_to_vector(themes: Set[StrategicTheme]) -> List[float]:
    """Convert set of themes to a multi-hot vector."""
    vector = [0.0] * NUM_THEMES
    for theme in themes:
        vector[theme.value] = 1.0
    return vector

def _has_active_pieces(board: chess.Board, color: chess.Color) -> bool:
    """Check if minor/major pieces are on active squares (ranks 4-7 for white)."""
    active_count = 0
    rank_threshold = 3 if color == chess.WHITE else 4
    
    for sq, piece in board.piece_map().items():
        if piece.color == color and piece.piece_type not in [chess.PAWN, chess.KING]:
            rank = chess.square_rank(sq)
            if (color == chess.WHITE and rank >= rank_threshold) or \
               (color == chess.BLACK and rank <= (7 - rank_threshold)):
                active_count += 1
    
    return active_count >= 2

def _is_king_active(board: chess.Board, color: chess.Color) -> bool:
    """Check if king is centralized/active (endgame metric)."""
    king_sq = board.king(color)
    if king_sq is None:
        return False
    
    rank = chess.square_rank(king_sq)
    file = chess.square_file(king_sq)
    
    # King is active if near center or advanced
    center_distance = abs(rank - 3.5) + abs(file - 3.5)
    return center_distance <= 3

def _has_passed_pawn(board: chess.Board, color: chess.Color) -> bool:
    """Check if we have a passed pawn."""
    pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, not color)
    
    for pawn_sq in pawns:
        file = chess.square_file(pawn_sq)
        rank = chess.square_rank(pawn_sq)
        
        is_passed = True
        for enemy_sq in enemy_pawns:
            enemy_file = chess.square_file(enemy_sq)
            enemy_rank = chess.square_rank(enemy_sq)
            
            if abs(enemy_file - file) <= 1:
                if color == chess.WHITE and enemy_rank > rank:
                    is_passed = False
                    break
                elif color == chess.BLACK and enemy_rank < rank:
                    is_passed = False
                    break
        
        if is_passed:
            return True
    
    return False

def _has_rook_on_open_file(board: chess.Board, color: chess.Color) -> bool:
    """Check if we have a rook on an open or semi-open file."""
    rooks = board.pieces(chess.ROOK, color)
    our_pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, not color)
    
    for rook_sq in rooks:
        file = chess.square_file(rook_sq)
        
        # Check if file is open (no pawns) or semi-open (no friendly pawns)
        has_our_pawn = any(chess.square_file(sq) == file for sq in our_pawns)
        
        if not has_our_pawn:
            return True
    
    return False

def _has_bishop_pair(board: chess.Board, color: chess.Color) -> bool:
    """Check if we have both bishops."""
    bishops = board.pieces(chess.BISHOP, color)
    return len(bishops) >= 2

def _has_kingside_attack_potential(board: chess.Board, color: chess.Color) -> bool:
    """Check for attacking pieces aimed at kingside."""
    enemy_king = board.king(not color)
    if enemy_king is None:
        return False
    
    king_file = chess.square_file(enemy_king)
    
    # Enemy king on kingside
    if king_file < 4:
        return False
    
    # Count attacking pieces on kingside
    attack_pieces = 0
    for sq, piece in board.piece_map().items():
        if piece.color == color and piece.piece_type != chess.PAWN:
            file = chess.square_file(sq)
            if file >= 4:  # Kingside
                attack_pieces += 1
    
    return attack_pieces >= 3

def _has_queenside_attack_potential(board: chess.Board, color: chess.Color) -> bool:
    """Check for attacking pieces aimed at queenside."""
    enemy_king = board.king(not color)
    if enemy_king is None:
        return False
    
    king_file = chess.square_file(enemy_king)
    
    # Enemy king on queenside
    if king_file >= 4:
        return False
    
    attack_pieces = 0
    for sq, piece in board.piece_map().items():
        if piece.color == color and piece.piece_type != chess.PAWN:
            file = chess.square_file(sq)
            if file <= 3:  # Queenside
                attack_pieces += 1
    
    return attack_pieces >= 3

def _has_space_advantage(board: chess.Board, color: chess.Color) -> bool:
    """Check for pawn space advantage."""
    our_pawns = board.pieces(chess.PAWN, color)
    
    advanced_pawns = 0
    for sq in our_pawns:
        rank = chess.square_rank(sq)
        if color == chess.WHITE and rank >= 4:
            advanced_pawns += 1
        elif color == chess.BLACK and rank <= 3:
            advanced_pawns += 1
    
    return advanced_pawns >= 3

def _controls_center(board: chess.Board, color: chess.Color) -> bool:
    """Check if we control the center squares (d4, d5, e4, e5)."""
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    control_count = 0
    
    for sq in center_squares:
        attackers = board.attackers(color, sq)
        if len(attackers) > 0:
            control_count += 1
    
    return control_count >= 3

def _has_outpost(board: chess.Board, color: chess.Color) -> bool:
    """Check if we have a knight on an outpost square."""
    knights = board.pieces(chess.KNIGHT, color)
    enemy_pawns = board.pieces(chess.PAWN, not color)
    
    for knight_sq in knights:
        file = chess.square_file(knight_sq)
        rank = chess.square_rank(knight_sq)
        
        # Check if outpost (advanced and not attackable by enemy pawns)
        if (color == chess.WHITE and rank >= 4) or (color == chess.BLACK and rank <= 3):
            is_outpost = True
            for pawn_sq in enemy_pawns:
                pawn_file = chess.square_file(pawn_sq)
                pawn_rank = chess.square_rank(pawn_sq)
                
                if abs(pawn_file - file) == 1:
                    if color == chess.WHITE and pawn_rank > rank:
                        is_outpost = False
                        break
                    elif color == chess.BLACK and pawn_rank < rank:
                        is_outpost = False
                        break
            
            if is_outpost:
                return True
    
    return False
