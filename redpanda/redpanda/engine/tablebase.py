
"""
Syzygy tablebase integration for perfect endgame play.
When there are 7 or fewer pieces on the board, we can probe tablebases
for the theoretically correct outcome and best move.
"""

import os
import chess
import chess.syzygy

# Global tablebase handle
_tablebase = None

def init_tablebase(path: str = None):
    """
    Initialize the Syzygy tablebase.
    
    Args:
        path: Path to the Syzygy tablebase files (.rtbw and .rtbz files).
              If None, we'll check common locations.
    """
    global _tablebase
    
    # Common tablebase locations
    if path is None:
        candidates = [
            "./syzygy",
            "../syzygy",
            os.path.expanduser("~/syzygy"),
            "C:/syzygy",
            "/usr/share/chess/syzygy"
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                path = candidate
                break
    
    if path and os.path.exists(path):
        try:
            _tablebase = chess.syzygy.open_tablebase(path)
            print(f"Syzygy tablebase loaded from: {path}")
            return True
        except Exception as e:
            print(f"Failed to load Syzygy tablebase: {e}")
            _tablebase = None
    else:
        print("No Syzygy tablebase found. Endgame probing disabled.")
        _tablebase = None
    
    return False

def is_tablebase_available() -> bool:
    """Check if tablebase is loaded and available."""
    return _tablebase is not None

def can_probe(board: chess.Board) -> bool:
    """
    Check if we can probe the tablebase for this position.
    
    Tablebases are available for positions with 7 or fewer pieces
    (including kings).
    """
    if not is_tablebase_available():
        return False
    
    piece_count = len(board.piece_map())
    return piece_count <= 7

def probe_wdl(board: chess.Board) -> int | None:
    """
    Probe the tablebase for Win/Draw/Loss result.
    
    Returns:
        2 = Win
        1 = Cursed Win (win but 50-move rule draw)
        0 = Draw
        -1 = Blessed Loss (losing but 50-move rule draw)
        -2 = Loss
        None = Cannot probe
    """
    if not can_probe(board):
        return None
    
    try:
        return _tablebase.probe_wdl(board)
    except chess.syzygy.MissingTableError:
        return None

def probe_dtz(board: chess.Board) -> int | None:
    """
    Probe the tablebase for Distance To Zeroing (checkmate/capture/pawn move).
    
    Returns:
        Positive = Win in N plies
        0 = Draw
        Negative = Loss in N plies
        None = Cannot probe
    """
    if not can_probe(board):
        return None
    
    try:
        return _tablebase.probe_dtz(board)
    except chess.syzygy.MissingTableError:
        return None

def get_tablebase_move(board: chess.Board) -> chess.Move | None:
    """
    Get the best move according to the tablebase.
    
    This finds the move that minimizes DTZ for losing positions or
    maximizes DTZ for winning positions (to delay the loss or
    expedite the win).
    
    Returns:
        The best move according to the tablebase, or None if no probe is available.
    """
    if not can_probe(board):
        return None
    
    best_move = None
    best_dtz = None
    
    for move in board.legal_moves:
        board.push(move)
        
        try:
            dtz = _tablebase.probe_dtz(board)
            
            # DTZ is from opponent's perspective after we move
            # So we negate it to get our perspective
            dtz = -dtz
            
            if best_dtz is None:
                best_dtz = dtz
                best_move = move
            else:
                # For winning positions (dtz > 0), prefer larger DTZ after move
                # For losing positions (dtz < 0), prefer smaller absolute DTZ
                # For draws (dtz == 0), prefer draws
                if dtz > 0 and (best_dtz <= 0 or dtz < best_dtz):
                    # Winning move found or faster win
                    best_dtz = dtz
                    best_move = move
                elif dtz == 0 and best_dtz < 0:
                    # Drawing move better than losing
                    best_dtz = dtz
                    best_move = move
                elif dtz < 0 and best_dtz < 0 and dtz > best_dtz:
                    # Slower loss is better
                    best_dtz = dtz
                    best_move = move
                    
        except chess.syzygy.MissingTableError:
            pass
        finally:
            board.pop()
    
    return best_move

def get_tablebase_wdl_probs(board: chess.Board) -> list | None:
    """
    Convert tablebase WDL result to probability format [P(Win), P(Draw), P(Loss)].
    
    Returns:
        [1.0, 0.0, 0.0] for Win
        [0.0, 1.0, 0.0] for Draw
        [0.0, 0.0, 1.0] for Loss
        None if cannot probe
    """
    wdl = probe_wdl(board)
    
    if wdl is None:
        return None
    
    if wdl >= 1:  # Win or Cursed Win
        return [1.0, 0.0, 0.0]
    elif wdl == 0:  # Draw
        return [0.0, 1.0, 0.0]
    else:  # Loss or Blessed Loss
        return [0.0, 0.0, 1.0]
