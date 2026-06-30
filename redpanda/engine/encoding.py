
import chess
import numpy as np


# ── Global action space ──
# The policy / action-value heads emit one logit per move-token id.
# The encoder's move vocabulary tops out at id 8125 (see _build_vocab), so the
# action space MUST be >= 8126 or promotion moves get silently truncated.
# 8192 = next multiple of 256, covers every legal move id with headroom.
ACTION_SPACE = 8192

# Integer ids for the three game phases (used for stratification + geometric
# advantage-vector bucketing). Kept consistent with the [OPENING]/[MIDDLEGAME]/
# [ENDGAME] string tokens emitted into the board encoding.
PHASE_OPENING = 0
PHASE_MIDDLEGAME = 1
PHASE_ENDGAME = 2
PHASE_NAMES = ["opening", "middlegame", "endgame"]

# Home squares of the minor pieces — used to gauge development (opening vs
# middlegame) from a static board, since eval-DB FENs carry no move counters.
_MINOR_HOME = {
    (chess.WHITE, chess.KNIGHT): frozenset({chess.B1, chess.G1}),
    (chess.WHITE, chess.BISHOP): frozenset({chess.C1, chess.F1}),
    (chess.BLACK, chess.KNIGHT): frozenset({chess.B8, chess.G8}),
    (chess.BLACK, chess.BISHOP): frozenset({chess.C8, chess.F8}),
}
_EMPTY = frozenset()


class ChessEncoder:
    """
    Enhanced board encoder for ChessMamba v2.
    
    Encodes a chess position as a sequence of tokens:
        [START] [move_history...] [BOARD_START] [64 squares] [metadata] [phase]
    
    v2 additions:
        - En passant square
        - Halfmove clock (50-move rule awareness)
        - Repetition count (threefold repetition awareness)
        - Material balance (bucketed)
    """

    def __init__(self):
        self.move_to_id = {}
        self.id_to_move = {}
        self.piece_to_id = {}
        self._build_vocab()

    def _build_vocab(self):
        next_id = 0

        def _add(name):
            nonlocal next_id
            self.move_to_id[name] = next_id
            next_id += 1
            return next_id - 1

        # ── 1. Special Tokens ──
        _add("[PAD]")
        _add("[START]")
        _add("[BOARD_START]")

        # ── 2. Side-to-move / Castling ──
        _add("W_TURN")
        _add("B_TURN")
        _add("W_OO")
        _add("W_OOO")
        _add("B_OO")
        _add("B_OOO")

        # ── 3. Game Phase ──
        _add("[OPENING]")
        _add("[MIDDLEGAME]")
        _add("[ENDGAME]")

        # ── 4. En Passant (16 possible squares + NO_EP) ──
        for file_char in "abcdefgh":
            for rank_char in ["3", "6"]:
                _add(f"EP_{file_char}{rank_char}")
        _add("NO_EP")

        # ── 5. Halfmove Clock buckets (50-move rule) ──
        for i in range(6):  # 0-9, 10-19, 20-29, 30-39, 40-49, 50+
            _add(f"HMC_{i}")

        # ── 6. Repetition count ──
        for i in range(3):  # 0, 1, 2+
            _add(f"REP_{i}")

        # ── 7. Material balance buckets (-5 to +5 pawns) ──
        for i in range(-5, 6):
            _add(f"MAT_{i}")

        # ── 8. Piece Tokens (board state) ──
        pieces = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k", "."]
        for p in pieces:
            self.piece_to_id[p] = next_id
            self.move_to_id[f"PIECE_{p}"] = next_id
            next_id += 1

        # ── 9. Move Tokens (policy head action space) ──
        squares = range(64)
        promotion_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

        for from_sq in squares:
            for to_sq in squares:
                if from_sq == to_sq:
                    continue

                move = chess.Move(from_sq, to_sq)
                uci = move.uci()

                if uci not in self.move_to_id:
                    self.move_to_id[uci] = next_id
                    self.id_to_move[next_id] = uci
                    next_id += 1

                to_rank = chess.square_rank(to_sq)
                if to_rank == 0 or to_rank == 7:
                    for piece_type in promotion_pieces:
                        prom_move = chess.Move(from_sq, to_sq, promotion=piece_type)
                        uci_p = prom_move.uci()
                        if uci_p not in self.move_to_id:
                            self.move_to_id[uci_p] = next_id
                            self.id_to_move[next_id] = uci_p
                            next_id += 1

        print(f"Vocab size: {len(self.move_to_id)}")

    # ── Move encoding / decoding ──

    def encode_move(self, move: chess.Move) -> int:
        if isinstance(move, str):
            return self.move_to_id.get(move)
        return self.move_to_id.get(move.uci())

    def decode_move(self, idx: int) -> chess.Move:
        uci = self.id_to_move.get(idx)
        if uci:
            try:
                return chess.Move.from_uci(uci)
            except Exception:
                return None
        return None

    # ── Board encoding ──

    def encode_board(self, board: chess.Board):
        """
        Encode full board state as a token sequence.
        
        Structure:
            [START]
            [move_history (up to 32 half-moves)]
            [BOARD_START]
            [64 piece tokens]
            [side_to_move]
            [castling rights]
            [en_passant]
            [halfmove_clock]
            [repetition_count]
            [material_balance]
            [game_phase]
        """
        # 1. Start token
        encoding = [self.move_to_id["[START]"]]

        # 2. Move history (last 32 half-moves)
        for move in board.move_stack[-32:]:
            mid = self.encode_move(move)
            if mid is not None:
                encoding.append(mid)

        # 3. Board snapshot
        encoding.append(self.move_to_id["[BOARD_START]"])
        for i in range(64):
            piece = board.piece_at(i)
            symbol = piece.symbol() if piece else "."
            encoding.append(self.piece_to_id[symbol])

        # 4. Side to move
        encoding.append(
            self.move_to_id["W_TURN" if board.turn == chess.WHITE else "B_TURN"]
        )

        # 5. Castling rights
        if board.has_kingside_castling_rights(chess.WHITE):
            encoding.append(self.move_to_id["W_OO"])
        if board.has_queenside_castling_rights(chess.WHITE):
            encoding.append(self.move_to_id["W_OOO"])
        if board.has_kingside_castling_rights(chess.BLACK):
            encoding.append(self.move_to_id["B_OO"])
        if board.has_queenside_castling_rights(chess.BLACK):
            encoding.append(self.move_to_id["B_OOO"])

        # 6. En passant square (NEW in v2)
        if board.has_legal_en_passant():
            ep_name = chess.square_name(board.ep_square)
            ep_key = f"EP_{ep_name}"
            if ep_key in self.move_to_id:
                encoding.append(self.move_to_id[ep_key])
            else:
                encoding.append(self.move_to_id["NO_EP"])
        else:
            encoding.append(self.move_to_id["NO_EP"])

        # 7. Halfmove clock bucket (NEW in v2)
        hmc_bucket = min(board.halfmove_clock // 10, 5)
        encoding.append(self.move_to_id[f"HMC_{hmc_bucket}"])

        # 8. Repetition count (NEW in v2)
        if board.is_repetition(3):
            rep = 2
        elif board.is_repetition(2):
            rep = 1
        else:
            rep = 0
        encoding.append(self.move_to_id[f"REP_{rep}"])

        # 9. Material balance (NEW in v2)
        mat = self._compute_material_balance(board)
        mat_bucket = max(-5, min(5, mat // 100))  # Buckets of ~1 pawn
        encoding.append(self.move_to_id[f"MAT_{mat_bucket}"])

        # 10. Game phase
        phase = self._detect_game_phase(board)
        encoding.append(self.move_to_id[phase])

        return encoding

    # ── Helper methods ──

    def _compute_material_balance(self, board: chess.Board) -> int:
        """
        Compute material balance in centipawns from White's perspective.
        P=100, N=320, B=330, R=500, Q=900
        """
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
        }
        balance = 0
        for sq, piece in board.piece_map().items():
            if piece.piece_type == chess.KING:
                continue
            val = piece_values.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                balance += val
            else:
                balance -= val
        return balance

    def _detect_game_phase(self, board: chess.Board) -> str:
        """
        Detect game phase based on material and development.
        Returns "[OPENING]", "[MIDDLEGAME]", or "[ENDGAME]".
        """
        pm = board.piece_map()
        piece_count = len(pm)

        # Count non-pawn, non-king pieces
        major_minor = sum(1 for p in pm.values()
                          if p.piece_type not in (chess.PAWN, chess.KING))

        if piece_count <= 12 or major_minor <= 4:
            return "[ENDGAME]"

        # Opening vs middlegame from DEVELOPMENT (eval-DB FENs have no move
        # counters): count minor pieces that have left their home squares.
        # <=2 developed -> still in the opening; otherwise middlegame.
        developed = 0
        for sq, p in pm.items():
            if p.piece_type in (chess.KNIGHT, chess.BISHOP):
                if sq not in _MINOR_HOME.get((p.color, p.piece_type), _EMPTY):
                    developed += 1
        if developed <= 2:
            return "[OPENING]"

        return "[MIDDLEGAME]"

    def vocab_size(self):
        return len(self.move_to_id)

    @property
    def action_space(self):
        """Size of the policy / action-value output space."""
        return ACTION_SPACE

    # ── Phase / stratification helpers ──

    def phase_index(self, board: chess.Board) -> int:
        """Return integer phase id: 0=opening, 1=middlegame, 2=endgame."""
        phase = self._detect_game_phase(board)
        return {
            "[OPENING]": PHASE_OPENING,
            "[MIDDLEGAME]": PHASE_MIDDLEGAME,
            "[ENDGAME]": PHASE_ENDGAME,
        }[phase]

    def encode_board_padded(self, board: chess.Board, max_len: int) -> np.ndarray:
        """Encode a board and right-pad (with [PAD]=0) to a fixed length."""
        ids = self.encode_board(board)
        arr = np.zeros(max_len, dtype=np.int64)
        arr[: min(len(ids), max_len)] = ids[:max_len]
        return arr


encoder = ChessEncoder()
