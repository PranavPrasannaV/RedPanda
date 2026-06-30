import chess
import chess.engine
import numpy as np
import torch
import math

from batched_mcts import BatchedMCTS
from mars_search import MARS
from encoding import encoder, ACTION_SPACE

class StockfishOracle:
    """Wrapper to interact with Stockfish as our neural-net replacement."""
    def __init__(self, sf_path, depth=1, multipv=5):
        self.engine = chess.engine.SimpleEngine.popen_uci(sf_path)
        self.depth = depth
        self.multipv = multipv
        self.calls = 0          # count evaluations for a fair budget comparison

    def evaluate(self, board):
        """Returns (value in [-1, 1], policy vector [8192])."""
        self.calls += 1
        info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth), multipv=self.multipv)
        
        # Calculate value from Stockfish's perspective
        best_info = info[0] if isinstance(info, list) else info
        score = best_info["score"].relative.score(mate_score=10000)
        if score is None:
            score = 0
        v = max(-1.0, min(1.0, score / 1000.0)) # scale to roughly -1 to 1

        # Create policy based on top moves
        policy = np.zeros(ACTION_SPACE, dtype=np.float32)
        if isinstance(info, list):
            for i, p_info in enumerate(info):
                if "pv" in p_info and p_info["pv"]:
                    move = p_info["pv"][0]
                    mid = encoder.encode_move(move)
                    if mid is not None and mid < ACTION_SPACE:
                        # simple rank-based weight
                        policy[mid] = 1.0 / (i + 1)
        else:
            if "pv" in info and info["pv"]:
                mid = encoder.encode_move(info["pv"][0])
                if mid is not None and mid < ACTION_SPACE:
                    policy[mid] = 1.0
        
        # Fallback to uniform if no PV
        if policy.sum() == 0:
            for m in board.legal_moves:
                mid = encoder.encode_move(m)
                if mid is not None and mid < ACTION_SPACE:
                    policy[mid] = 1.0
        
        policy /= max(1e-9, policy.sum())
        return v, policy

    def close(self):
        self.engine.quit()

class OracleMCTS(BatchedMCTS):
    """MCTS that queries Stockfish instead of Eval Mamba."""
    def __init__(self, oracle, **kwargs):
        # Pass dummy model
        super().__init__(model=torch.nn.Linear(1, 1), **kwargs)
        self.oracle = oracle
        self.device = torch.device("cpu") # ignore cuda for oracle

    @torch.no_grad()
    def _evaluate_batch(self, boards):
        results = []
        for b in boards:
            v, policy = self.oracle.evaluate(b)
            # simulate wdl: [Win, Draw, Loss] from side-to-move perspective
            w = max(0, v)
            l = max(0, -v)
            d = 1.0 - w - l
            wdl = np.array([w, d, l], dtype=np.float32)
            
            results.append({
                "policy": policy,
                "wdl": wdl,
                "action_value": None,
                "uncertainty": 0.5 # constant low uncertainty
            })
        return results

class DummySearchModel:
    def prime_board(self, enc):
        dummy = torch.zeros(1)
        return dummy, [(dummy, dummy)]
    def eval_step(self, cache, mid):
        return 0.0, None, cache
    def continuation_head(self, hidden):
        # Required by MARS's re-anchor branch; OracleMARS re-queries the oracle
        # for move ordering anyway, so a uniform placeholder is fine.
        return torch.zeros(1, ACTION_SPACE)

class OracleMARS(MARS):
    """
    MARS that queries Stockfish instead of the Search Mamba — a faithful Stage-0
    test of the SEARCH ALGORITHM (not network speed). With k_anchor=1 every node
    is evaluated by the oracle (Stockfish), quiescence/adaptive are off, and the
    only neural component (eval_step) is never used. This isolates whether MARS's
    Gumbel + sequential-halving + recurrent-PV structure beats MCTS's PUCT at an
    equal oracle-call budget.
    """
    def __init__(self, oracle, **kwargs):
        # Force the oracle to evaluate EVERY node and disable the parts that need
        # a trained Search Mamba, so this is a clean algorithm-only comparison.
        kwargs.setdefault("k_anchor", 1)
        kwargs.setdefault("quiescence", False)
        kwargs.setdefault("adaptive_sims", False)
        super().__init__(eval_model=torch.nn.Linear(1,1), search_model=DummySearchModel(), **kwargs)
        self.oracle = oracle
        self.device = torch.device("cpu")

    @torch.no_grad()
    def _eval_root(self, board):
        enc = encoder.encode_board(board)
        v, policy = self.oracle.evaluate(board)
        w, l = max(0, v), max(0, -v)
        wdl = np.array([w, 1.0 - w - l, l], dtype=np.float32)
        return enc, policy, wdl, None, 0.5

    @torch.no_grad()
    def _batch_values(self, encodings):
        # We don't have boards here in MARS originally, but OracleMARS bypasses this anyway
        return [0.0] * len(encodings)

    def _anchor_value(self, board):
        v, _ = self.oracle.evaluate(board)
        return v

    def _priors_to_move(self, board, cont_logits):
        # Sample stochastically from the oracle policy instead of argmax
        # This mimics the Search Mamba's stochastic continuation head!
        _, policy = self.oracle.evaluate(board)
        
        legal = list(board.legal_moves)
        if not legal:
            return None
            
        p = np.array([policy[encoder.encode_move(m)] if encoder.encode_move(m) is not None else 0.0 for m in legal], dtype=np.float64)
        
        if p.sum() == 0:
            p = np.ones(len(legal), dtype=np.float64)
            
        p /= p.sum()
        
        idx = np.random.choice(len(legal), p=p)
        return legal[idx]
