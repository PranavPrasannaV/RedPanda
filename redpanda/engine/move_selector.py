
"""
Move Selector - MCTS-primary move selection (ChessMamba v3).

Pipeline (replaces the v2 weighted 3-signal blend):

    Phase 0: Syzygy tablebase     -> perfect play with <=7 pieces
    Phase A: Batched neural MCTS  -> adversarially-explored visit distribution
               - Eval Mamba supplies policy prior + WDL value + per-move Q-init
               - uncertainty head sets the simulation budget (adaptive)
               - Search Mamba (optional) refines deep branches
    Phase B: Geometric tie-break  -> when the top MCTS moves are within ~10%
               visits, pick the one pointing furthest along the phase
               advantage vector ("strategic preference")

No alpha-beta. The whole engine is policy + value + learned search + MCTS.
"""

import torch
import chess
import random
from typing import List, Optional, Dict

from encoding import encoder
from batched_mcts import BatchedMCTS


class MoveSelector:
    def __init__(self, eval_model, search_model=None, navigator=None,
                 backend: str = "mcts", num_simulations: int = 800,
                 c_puct: float = 1.5, contempt: float = 0.0, batch_size: int = 16,
                 adaptive_sims: bool = True, use_tablebase: bool = True,
                 tie_threshold: float = 0.10, add_root_noise: bool = False,
                 mars_kwargs: dict = None,
                 # legacy kwargs accepted for backward-compat (ignored):
                 policy_weight=None, geo_weight=None, search_weight=None,
                 top_k=None, search_depth=None, deep_search_depth=None):
        self.eval_model = eval_model
        self.search_model = search_model
        self.navigator = navigator
        self.device = next(eval_model.parameters()).device
        self.use_tablebase = use_tablebase
        # MARS needs the value-equivalent Search Mamba; fall back to MCTS without it.
        self.backend = "mcts" if (backend == "mars" and search_model is None) else backend

        if self.backend == "mars":
            from mars_search import MARS
            mk = dict(navigator=navigator, use_tablebase=use_tablebase, contempt=contempt,
                      adaptive_sims=adaptive_sims, tie_threshold=tie_threshold,
                      add_root_noise=add_root_noise)
            mk.update(mars_kwargs or {})          # explicit mars_kwargs win
            self.mars = MARS(eval_model, search_model, **mk)
            self.mcts = None
        else:
            self.mars = None
            self.mcts = BatchedMCTS(
                eval_model, num_simulations=num_simulations, c_puct=c_puct,
                contempt=contempt, batch_size=batch_size, adaptive_sims=adaptive_sims,
                use_tablebase=use_tablebase, navigator=navigator,
                tie_threshold=tie_threshold, add_root_noise=add_root_noise)

    # ── Move selection ──

    def select_move(self, board: chess.Board, verbose: bool = False,
                    temperature: float = 0.0) -> Optional[chess.Move]:
        legal = list(board.legal_moves)
        if not legal:
            return None
        if len(legal) == 1:
            return legal[0]

        # Phase 0: tablebase shortcut for exact endgames.
        if self.use_tablebase:
            tb = self._tablebase_move(board)
            if tb is not None:
                if verbose:
                    print(f"[tablebase] {tb.uci()}")
                return tb

        # Phase A: search (MARS primary, MCTS legacy).
        if self.backend == "mars":
            if verbose:
                res = self.mars.run_search(board)
                print(f"[mars] {res['best']} root_value={res['root_value']:.3f}")
                return res["best"]
            return self.mars.search(board, temperature=temperature)

        root = self.mcts.run_search(board)
        if not root.children:
            return random.choice(legal)
        if verbose:
            self._print_analysis(board, root)
        return self._pick_from_root(board, root, temperature)

    def _pick_from_root(self, board, root, temperature):
        ranked = sorted(root.children.items(),
                        key=lambda kv: kv[1].visit_count, reverse=True)
        if temperature > 1e-3 and len(ranked) > 1:
            import numpy as np
            moves = [m for m, _ in ranked]
            visits = np.array([c.visit_count for _, c in ranked], dtype=np.float64)
            probs = visits ** (1.0 / temperature)
            probs /= probs.sum()
            return moves[int(np.random.choice(len(moves), p=probs))]

        top_move, top_child = ranked[0]
        nav = self.navigator
        if nav is not None and getattr(nav, "advantage_vectors", None):
            top_v = top_child.visit_count
            close = [m for m, c in ranked
                     if top_v > 0 and c.visit_count >= (1 - self.mcts.tie_threshold) * top_v]
            if len(close) > 1:
                geo = nav.score_moves(board, self.eval_model, close)
                return close[int(torch.as_tensor(geo).argmax().item())]
        return top_move

    # ── Analysis (for UI / UCI info) ──

    def get_analysis(self, board: chess.Board, top_n: int = 5) -> List[Dict]:
        if board.is_game_over():
            return []
        if self.backend == "mars":
            res = self.mars.run_search(board)
            ranked = sorted(res["edges"].items(), key=lambda kv: kv[1][0], reverse=True)
            return [{"move": m.uci(), "visits": n, "value": q, "rank": i + 1}
                    for i, (m, (n, q)) in enumerate(ranked[:top_n])]
        root = self.mcts.run_search(board)
        stats = self.mcts.get_move_stats(root)[:top_n]
        for i, s in enumerate(stats):
            s["rank"] = i + 1
        return stats

    def root_evaluation(self, board: chess.Board) -> Dict:
        """WDL / uncertainty / value for the current position."""
        with torch.no_grad():
            enc = encoder.encode_board(board)
            x = torch.tensor([enc], dtype=torch.long, device=self.device)
            out = self.eval_model.forward_full(x)
        return {
            "wdl": out["wdl"][0].cpu().tolist(),
            "uncertainty": out["uncertainty"][0, 0].item(),
            "strategy": out["strategy"][0].cpu().tolist(),
        }

    # ── Internals ──

    def _tablebase_move(self, board: chess.Board):
        try:
            from tablebase import can_probe, get_tablebase_move
        except Exception:
            return None
        if not can_probe(board):
            return None
        return get_tablebase_move(board)

    def _print_analysis(self, board, root):
        stats = self.mcts.get_move_stats(root)
        ev = self.root_evaluation(board)
        wdl = ev["wdl"]
        print(f"\n{'='*56}")
        print(f"WDL [{wdl[0]:.2f} {wdl[1]:.2f} {wdl[2]:.2f}] "
              f"unc {ev['uncertainty']:.3f} | root visits {root.visit_count}")
        print(f"{'Move':<8}{'Visits':>8}{'Q':>9}{'Prior':>9}")
        print("-" * 40)
        for s in stats[:10]:
            print(f"{s['move']:<8}{s['visits']:>8}{s['value']:>9.3f}{s['prior']:>9.3f}")
