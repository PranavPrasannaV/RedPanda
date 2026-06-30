
"""
Batched neural MCTS - the PRIMARY search for ChessMamba v3.

Upgrades over v2:
    - PUCT selection with **action-value initialisation** (KataGo-style): a new
      child starts with the eval network's Q-estimate for that move instead of 0,
      killing the cold-start problem.
    - **Dirichlet noise** at the root (AlphaZero) for exploration.
    - **Progressive widening**: start with the top moves, widen as visits grow -
      avoids copying a board for every one of ~35 legal moves at every node.
    - **First-Play-Urgency (FPU)** fallback for unvisited children.
    - **Uncertainty-adaptive simulation count** driven by the uncertainty head.
    - **Syzygy** probing at leaves -> theoretically perfect ≤7-piece endgames.
    - **Virtual loss** so a batch of leaves is genuinely diverse.
    - **Geometric tie-break** when the top moves are within a visit threshold.

Value convention (important): every node's running value is stored from ITS OWN
side-to-move perspective. When a parent scores a child it therefore uses
`-child.value()` (the v2 code maximised `+child.value()`, i.e. it preferred moves
that were good for the OPPONENT - a silent strength-destroying bug, fixed here).
"""

import math
import numpy as np
import torch
import chess
from typing import List, Tuple, Optional, Dict

from encoding import encoder, ACTION_SPACE


# ─── Node ────────────────────────────────────────────────────────────────────

class BatchedMCTSNode:
    __slots__ = (
        "game_state", "parent", "action_taken", "children", "unexpanded",
        "visit_count", "value_sum", "virtual_loss", "prior", "init_q",
        "wdl", "is_terminal", "expanded",
    )

    def __init__(self, game_state: chess.Board, parent=None, action_taken=None,
                 prior=0.0, init_q=0.0):
        self.game_state = game_state
        self.parent = parent
        self.action_taken = action_taken
        self.children: Dict[chess.Move, "BatchedMCTSNode"] = {}
        # Lazily-instantiated children: list of (move, prior, init_q) sorted desc.
        self.unexpanded: List[Tuple[chess.Move, float, float]] = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.virtual_loss = 0
        self.prior = prior
        self.init_q = init_q          # Q from the PARENT's perspective (AV head)
        self.wdl = None
        self.is_terminal = False
        self.expanded = False

    def value(self) -> float:
        """Mean value from THIS node's side-to-move perspective (virtual-loss aware)."""
        denom = self.visit_count + self.virtual_loss
        if denom == 0:
            return 0.0
        return (self.value_sum - self.virtual_loss) / denom

    def expand(self, priors: Dict[chess.Move, float],
               init_qs: Dict[chess.Move, float]):
        """Store sorted (move, prior, init_q) for lazy progressive widening."""
        items = [(m, priors[m], init_qs.get(m, 0.0)) for m in priors]
        items.sort(key=lambda x: x[1], reverse=True)
        self.unexpanded = items
        self.expanded = True

    def _widen(self, init_width: int, pw_c: float, pw_alpha: float):
        """Instantiate child nodes up to the progressive-widening budget."""
        if not self.unexpanded:
            return
        allowed = max(init_width, int(pw_c * (self.visit_count ** pw_alpha)))
        allowed = min(allowed, len(self.children) + len(self.unexpanded))
        while len(self.children) < allowed and self.unexpanded:
            move, prior, init_q = self.unexpanded.pop(0)
            child_board = self.game_state.copy(stack=False)
            child_board.push(move)
            self.children[move] = BatchedMCTSNode(
                child_board, parent=self, action_taken=move,
                prior=prior, init_q=init_q,
            )

    def select_child(self, c_puct, fpu_reduction, init_width, pw_c, pw_alpha):
        self._widen(init_width, pw_c, pw_alpha)
        parent_visits = self.visit_count + self.virtual_loss
        sqrt_parent = math.sqrt(max(1, parent_visits))
        parent_v = self.value()

        best_score, best_child = -float("inf"), None
        for child in self.children.values():
            cv = child.visit_count + child.virtual_loss
            if cv > 0:
                q = -child.value()                 # child stores opp-perspective value
            elif child.init_q != 0.0:
                q = child.init_q                   # AV-head estimate (parent perspective)
            else:
                q = parent_v - fpu_reduction       # FPU fallback
            u = c_puct * child.prior * sqrt_parent / (1 + cv)
            score = q + u
            if score > best_score:
                best_score, best_child = score, child
        return best_child

    def is_leaf(self) -> bool:
        # A node is a search leaf until its policy/priors have been set.
        return not self.expanded


# ─── Search ──────────────────────────────────────────────────────────────────

class BatchedMCTS:
    def __init__(
        self,
        model,
        num_simulations: int = 800,
        c_puct: float = 1.5,
        contempt: float = 0.0,
        batch_size: int = 16,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        fpu_reduction: float = 0.2,
        init_width: int = 10,
        pw_c: float = 2.0,
        pw_alpha: float = 0.5,
        use_tablebase: bool = True,
        adaptive_sims: bool = False,
        sim_bounds: Tuple[int, int, int] = (400, 800, 2000),
        unc_bounds: Tuple[float, float] = (0.3, 0.6),
        navigator=None,
        tie_threshold: float = 0.10,
        add_root_noise: bool = True,
    ):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.contempt = contempt
        self.batch_size = batch_size
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.fpu_reduction = fpu_reduction
        self.init_width = init_width
        self.pw_c = pw_c
        self.pw_alpha = pw_alpha
        self.use_tablebase = use_tablebase
        self.adaptive_sims = adaptive_sims
        self.sim_bounds = sim_bounds
        self.unc_bounds = unc_bounds
        self.navigator = navigator
        self.tie_threshold = tie_threshold
        self.add_root_noise = add_root_noise
        self.device = next(model.parameters()).device
        # Instrumentation ONLY (speed_bench / diagnostics): positions sent to
        # the eval net. Never read by the search — zero behavior impact.
        self.eval_count = 0

    # ── Neural evaluation ──

    @torch.no_grad()
    def _evaluate_batch(self, boards: List[chess.Board]):
        """Return list of dicts {policy, wdl, action_value, uncertainty}."""
        self.eval_count += len(boards)
        encodings = [encoder.encode_board(b) for b in boards]
        max_len = max(len(e) for e in encodings)
        padded = np.zeros((len(boards), max_len), dtype=np.int64)
        for i, e in enumerate(encodings):
            padded[i, : len(e)] = e
        x = torch.from_numpy(padded).to(self.device)

        out = self.model(x, return_dict=True)
        policy = torch.softmax(out["policy"], dim=-1).cpu().numpy()
        wdl = out["wdl"].cpu().numpy()
        av = out["action_value"].cpu().numpy() if "action_value" in out else None
        unc = out["uncertainty"].squeeze(-1).cpu().numpy() if "uncertainty" in out else None
        results = []
        for i in range(len(boards)):
            results.append({
                "policy": policy[i],
                "wdl": wdl[i],
                "action_value": av[i] if av is not None else None,
                "uncertainty": float(unc[i]) if unc is not None else 0.5,
            })
        return results

    def _value_from_wdl(self, wdl) -> float:
        v = float(wdl[0] - wdl[2]) + self.contempt * float(wdl[1])
        return max(-1.0, min(1.0, v))

    def _tablebase_eval(self, board: chess.Board):
        """Return (value, wdl) from Syzygy or None if unavailable."""
        if not self.use_tablebase:
            return None
        try:
            from tablebase import can_probe, get_tablebase_wdl_probs
        except Exception:
            return None
        if not can_probe(board):
            return None
        wdl = get_tablebase_wdl_probs(board)   # side-to-move [W,D,L]
        if wdl is None:
            return None
        return self._value_from_wdl(wdl), np.array(wdl, dtype=np.float32)

    # ── Expansion ──

    def _priors_and_q(self, board, policy_probs, action_values):
        legal = list(board.legal_moves)
        priors, init_qs = {}, {}
        total = 0.0
        for mv in legal:
            mid = encoder.encode_move(mv)
            p = float(policy_probs[mid]) if (mid is not None and mid < len(policy_probs)) else 1e-4
            priors[mv] = p
            total += p
            if action_values is not None and mid is not None and mid < len(action_values):
                init_qs[mv] = float(action_values[mid])
        if total > 0:
            for mv in priors:
                priors[mv] /= total
        else:
            for mv in legal:
                priors[mv] = 1.0 / max(1, len(legal))
        return priors, init_qs

    def _expand_node(self, node: BatchedMCTSNode, evaluation):
        board = node.game_state
        priors, init_qs = self._priors_and_q(
            board, evaluation["policy"], evaluation.get("action_value")
        )
        node.expand(priors, init_qs)

    def _add_dirichlet(self, node: BatchedMCTSNode):
        if not node.unexpanded:
            return
        n = len(node.unexpanded)
        noise = np.random.dirichlet([self.dirichlet_alpha] * n)
        eps = self.dirichlet_eps
        node.unexpanded = [
            (m, (1 - eps) * p + eps * float(z), q)
            for (m, p, q), z in zip(node.unexpanded, noise)
        ]
        node.unexpanded.sort(key=lambda x: x[1], reverse=True)

    # ── Selection / backup ──

    def _select_leaf(self, root):
        path = [root]
        node = root
        while not node.is_leaf():
            child = node.select_child(
                self.c_puct, self.fpu_reduction,
                self.init_width, self.pw_c, self.pw_alpha,
            )
            if child is None:
                break
            node = child
            path.append(node)
        for n in path:
            n.virtual_loss += 1
        return node, path

    @staticmethod
    def _backup(path, value):
        for node in reversed(path):
            node.virtual_loss -= 1
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _terminal_value(self, board: chess.Board) -> float:
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            return 0.0
        if outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == board.turn else -1.0

    # ── Public search ──

    @torch.no_grad()
    def run_search(self, board: chess.Board) -> BatchedMCTSNode:
        self.eval_count = 0
        root = BatchedMCTSNode(board.copy(stack=False))

        # Root evaluation (+ adaptive sim count, + Dirichlet noise).
        root_eval = self._evaluate_batch([root.game_state])[0]
        root.wdl = root_eval["wdl"]
        self._expand_node(root, root_eval)
        if self.add_root_noise:
            self._add_dirichlet(root)

        num_sims = self.num_simulations
        if self.adaptive_sims:
            u = root_eval["uncertainty"]
            lo, mid, hi = self.sim_bounds
            t_lo, t_hi = self.unc_bounds
            num_sims = lo if u < t_lo else (mid if u < t_hi else hi)

        done = 0
        while done < num_sims:
            leaves, paths, pending = [], [], []
            target = min(self.batch_size, num_sims - done)
            while len(pending) < target:
                leaf, path = self._select_leaf(root)

                # Terminal?
                if leaf.game_state.is_game_over(claim_draw=True):
                    leaf.is_terminal = True
                    self._backup(path, self._terminal_value(leaf.game_state))
                    done += 1
                    if done >= num_sims:
                        break
                    continue

                # Syzygy exact value (still expand with uniform-ish priors).
                tb = self._tablebase_eval(leaf.game_state)
                if tb is not None:
                    tb_val, tb_wdl = tb
                    leaf.wdl = tb_wdl
                    if not leaf.expanded:
                        uniform = np.full(ACTION_SPACE, 0.0, dtype=np.float32)
                        for mv in leaf.game_state.legal_moves:
                            mid = encoder.encode_move(mv)
                            if mid is not None and mid < ACTION_SPACE:
                                uniform[mid] = 1.0
                        self._expand_node(leaf, {"policy": uniform, "action_value": None})
                    self._backup(path, tb_val)
                    done += 1
                    if done >= num_sims:
                        break
                    continue

                leaves.append(leaf)
                paths.append(path)
                pending.append(leaf)

            if not leaves:
                continue

            evals = self._evaluate_batch([lf.game_state for lf in leaves])
            for leaf, path, ev in zip(leaves, paths, evals):
                leaf.wdl = ev["wdl"]
                self._expand_node(leaf, ev)
                self._backup(path, self._value_from_wdl(ev["wdl"]))
                done += 1

        return root

    # ── Outputs ──

    def get_move_stats(self, root) -> List[dict]:
        stats = []
        for move, child in root.children.items():
            stats.append({
                "move": move.uci(),
                "visits": child.visit_count,
                "value": -child.value(),     # from root's perspective
                "prior": child.prior,
                "wdl": child.wdl.tolist() if child.wdl is not None else None,
            })
        return sorted(stats, key=lambda x: x["visits"], reverse=True)

    def get_policy_target(self, root, temperature: float = 1.0) -> np.ndarray:
        """Dense visit-count distribution over the action space (self-play π)."""
        pi = np.zeros(ACTION_SPACE, dtype=np.float32)
        moves, visits = [], []
        for move, child in root.children.items():
            mid = encoder.encode_move(move)
            if mid is None or mid >= ACTION_SPACE:
                continue
            moves.append(mid)
            visits.append(child.visit_count)
        if not moves:
            return pi
        visits = np.array(visits, dtype=np.float64)
        if temperature <= 1e-3:
            pi[moves[int(visits.argmax())]] = 1.0
        else:
            v = visits ** (1.0 / temperature)
            s = v.sum()
            if s <= 0:
                v = np.ones_like(v)
                s = v.sum()
            for mid, p in zip(moves, v / s):
                pi[mid] = p
        return pi

    def root_value(self, root) -> float:
        """Value of the root from its own side-to-move perspective."""
        return root.value()

    def search(self, board: chess.Board, temperature: float = 0.0,
               eval_model=None) -> Optional[chess.Move]:
        """Run search and return a move (geometric tie-break when close)."""
        root = self.run_search(board)
        if not root.children:
            return None

        ranked = sorted(root.children.items(),
                        key=lambda kv: kv[1].visit_count, reverse=True)

        if temperature > 1e-3 and len(ranked) > 1:
            moves = [m for m, _ in ranked]
            visits = np.array([c.visit_count for _, c in ranked], dtype=np.float64)
            probs = visits ** (1.0 / temperature)
            probs /= probs.sum()
            return moves[int(np.random.choice(len(moves), p=probs))]

        # Geometric tie-break among near-equal top moves.
        top_move, top_child = ranked[0]
        nav = self.navigator
        if nav is not None and getattr(nav, "advantage_vectors", None) and eval_model is not None:
            top_v = top_child.visit_count
            close = [m for m, c in ranked
                     if top_v > 0 and c.visit_count >= (1 - self.tie_threshold) * top_v]
            if len(close) > 1:
                geo = nav.score_moves(board, eval_model, close)
                return close[int(torch.as_tensor(geo).argmax().item())]
        return top_move
