
"""
MARS - Mamba Adversarial Recurrent Search (ChessMamba v4).

A learned, best-first, Gumbel search that replaces MCTS/alpha-beta. Its edge:
every node is evaluated by an O(1) recurrent `step()` of the value-equivalent
Search Mamba (carrying SSM state down the line, no board re-encoding), and the
heavy Eval Mamba is paid only at the root and at periodic re-anchor points.

Per move:
  1. Eval Mamba at the ROOT -> policy prior, value, per-move action-value, uncertainty
     (uncertainty sets the simulation budget).
  2. Gumbel top-k selects <= m_root candidate root moves.
  3. SEQUENTIAL HALVING allocates the budget across candidates; each "simulation"
     is a stochastic principal-variation ROLLOUT driven by the Search Mamba's
     continuation head (learned, adversarial move ordering), re-anchored to the
     Eval Mamba every k plies, with captures/checks quiescence at the leaf.
  4. MCGS transposition table shares position values across rollouts.
  5. Negamax backup to each candidate's edge; pick the Gumbel-improved best.

Exactness: rules/legality/transitions/terminals/Syzygy come from python-chess
(free, exact). The learned model only makes per-node *evaluation* cheap; value
drift is bounded by re-anchoring. The Search Mamba is causal, so its `step()` is
bit-exact (verified) - the append-only `board + moves` stream is what makes this
valid where it would be invalid for the board-snapshot Eval Mamba.
"""

import math
import numpy as np
import torch
import chess

from encoding import encoder, ACTION_SPACE
from mcgs import TranspositionTable, zobrist_key


def _gumbel(n):
    u = np.random.uniform(1e-9, 1.0, size=n)
    return -np.log(-np.log(u))


# Move-encoding cache: encoder.encode_move does a dict lookup + uci() string
# build every call; the rollout hits the same moves constantly. Pure speed,
# identical values.
_ENC_CACHE = {}


def _enc_move(move):
    key = move.uci()
    v = _ENC_CACHE.get(key)
    if v is None:
        v = encoder.encode_move(move)
        _ENC_CACHE[key] = v
    return v


class MARS:
    def __init__(self, eval_model, search_model, navigator=None,
                 sim_budget=64, m_root=16, k_anchor=4, depth_cap=12,
                 rollout_temp=0.6, q_scale=4.0, use_tablebase=True,
                 quiescence=True, q_max=8, contempt=0.0,
                 adaptive_sims=True, sim_bounds=(32, 64, 128),
                 unc_bounds=(0.3, 0.6), tie_threshold=0.10,
                 add_root_noise=False, dirichlet_alpha=0.3, dirichlet_eps=0.25,
                 batched=False, use_tt=True,
                 policy_guided=True, policy_top_k=8):
        self.eval_model = eval_model
        self.search_model = search_model
        self.navigator = navigator
        self.sim_budget = sim_budget
        self.m_root = m_root
        self.k_anchor = k_anchor
        self.depth_cap = depth_cap
        self.rollout_temp = rollout_temp
        self.q_scale = q_scale
        self.use_tablebase = use_tablebase
        self.quiescence = quiescence
        self.q_max = q_max
        self.contempt = contempt
        self.adaptive_sims = adaptive_sims
        self.sim_bounds = sim_bounds
        self.unc_bounds = unc_bounds
        self.tie_threshold = tie_threshold
        self.add_root_noise = add_root_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        # batched=True routes run_search through the lockstep batched rollout
        # (THE speed fix). use_tt=False disables the MCGS transposition table
        # (cleaner speed A/B — the TT's cross-rollout cutoffs differ slightly
        # between sequential and batched ordering).
        self.batched = batched
        self.use_tt = use_tt
        # policy_guided: skip full legal-move generation by validating the top-k
        # continuation moves with is_legal() (real win with a trained net; with
        # random weights it harmlessly falls back). Correctness-preserving —
        # every returned move is is_legal()-checked.
        self.policy_guided = policy_guided
        self.policy_top_k = policy_top_k
        self.device = next(eval_model.parameters()).device
        # Instrumentation ONLY (speed_bench / diagnostics). Never read by the
        # search itself — zero behavior impact. Reset at each run_search().
        self.stats = {"sm_steps": 0, "anchors": 0, "eval_positions": 0,
                      "rollouts": 0, "rollout_plies": 0}

    # ── Heavy (Eval Mamba) evaluations ──

    @torch.no_grad()
    def _eval_root(self, board):
        enc = encoder.encode_board(board)
        x = torch.tensor([enc], dtype=torch.long, device=self.device)
        out = self.eval_model(x, return_dict=True)
        policy = torch.softmax(out["policy"][0], -1).cpu().numpy()
        wdl = out["wdl"][0].cpu().numpy()
        av = out["action_value"][0].cpu().numpy() if "action_value" in out else None
        unc = float(out["uncertainty"][0, 0]) if "uncertainty" in out else 0.5
        return enc, policy, wdl, av, unc

    def _value_from_wdl(self, wdl):
        v = float(wdl[0] - wdl[2]) + self.contempt * float(wdl[1])
        return max(-1.0, min(1.0, v))

    @torch.no_grad()
    def _batch_values(self, encodings, sub=128):
        """Side-to-move values for a list of board encodings. Chunked over `sub`
        rows so a large batch (e.g. batched-rollout leaves) can't OOM the Eval
        forward; numerically identical (rows are independent)."""
        if not encodings:
            return []
        self.stats["eval_positions"] += len(encodings)
        maxlen = max(len(e) for e in encodings)
        arr = np.zeros((len(encodings), maxlen), dtype=np.int64)
        for i, e in enumerate(encodings):
            arr[i, :len(e)] = e
        out_vals = []
        for s in range(0, len(encodings), sub):
            x = torch.from_numpy(arr[s:s + sub]).to(self.device)
            wdl = self.eval_model(x, return_dict=True)["wdl"].cpu().numpy()
            out_vals.extend(self._value_from_wdl(w) for w in wdl)
        return out_vals

    @torch.no_grad()
    def _prime_batched(self, padded, sub=64):
        """Search-Mamba prime over a (B, L) batch, chunked over `sub` rows.
        The prime falls back to the chunked PyTorch scan (return_state bypasses
        the Triton kernel), which is memory-heavy at large B — chunking bounds
        it. Rows are independent, so this is numerically identical to one call."""
        sm = self.search_model
        if padded.shape[0] <= sub:
            return sm.prime_board(padded)
        hs, layers = [], None
        for s in range(0, padded.shape[0], sub):
            h, c = sm.prime_board(padded[s:s + sub])
            hs.append(h)
            if layers is None:
                layers = [[ch, cd] for (ch, cd) in c]
            else:
                for L, (ch, cd) in enumerate(c):
                    layers[L][0] = torch.cat([layers[L][0], ch], 0)
                    layers[L][1] = torch.cat([layers[L][1], cd], 0)
        return torch.cat(hs, 0), [(a, b) for a, b in layers]

    def _anchor_value(self, board):
        return self._batch_values([encoder.encode_board(board)])[0]

    def _quiescence(self, board, static_value):
        """1-ply captures/checks search: catch hanging pieces (not alpha-beta)."""
        if not self.quiescence:
            return static_value
        forcing = [m for m in board.legal_moves
                   if board.is_capture(m) or board.gives_check(m)]
        if not forcing:
            return static_value
        encs = []
        for m in forcing[:self.q_max]:
            board.push(m)
            encs.append(encoder.encode_board(board))
            board.pop()
        child_vals = self._batch_values(encs)        # opponent-perspective
        best_gain = max(-v for v in child_vals)       # our perspective after the capture
        return max(static_value, best_gain)           # stand-pat vs best forcing

    # ── Exact (python-chess) checks ──

    def _terminal_value(self, board):
        o = board.outcome(claim_draw=True)
        if o is None:
            return None
        if o.winner is None:
            return 0.0
        return 1.0 if o.winner == board.turn else -1.0

    def _cheap_terminal(self, board):
        """DRAW-only terminal checks that DON'T need legal-move generation
        (insufficient material, fifty-move, guarded threefold). Mate/stalemate
        are detected for free inside _select_move (empty legal list). Returns a
        draw value (0.0) or None. The threefold check is gated on
        halfmove_clock >= 8 — a 3-fold is impossible with fewer reversible plies,
        so this skips the costly stack scan on most rollout positions."""
        if board.is_insufficient_material():
            return 0.0
        if board.halfmove_clock >= 100:               # fifty-move rule
            return 0.0
        if board.halfmove_clock >= 8 and board.is_repetition(3):
            return 0.0
        return None

    @torch.no_grad()
    def _tablebase_value(self, board):
        if not self.use_tablebase:
            return None
        try:
            from tablebase import can_probe, get_tablebase_wdl_probs
        except Exception:
            return None
        if not can_probe(board):
            return None
        w = get_tablebase_wdl_probs(board)
        return None if w is None else self._value_from_wdl(w)

    # ── Search Mamba helpers ──

    @staticmethod
    def _clone_cache(cache):
        return [(h.clone(), d.clone()) for (h, d) in cache]

    def _priors_to_move(self, board, cont_logits, legal=None):
        """Sample a legal move from continuation logits (learned move ordering).
        `legal` may be passed in to avoid a redundant legal-move generation."""
        if legal is None:
            legal = list(board.legal_moves)
        if not legal:
            return None
        ids, idxs = [], []
        for i, m in enumerate(legal):
            mid = _enc_move(m)
            if mid is not None and mid < cont_logits.shape[0]:
                ids.append(mid)
                idxs.append(i)
        if not ids:
            return legal[0]
        logits = cont_logits[ids].float().cpu().numpy()
        if self.rollout_temp <= 1e-3:
            return legal[idxs[int(logits.argmax())]]
        logits = logits / self.rollout_temp
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        return legal[idxs[int(np.random.choice(len(ids), p=p))]]

    def _select_move(self, board, cont_logits):
        """Pick the rollout move. THE per-node speed lever: with policy_guided
        on, try the top-k continuation moves and validate each with the cheap
        board.is_legal() — avoiding the expensive full legal-move generation
        entirely when a top-k move is legal (the common case with a TRAINED
        net). Falls back to full enumeration otherwise, which also detects
        mate/stalemate (empty legal list).

        Returns (move, had_legal): move is None ONLY when the position has no
        legal move (checkmate/stalemate). had_legal flags whether full
        enumeration ran (for the caller's bookkeeping)."""
        if self.policy_guided:
            ln = cont_logits.float().cpu().numpy()
            k = min(self.policy_top_k, ln.shape[0])
            top = np.argpartition(ln, -k)[-k:]
            top = top[np.argsort(ln[top])[::-1]]        # highest policy first
            cand = []
            for mid in top:
                mv = encoder.decode_move(int(mid))
                if mv is not None and board.is_legal(mv):
                    if self.rollout_temp <= 1e-3:
                        return mv, False                # greedy: highest legal
                    cand.append((mv, float(ln[mid])))
            if cand:                                    # stochastic over legal top-k
                lg = np.array([c[1] for c in cand]) / self.rollout_temp
                lg -= lg.max(); pr = np.exp(lg); pr /= pr.sum()
                return cand[int(np.random.choice(len(cand), p=pr))][0], False
        # Fallback: full enumeration (always correct; detects mate/stalemate).
        legal = list(board.legal_moves)
        if not legal:
            return None, True
        return self._priors_to_move(board, cont_logits, legal), True

    # ── One stochastic PV rollout ──

    @torch.no_grad()
    def _rollout(self, board, cache, cont_logits, tt):
        """
        Roll out a stochastic principal variation from `board` (a copy we own),
        whose Search-Mamba `cache` is consistent with it and `cont_logits` are the
        priors for the side to move. Returns the value of `board` from ITS
        side-to-move perspective (negamax of the single PV line).
        """
        ply = 0
        leaf = None
        path = []                                       # (key, value) visited this rollout
        self.stats["rollouts"] += 1
        while True:
            # Cheap draw-only terminal (no legal-move gen); mate/stalemate is
            # detected for free by _select_move's fallback below.
            dv = self._cheap_terminal(board)
            if dv is not None:
                leaf = dv; break
            tb = self._tablebase_value(board)
            if tb is not None:
                leaf = tb; break
            if ply > 0 and tt is not None:
                hit = tt.get_value(zobrist_key(board))
                if hit is not None:                    # transposition cutoff (MCGS)
                    leaf = hit; break
            if ply >= self.depth_cap:
                leaf = self._quiescence(board, self._anchor_value(board)); break

            move, _had_legal = self._select_move(board, cont_logits)
            if move is None:                           # no legal move -> mate/stalemate
                leaf = (-1.0 if board.is_check() else 0.0); break
            mid = _enc_move(move)
            board.push(move)
            ply += 1
            self.stats["rollout_plies"] += 1

            if ply % self.k_anchor == 0:               # re-anchor: re-encode truth
                self.stats["anchors"] += 1
                hidden, cache = self.search_model.prime_board(encoder.encode_board(board))
                cont_logits = self.search_model.continuation_head(hidden).squeeze(0)
                val_cur = self._quiescence(board, self._anchor_value(board))
            else:                                      # cheap O(1) step
                self.stats["sm_steps"] += 1
                val_cur, cont_logits, cache = self.search_model.eval_step(cache, mid)
            path.append((zobrist_key(board), val_cur))  # DEFER the TT write

        # Update the transposition table only AFTER the rollout completes, so the
        # cutoff above can only fire on positions a *previous* rollout established
        # (not ones this rollout just created). Writing eagerly mid-rollout made
        # every rollout cut itself off at ply 1 — collapsing MARS to 1-ply search.
        if tt is not None:
            for k, v in path:
                tt.update(k, v)

        return (leaf if ply % 2 == 0 else -leaf)

    # ── Batched rollout (THE speed fix: B rollouts in lockstep) ──────────────
    #
    # The sequential _rollout above runs ONE rollout at a time at batch-1, which
    # starves the GPU (a single-token step is ~50x less efficient per token than
    # a parallel forward — pure launch overhead). _rollout_batch runs B rollouts
    # simultaneously: every Search-Mamba step / anchor prime / Eval value call is
    # batched across all B rollouts into ONE GPU call. Plies still advance in
    # lockstep (the recurrence is inherently sequential WITHIN a rollout), but
    # the B rollouts share each GPU launch. Faithful batched mirror of _rollout;
    # with greedy moves (temp<=0) and tt=None it is bit-equivalent (verified).

    @torch.no_grad()
    def _rollout_batch(self, boards, cache, cont, tt):
        """
        boards: list[chess.Board] we own (already at the post-candidate position).
        cache:  ONE Search-Mamba cache whose tensors are (B, ...) — row i = rollout i.
        cont:   (B, vocab) continuation logits, row i = priors for boards[i] STM.
        Returns: list[float] of root-perspective values (negamax-signed per rollout).
        """
        sm = self.search_model
        B = len(boards)
        device = self.device
        active = [True] * B
        done_val = [0.0] * B          # leaf value (pre-sign)
        done_ply = [0] * B            # plies pushed in that rollout at termination
        paths = [[] for _ in range(B)]
        ply = 0
        self.stats["rollouts"] += B

        def _pad_encs(idxs):
            encs = [encoder.encode_board(boards[i]) for i in idxs]
            L = max(len(e) for e in encs)
            arr = np.zeros((len(encs), L), dtype=np.int64)
            for j, e in enumerate(encs):
                arr[j, :len(e)] = e
            return torch.from_numpy(arr).to(device)

        while True:
            # ── Phase A: per-rollout termination checks (active rollouts) ──
            depth_leaves = []
            for i in range(B):
                if not active[i]:
                    continue
                dv = self._cheap_terminal(boards[i])   # draws only (no legal gen)
                if dv is not None:
                    done_val[i] = dv; done_ply[i] = ply; active[i] = False; continue
                tb = self._tablebase_value(boards[i])
                if tb is not None:
                    done_val[i] = tb; done_ply[i] = ply; active[i] = False; continue
                if ply > 0 and tt is not None:
                    hit = tt.get_value(zobrist_key(boards[i]))
                    if hit is not None:
                        done_val[i] = hit; done_ply[i] = ply; active[i] = False; continue
                if ply >= self.depth_cap:
                    depth_leaves.append(i); active[i] = False
            # batched stand-pat eval for the depth-cap leaves, then quiescence
            if depth_leaves:
                svals = self._batch_values([encoder.encode_board(boards[i])
                                            for i in depth_leaves])
                for j, i in enumerate(depth_leaves):
                    done_val[i] = (self._quiescence(boards[i], svals[j])
                                   if self.quiescence else svals[j])
                    done_ply[i] = ply
            if not any(active):
                break

            # ── Phase B: per-rollout move selection + push (active rollouts) ──
            tokens = [0] * B
            moved = [False] * B
            for i in range(B):
                if not active[i]:
                    continue
                move, _hl = self._select_move(boards[i], cont[i])
                if move is None:                       # no legal move -> mate/stalemate
                    done_val[i] = (-1.0 if boards[i].is_check() else 0.0)
                    done_ply[i] = ply; active[i] = False; continue
                mid = _enc_move(move)
                boards[i].push(move)
                tokens[i] = int(mid) if mid is not None else 0
                moved[i] = True
            ply += 1
            self.stats["rollout_plies"] += sum(moved)
            if not any(moved):
                break

            # ── Phase C: ONE batched Search-Mamba update for all B rows ──
            if ply % self.k_anchor == 0:
                # re-anchor: batched re-encode + prime (search) + value (eval)
                self.stats["anchors"] += sum(moved)
                padded = _pad_encs(list(range(B)))      # all rows (done ones harmless)
                hidden, cache = self._prime_batched(padded)
                cont = sm.continuation_head(hidden)
                avals = self._batch_values([encoder.encode_board(boards[i])
                                            for i in range(B)])
                for i in range(B):
                    if moved[i]:
                        v = (self._quiescence(boards[i], avals[i])
                             if self.quiescence else avals[i])
                        paths[i].append((zobrist_key(boards[i]), v))
            else:
                self.stats["sm_steps"] += sum(moved)
                tok_t = torch.tensor(tokens, dtype=torch.long, device=device)
                hidden, cache = sm.backbone.step(token_id=tok_t, cache=cache)
                vals = torch.tanh(sm.value_head(hidden)).squeeze(-1)
                cont = sm.continuation_head(hidden)
                vlist = vals.detach().cpu().tolist()
                for i in range(B):
                    if moved[i]:
                        paths[i].append((zobrist_key(boards[i]), float(vlist[i])))

        # deferred TT writes (per rollout, after IT completes) — same as sequential
        if tt is not None:
            for i in range(B):
                for k, v in paths[i]:
                    tt.update(k, v)

        return [done_val[i] if done_ply[i] % 2 == 0 else -done_val[i]
                for i in range(B)]

    # ── FASTCHESS rollout (the speed fix: board ops in Numba, not python-chess) ─
    def _select_fc_move(self, fb, buf, n, cont_row):
        """Pick a rollout move from the n fastchess legal moves in `buf` using the
        continuation policy `cont_row`. Mirrors _priors_to_move over fc moves."""
        import fastchess as fc
        toks = np.empty(n, dtype=np.int64)
        idxs = np.empty(n, dtype=np.int64)
        v = 0
        vocab = cont_row.shape[0]
        for i in range(n):
            t = fc.move_token(buf[i])
            if 0 <= t < vocab:
                toks[v] = t; idxs[v] = i; v += 1
        if v == 0:
            return buf[0]
        lg = cont_row[torch.from_numpy(toks[:v]).to(cont_row.device)].float().cpu().numpy()
        if self.rollout_temp <= 1e-3:
            return buf[idxs[int(lg.argmax())]]
        lg = lg / self.rollout_temp; lg -= lg.max()
        p = np.exp(lg); p /= p.sum()
        return buf[idxs[int(np.random.choice(v, p=p))]]

    @torch.no_grad()
    def _rollout_batch_fast(self, fboards, histories, cache, cont, tt):
        """Batched lockstep rollout on FASTCHESS boards. Identical search shape to
        _rollout_batch, but every per-ply board op (legal moves, make, terminal,
        zobrist) runs in Numba. The neural step/anchor/eval are unchanged — they
        get bit-identical input via fc_encode_board. quiescence is stand-pat here
        (fast path); set quiescence=False on both engines for a clean A/B."""
        import fastchess as fc
        sm = self.search_model
        B = len(fboards)
        device = self.device
        active = [True] * B
        done_val = [0.0] * B
        done_ply = [0] * B
        paths = [[] for _ in range(B)]
        buf = np.empty(256, dtype=np.uint32)
        zhist = [{int(fc.zobrist(fboards[i])): 1} for i in range(B)]
        ply = 0
        self.stats["rollouts"] += B

        def _enc(i):
            zk = int(fc.zobrist(fboards[i]))
            rep = min(zhist[i].get(zk, 1) - 1, 2)
            return fc.fc_encode_board(fboards[i], histories[i], rep, encoder)

        while True:
            depth_leaves = []
            for i in range(B):
                if not active[i]:
                    continue
                a = fboards[i]
                if fc.is_insufficient(a) or int(a[fc.HALF]) >= 100:
                    done_val[i] = 0.0; done_ply[i] = ply; active[i] = False; continue
                zk = int(fc.zobrist(a))
                if zhist[i].get(zk, 0) >= 3:
                    done_val[i] = 0.0; done_ply[i] = ply; active[i] = False; continue
                if ply > 0 and tt is not None:
                    hit = tt.get_value(zk)
                    if hit is not None:
                        done_val[i] = hit; done_ply[i] = ply; active[i] = False; continue
                if ply >= self.depth_cap:
                    depth_leaves.append(i); active[i] = False
            if depth_leaves:
                svals = self._batch_values([_enc(i) for i in depth_leaves])
                for j, i in enumerate(depth_leaves):
                    done_val[i] = svals[j]; done_ply[i] = ply
            if not any(active):
                break
            tokens = [0] * B
            moved = [False] * B
            for i in range(B):
                if not active[i]:
                    continue
                a = fboards[i]
                n = fc.gen_legal(a, buf)
                if n == 0:                              # mate / stalemate
                    done_val[i] = -1.0 if fc._in_check(a, a[fc.SIDE] == 0) else 0.0
                    done_ply[i] = ply; active[i] = False; continue
                mv = self._select_fc_move(a, buf, n, cont[i])
                tok = fc.move_token(mv)
                fboards[i] = fc.make(a, mv)
                histories[i] = histories[i] + [tok]
                tokens[i] = tok if tok >= 0 else 0
                moved[i] = True
                zk = int(fc.zobrist(fboards[i]))
                zhist[i][zk] = zhist[i].get(zk, 0) + 1
            ply += 1
            self.stats["rollout_plies"] += sum(moved)
            if not any(moved):
                break
            if ply % self.k_anchor == 0:
                self.stats["anchors"] += sum(moved)
                hidden, cache = self._prime_batched(self._pad_encs_fast(
                    [_enc(i) for i in range(B)]))
                cont = sm.continuation_head(hidden)
                avals = self._batch_values([_enc(i) for i in range(B)])
                for i in range(B):
                    if moved[i]:
                        paths[i].append((int(fc.zobrist(fboards[i])), avals[i]))
            else:
                self.stats["sm_steps"] += sum(moved)
                tok_t = torch.tensor(tokens, dtype=torch.long, device=device)
                hidden, cache = sm.backbone.step(token_id=tok_t, cache=cache)
                vals = torch.tanh(sm.value_head(hidden)).squeeze(-1)
                cont = sm.continuation_head(hidden)
                vlist = vals.detach().cpu().tolist()
                for i in range(B):
                    if moved[i]:
                        paths[i].append((int(fc.zobrist(fboards[i])), float(vlist[i])))
        if tt is not None:
            for i in range(B):
                for k, v in paths[i]:
                    tt.update(k, v)
        return [done_val[i] if done_ply[i] % 2 == 0 else -done_val[i] for i in range(B)]

    @staticmethod
    def _pad_encs_fast(encs):
        L = max(len(e) for e in encs)
        arr = np.zeros((len(encs), L), dtype=np.int64)
        for j, e in enumerate(encs):
            arr[j, :len(e)] = e
        return arr

    # ── Public search ──

    @torch.no_grad()
    def run_search(self, board: chess.Board):
        if self.batched:
            return self._run_search_batched(board)
        self.stats = {"sm_steps": 0, "anchors": 0, "eval_positions": 0,
                      "rollouts": 0, "rollout_plies": 0}
        enc, policy, wdl, av, unc = self._eval_root(board)
        self.stats["eval_positions"] += 1               # the root eval
        budget = self.sim_budget
        if self.adaptive_sims:
            lo, mid, hi = self.sim_bounds
            t_lo, t_hi = self.unc_bounds
            budget = lo if unc < t_lo else (mid if unc < t_hi else hi)

        legal = list(board.legal_moves)
        if not legal:
            return {"best": None, "edges": {}, "wdl": wdl, "root_value": self._value_from_wdl(wdl)}
        if len(legal) == 1:
            return {"best": legal[0], "edges": {legal[0]: (1, 0.0)}, "wdl": wdl,
                    "root_value": self._value_from_wdl(wdl)}

        # Priors over legal moves (+ optional Dirichlet root noise).
        priors = np.array([policy[encoder.encode_move(m)]
                           if encoder.encode_move(m) is not None else 1e-8 for m in legal])
        priors = np.clip(priors, 1e-12, None)
        priors /= priors.sum()
        if self.add_root_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal))
            priors = (1 - self.dirichlet_eps) * priors + self.dirichlet_eps * noise
        logits = np.log(priors)

        # Gumbel top-k candidate selection.
        g = _gumbel(len(legal))
        m_cand = min(self.m_root, len(legal))
        order = np.argsort(-(g + logits))[:m_cand]
        cand = [legal[i] for i in order]
        gmap = {legal[i]: g[i] for i in order}
        lmap = {legal[i]: logits[i] for i in order}
        qmap = {m: 0.0 for m in cand}
        nmap = {m: 0 for m in cand}

        # Prime the Search Mamba once at the root.
        _, cache0 = self.search_model.prime_board(enc)

        tt = TranspositionTable()
        rounds = max(1, int(math.ceil(math.log2(m_cand))))
        survivors = list(cand)

        for r in range(rounds):
            sims_each = max(1, budget // (rounds * max(1, len(survivors))))
            for a in survivors:
                mid_a = encoder.encode_move(a)
                for _ in range(sims_each):
                    val_a, cont_a, cache_a = self.search_model.eval_step(self._clone_cache(cache0), mid_a)
                    b2 = board.copy(stack=False)
                    b2.push(a)
                    # value of position-after-a from ITS (opponent) perspective:
                    v_after = self._rollout(b2, cache_a, cont_a, tt)
                    # root-perspective Q(a) = -v_after
                    qmap[a] += -v_after
                    nmap[a] += 1
            if len(survivors) <= 1:
                break

            def score(m):
                q = qmap[m] / max(1, nmap[m])
                return gmap[m] + lmap[m] + self.q_scale * q
            survivors.sort(key=score, reverse=True)
            survivors = survivors[: max(1, len(survivors) // 2)]

        edges = {m: (nmap[m], qmap[m] / max(1, nmap[m])) for m in cand}
        best = max(cand, key=lambda m: (qmap[m] / max(1, nmap[m]),
                                        nmap[m], lmap[m]))

        # Geometric tie-break among near-equal-Q candidates.
        if self.navigator is not None and getattr(self.navigator, "advantage_vectors", None):
            bq = qmap[best] / max(1, nmap[best])
            close = [m for m in cand if (qmap[m] / max(1, nmap[m])) >= bq - self.tie_threshold]
            if len(close) > 1:
                geo = self.navigator.score_moves(board, self.eval_model, close)
                best = close[int(torch.as_tensor(geo).argmax().item())]

        return {"best": best, "edges": edges, "wdl": wdl,
                "root_value": self._value_from_wdl(wdl), "uncertainty": unc}

    @torch.no_grad()
    def _run_search_batched(self, board: chess.Board):
        """Batched-rollout MARS: identical search to run_search, but every round's
        rollouts run in lockstep through _rollout_batch (THE speed fix)."""
        self.stats = {"sm_steps": 0, "anchors": 0, "eval_positions": 0,
                      "rollouts": 0, "rollout_plies": 0}
        enc, policy, wdl, av, unc = self._eval_root(board)
        self.stats["eval_positions"] += 1
        budget = self.sim_budget
        if self.adaptive_sims:
            lo, mid, hi = self.sim_bounds
            t_lo, t_hi = self.unc_bounds
            budget = lo if unc < t_lo else (mid if unc < t_hi else hi)

        legal = list(board.legal_moves)
        if not legal:
            return {"best": None, "edges": {}, "wdl": wdl,
                    "root_value": self._value_from_wdl(wdl)}
        if len(legal) == 1:
            return {"best": legal[0], "edges": {legal[0]: (1, 0.0)}, "wdl": wdl,
                    "root_value": self._value_from_wdl(wdl)}

        priors = np.array([policy[encoder.encode_move(m)]
                           if encoder.encode_move(m) is not None else 1e-8 for m in legal])
        priors = np.clip(priors, 1e-12, None); priors /= priors.sum()
        if self.add_root_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal))
            priors = (1 - self.dirichlet_eps) * priors + self.dirichlet_eps * noise
        logits = np.log(priors)

        g = _gumbel(len(legal))
        m_cand = min(self.m_root, len(legal))
        order = np.argsort(-(g + logits))[:m_cand]
        cand = [legal[i] for i in order]
        gmap = {legal[i]: g[i] for i in order}
        lmap = {legal[i]: logits[i] for i in order}
        qmap = {m: 0.0 for m in cand}
        nmap = {m: 0 for m in cand}

        _, cache0 = self.search_model.prime_board(enc)
        tt = TranspositionTable() if self.use_tt else None
        rounds = max(1, int(math.ceil(math.log2(m_cand))))
        survivors = list(cand)

        for r in range(rounds):
            sims_each = max(1, budget // (rounds * max(1, len(survivors))))
            # Build this round's full rollout batch: each survivor x sims_each.
            specs = [a for a in survivors for _ in range(sims_each)]
            B = len(specs)
            # Batched candidate step: expand root cache to B, step each candidate.
            cacheB = self.search_model._expand_cache(cache0, B)
            cand_tok = torch.tensor([int(encoder.encode_move(a)) for a in specs],
                                    dtype=torch.long, device=self.device)
            hB, cacheB = self.search_model.backbone.step(token_id=cand_tok, cache=cacheB)
            contB = self.search_model.continuation_head(hB)
            boards = []
            for a in specs:
                b2 = board.copy(stack=False); b2.push(a); boards.append(b2)
            leaves = self._rollout_batch(boards, cacheB, contB, tt)
            for j, a in enumerate(specs):
                qmap[a] += -leaves[j]      # root-perspective Q(a) = -v_after
                nmap[a] += 1
            if len(survivors) <= 1:
                break

            def score(m):
                q = qmap[m] / max(1, nmap[m])
                return gmap[m] + lmap[m] + self.q_scale * q
            survivors.sort(key=score, reverse=True)
            survivors = survivors[: max(1, len(survivors) // 2)]

        edges = {m: (nmap[m], qmap[m] / max(1, nmap[m])) for m in cand}
        best = max(cand, key=lambda m: (qmap[m] / max(1, nmap[m]), nmap[m], lmap[m]))
        if self.navigator is not None and getattr(self.navigator, "advantage_vectors", None):
            bq = qmap[best] / max(1, nmap[best])
            close = [m for m in cand if (qmap[m] / max(1, nmap[m])) >= bq - self.tie_threshold]
            if len(close) > 1:
                geo = self.navigator.score_moves(board, self.eval_model, close)
                best = close[int(torch.as_tensor(geo).argmax().item())]
        return {"best": best, "edges": edges, "wdl": wdl,
                "root_value": self._value_from_wdl(wdl), "uncertainty": unc}

    def search(self, board: chess.Board, temperature: float = 0.0):
        res = self.run_search(board)
        if res["best"] is None:
            return None
        if temperature <= 1e-3:
            return res["best"]
        moves = list(res["edges"].keys())
        visits = np.array([res["edges"][m][0] for m in moves], dtype=np.float64)
        p = visits ** (1.0 / temperature)
        p /= p.sum()
        return moves[int(np.random.choice(len(moves), p=p))]

    def get_policy_target(self, res, temperature=1.0):
        """Dense visit distribution over the action space (for self-play)."""
        pi = np.zeros(ACTION_SPACE, dtype=np.float32)
        items = [(encoder.encode_move(m), n) for m, (n, q) in res["edges"].items()
                 if encoder.encode_move(m) is not None and encoder.encode_move(m) < ACTION_SPACE]
        if not items:
            return pi
        ids = [i for i, _ in items]
        visits = np.array([n for _, n in items], dtype=np.float64)
        if temperature <= 1e-3:
            pi[ids[int(visits.argmax())]] = 1.0
        else:
            v = visits ** (1.0 / temperature)
            v = v / v.sum() if v.sum() > 0 else np.ones_like(v) / len(v)
            for i, p in zip(ids, v):
                pi[i] = p
        return pi
