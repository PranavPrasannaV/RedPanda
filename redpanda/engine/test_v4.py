"""
ChessMamba v4 (MARS) - integration tests (CPU, tiny configs).

Run:  python test_v4.py
Covers every v4 addition: negative-eigenvalue tracking channels, bidirectional
Eval backbone, value-equivalent Search-Mamba dynamics (+ consistency loss),
MCGS transposition table, the MARS search (Gumbel sequential halving + recurrent
rollouts + re-anchor + quiescence + MCGS), the MoveSelector MARS backend,
self-play via MARS, and the dynamics-training data path.
"""

import sys, os, tempfile, csv, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, chess, random
import numpy as np
from dataclasses import asdict

from encoding import encoder, ACTION_SPACE
from mamba import Mamba, MambaConfig
from model import ChessMamba
from search_mamba import SearchMamba, SearchMambaConfig

PASS = FAIL = 0
def test(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  [PASS] {name}")
    else: FAIL += 1; print(f"  [FAIL] {name} -- {detail}")

torch.manual_seed(0); np.random.seed(0); random.seed(0)
print("=" * 64 + "\nChessMamba v4 (MARS) - Integration Tests\n" + "=" * 64)
vocab = encoder.vocab_size() + 100

# ── 1. Backbone: negative-eigenvalue tracking + bidirectional ──
print("\n[1] Backbone upgrades")
cfg = MambaConfig(d_model=32, n_layer=3, d_state=12, n_track_state=4, mimo_p=2,
                  vocab_size=vocab, bidirectional=False)
m = Mamba(cfg).eval()
ids = torch.randint(0, vocab, (2, 10))
with torch.no_grad():
    full = m(ids)
    cache = None; outs = []
    for t in range(10):
        h, cache = m.step(token_id=ids[:, t], cache=cache); outs.append(h)
    rec = torch.stack(outs, 1)
test("step() bit-exact WITH tracking channels", torch.allclose(full, rec, atol=1e-4),
     f"max {abs(full-rec).max().item():.2e}")
blk = m.layers[0]["mixer"]
test("tracking eigenvalues span negative", float(blk._track_gate().min()) < -0.5,
     f"min {float(blk._track_gate().min()):.3f}")
ecfg = MambaConfig(d_model=32, n_layer=2, d_state=8, mimo_p=2, vocab_size=vocab,
                   bidirectional=True)
mb = Mamba(ecfg).eval()
with torch.no_grad():
    yb = mb(ids)
test("bidirectional Eval forward runs", yb.shape == (2, 10, 32))
try:
    mb.prime_cache(ids); test("bidirectional rejects prime_cache", False)
except AssertionError:
    test("bidirectional rejects prime_cache", True)

# ── 2. Search Mamba value-equivalent dynamics ──
print("\n[2] Search Mamba dynamics")
scfg = SearchMambaConfig(d_model=24, n_layer=2, d_state=8, n_track_state=2, mimo_p=2,
                         vocab_size=vocab)
sm = SearchMamba(scfg, vocab_size=vocab, eval_dim=32)
board = torch.randint(1, vocab, (3, 20))
moves = torch.randint(1, vocab, (3, 5))
v, c, f = sm.dynamics_rollout(board, moves)
test("dynamics_rollout shapes", v.shape == (3, 5) and c.shape == (3, 5, scfg.vocab_size)
     and f.shape == (3, 5, 32))
test("dynamics values tanh-bounded", v.abs().max().item() <= 1.0001)
(v.pow(2).mean() + c.mean() + f.pow(2).mean()).backward()
g = sum(p.grad.abs().sum().item() for p in sm.parameters() if p.grad is not None)
test("dynamics differentiable", math.isfinite(g) and g > 0)
sm.eval()
_, cache = sm.prime_board(board[0])
val, cont, cache = sm.eval_step(cache, int(moves[0, 0]))
test("eval_step returns (value, cont)", isinstance(val, float) and cont.shape[0] == scfg.vocab_size)

# ── 3. MCGS transposition table ──
print("\n[3] MCGS")
from mcgs import TranspositionTable, zobrist_key
tt = TranspositionTable()
b = chess.Board()
k = zobrist_key(b)
test("zobrist transposition equal", zobrist_key(chess.Board()) == k)
tt.update(k, 0.5); tt.update(k, -0.1)
test("TT averages value", abs(tt.get_value(k) - 0.2) < 1e-6)
test("TT visit count", tt.visits(k) == 2)

# ── 4. MARS search ──
print("\n[4] MARS search")
from mars_search import MARS
eval_model = ChessMamba(MambaConfig(d_model=40, n_layer=2, d_state=12, n_track_state=4,
                                    mimo_p=2, vocab_size=vocab, bidirectional=True)).eval()
search_model = SearchMamba(SearchMambaConfig(d_model=32, n_layer=2, d_state=8,
                                             n_track_state=2, mimo_p=2, vocab_size=vocab),
                           vocab_size=vocab, eval_dim=40).eval()
mars = MARS(eval_model, search_model, sim_budget=24, m_root=8, k_anchor=3, depth_cap=6,
            use_tablebase=False, adaptive_sims=False)
bb = chess.Board(); bb.push_san("e4"); bb.push_san("e5")
res = mars.run_search(bb)
test("MARS best move legal", res["best"] in bb.legal_moves)
pi = mars.get_policy_target(res, temperature=1.0)
test("MARS policy target sums to 1", abs(pi.sum() - 1) < 1e-4)
test("MARS policy on legal moves only",
     all(encoder.decode_move(i) in bb.legal_moves for i in np.nonzero(pi)[0]))
test("MARS temperature sampling legal", mars.search(bb, temperature=1.0) in bb.legal_moves)
mate = chess.Board("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1")
# Use m_root >= legal-move count so Re8# is always a CANDIDATE (otherwise we'd be
# testing the Gumbel sampling lottery, not the search). With the mate in the
# candidate set the deep rollout must back up the terminal win and pick it.
mate_mars = MARS(eval_model, search_model, sim_budget=24, m_root=24, depth_cap=6,
                 use_tablebase=False, adaptive_sims=False)
hits = sum(1 for _ in range(6) if mate.san(mate_mars.run_search(mate)["best"]) == "Re8#")
test("MARS finds mate-in-1 (mate in candidate set, random nets)", hits >= 5, f"{hits}/6")

# ── 5. MoveSelector MARS backend ──
print("\n[5] MoveSelector backend")
from move_selector import MoveSelector
sel = MoveSelector(eval_model, search_model, backend="mars", use_tablebase=False,
                   mars_kwargs={"sim_budget": 16, "m_root": 6, "depth_cap": 5,
                                "adaptive_sims": False})
test("selector backend is mars", sel.backend == "mars")
test("selector MARS returns legal move", sel.select_move(chess.Board()) in chess.Board().legal_moves)
sel_mcts = MoveSelector(eval_model, None, backend="mars", use_tablebase=False)
test("selector falls back to mcts without search model", sel_mcts.backend == "mcts")
an = sel.get_analysis(chess.Board(), top_n=3)
test("MARS analysis returns ranked moves", len(an) >= 1 and "visits" in an[0])

# ── 6. Self-play via MARS ──
print("\n[6] Self-play (MARS)")
import self_play
sp_mars = MARS(eval_model, search_model, sim_budget=10, m_root=5, depth_cap=4,
               use_tablebase=False, adaptive_sims=False, add_root_noise=True)
samples, z = self_play.play_game_mars(sp_mars, max_moves=6, temp_moves=3)
test("self-play(MARS) produced samples", len(samples) > 0)
test("self-play(MARS) values in [-1,1]", all(-1 <= float(s[3]) <= 1 for s in samples))

# ── 7. Data + dynamics training path ──
print("\n[7] Dynamics training path")
d = tempfile.mkdtemp()
seen = set(); rows = []
for gi in range(25):
    bd = chess.Board()
    for _ in range(14):
        lm = list(bd.legal_moves)
        if not lm: break
        bd.push(random.choice(lm))
        key = " ".join(bd.fen().split()[:4])
        if key in seen: continue
        seen.add(key); lm2 = list(bd.legal_moves)
        if len(lm2) < 2: break
        pv = [random.choice(lm2).uci()]; tb = bd.copy()
        for _ in range(2):
            tb.push(chess.Move.from_uci(pv[-1])); l3 = list(tb.legal_moves)
            if not l3: break
            pv.append(random.choice(l3).uci())
        rows.append((bd.fen(), " ".join(pv), 25, random.randint(-250, 250)))
with open(os.path.join(d, "e.csv"), "w", newline="") as fcsv:
    w = csv.writer(fcsv); w.writerow(["fen", "line", "depth", "cp", "mate"])
    for fen, line, dep, cp in rows: w.writerow([fen, line, dep, cp, ""])
out = os.path.join(d, "td")
from data.convert_lichess_evals import convert_lichess_evals
convert_lichess_evals(d, out, max_positions=400, min_depth=20, stratify=False)
test("converter emits start_fens.json", os.path.exists(os.path.join(out, "start_fens.json")))
from data.dataset import DynamicsDataset
dds = DynamicsDataset(out, max_pv=6)
it = dds[0]
test("DynamicsDataset item shapes",
     it["board_enc"].shape[0] == 160 and it["inter_enc"].shape == (6, 160))
work = tempfile.mkdtemp(); os.chdir(work)
tcfg = MambaConfig(d_model=32, n_layer=2, d_state=8, n_track_state=2, mimo_p=2,
                   vocab_size=vocab, bidirectional=True)
em = ChessMamba(tcfg)
torch.save(em.state_dict(), "chess_mamba.pt")
json.dump(asdict(tcfg), open("chess_mamba_config.json", "w"))
from train import DynamicsTrainer
dt = DynamicsTrainer(out, "chess_mamba.pt", "chess_mamba_config.json",
                     SearchMambaConfig(d_model=24, n_layer=2, d_state=8, n_track_state=2,
                                       mimo_p=2, vocab_size=vocab, bidirectional=False),
                     batch_size=8, device="cpu", num_workers=0, max_pv=6)
trl, _ = dt.create_dataloader()
loss, parts = dt._loss(next(iter(trl)))
test("dynamics loss has value+cont+consist", set(parts) == {"value", "cont", "consist"})
loss.backward()
gg = sum(p.grad.abs().sum().item() for p in dt.search.parameters() if p.grad is not None)
test("dynamics training backward finite", math.isfinite(gg) and gg > 0)

print("\n" + "=" * 64)
print(f"Results: {PASS} passed, {FAIL} failed")
print("=" * 64)
sys.exit(1 if FAIL else 0)
