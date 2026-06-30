"""
ChessMamba v3 - end-to-end integration test (CPU, tiny configs).

Run:  python test_v3.py
Exercises every v3 change: action-value head, O(1) step inference, fast cached
line eval, enhanced MCTS (PUCT/AV-init/Dirichlet/progressive-widening, correct
value sign), geometric 15-vector navigation, MCTS-primary selector, self-play,
the data pipeline (soft policy + side-to-move values + strategy labels), and the
full training loss with backward.
"""

import sys, os, tempfile, csv, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, chess
import numpy as np

from encoding import encoder, ACTION_SPACE
from mamba import MambaConfig, Mamba
from model import ChessMamba
from search_mamba import SearchMamba, SearchMambaConfig
from batched_mcts import BatchedMCTS
from move_selector import MoveSelector

PASS = FAIL = 0
def test(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  [PASS] {name}")
    else:
        FAIL += 1; print(f"  [FAIL] {name} -- {detail}")

torch.manual_seed(0)
print("=" * 64 + "\nChessMamba v3 - Integration Tests\n" + "=" * 64)

vocab = encoder.vocab_size() + 100
cfg = MambaConfig(d_model=48, n_layer=3, d_state=8, mimo_p=2, vocab_size=vocab)

# ── 1. Encoding + action space ──
print("\n[1] Encoding / action space")
test("ACTION_SPACE covers all move ids", ACTION_SPACE > max(encoder.id_to_move.keys()))
test("Start position encodes", len(encoder.encode_board(chess.Board())) > 70)
test("phase_index opening", encoder.phase_index(chess.Board()) == 0)

# ── 2. Mamba O(1) step == parallel forward ──
print("\n[2] Mamba step() exactness")
m = Mamba(cfg).eval()
ids = torch.randint(0, vocab, (2, 12))
with torch.no_grad():
    full = m(ids)
    cache = None; outs = []
    for t in range(12):
        h, cache = m.step(token_id=ids[:, t], cache=cache)
        outs.append(h)
    rec = torch.stack(outs, 1)
test("step matches forward", torch.allclose(full, rec, atol=1e-4),
     f"max {abs((full-rec)).max().item():.2e}")
with torch.no_grad():
    fin, cache = m.prime_cache(ids[:, :6])
    cont = torch.stack([m.step(token_id=ids[:, t], cache=cache)[0] if t == 6
                        else None for t in [6]], 0)
test("prime_cache last == forward last",
     torch.allclose(fin, m(ids[:, :6])[:, -1, :], atol=1e-4))

# ── 3. Model heads (incl action-value) ──
print("\n[3] Model heads")
model = ChessMamba(cfg).eval()
x = torch.tensor([encoder.encode_board(chess.Board())], dtype=torch.long)
out = model(x, return_dict=True)
test("policy shape", out["policy"].shape == (1, ACTION_SPACE))
test("wdl sums to 1", abs(out["wdl"].sum().item() - 1) < 1e-3)
test("action_value head present", "action_value" in out)
test("action_value bounded [-1,1]", out["action_value"].abs().max().item() <= 1.0001)
test("action_value shape", out["action_value"].shape == (1, ACTION_SPACE))
full = model.forward_full(x)
test("forward_full has all 6 outputs",
     all(k in full for k in ["embedding","policy","wdl","uncertainty","strategy","action_value"]))

# ── 4. Search Mamba fast == slow ──
print("\n[4] Search Mamba O(1) cached line eval")
scfg = SearchMambaConfig(d_model=32, n_layer=2, d_state=8, mimo_p=2, vocab_size=vocab)
sm = SearchMamba(scfg, vocab_size=vocab).eval()
board_enc = torch.tensor(encoder.encode_board(chess.Board()), dtype=torch.long)
lines = torch.randint(1, vocab, (6, 5))
with torch.no_grad():
    slow = sm.evaluate_lines(board_enc, lines)
    fast = sm.evaluate_lines_fast(board_enc, lines)
test("fast line eval == slow", torch.allclose(slow, fast, atol=1e-4),
     f"max {abs(slow-fast).max().item():.2e}")

# ── 5. Batched MCTS ──
print("\n[5] Batched MCTS")
mcts = BatchedMCTS(model, num_simulations=80, batch_size=8, use_tablebase=False,
                   adaptive_sims=False, add_root_noise=True)
b = chess.Board(); b.push_san("e4"); b.push_san("e5")
root = mcts.run_search(b)
test("root visited num_simulations times", root.visit_count == 80, f"{root.visit_count}")
test("root expanded children", len(root.children) > 0)
pi = mcts.get_policy_target(root, temperature=1.0)
test("policy target sums to 1", abs(pi.sum() - 1) < 1e-4)
test("policy target only on legal moves",
     all(encoder.decode_move(i) in b.legal_moves for i in np.nonzero(pi)[0]))
mv = mcts.search(b, temperature=0.0)
test("search returns legal move", mv in b.legal_moves)

# value-sign sanity: a forced-mate-in-1 for side to move should be strongly +ve
mate_board = chess.Board("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1")  # Re8#
mroot = mcts.run_search(mate_board)
best = max(mroot.children, key=lambda mm: mroot.children[mm].visit_count)
test("MCTS finds the mate (Re8#)", mate_board.san(best) in ("Re8#", "Re8"),
     f"got {mate_board.san(best)}")

# ── 6. Geometric navigation (15-vector) ──
print("\n[6] Geometric navigation")
from geometric import GeometricNavigator, compute_advantage_vectors, CONTEXTS
fake = {"global": torch.randn(48), "middlegame": torch.randn(48),
        "middlegame:attacking": torch.randn(48)}
nav = GeometricNavigator(fake)
legal = list(chess.Board().legal_moves)[:5]
gs = nav.score_moves(chess.Board(), model, legal)
test("geo score_moves shape", gs.shape[0] == len(legal))
test("geo vector fallback chain", nav._select_vector("endgame", "holding") is not None)

# ── 7. Move selector (MCTS-primary) ──
print("\n[7] Move selector")
sel = MoveSelector(model, num_simulations=40, batch_size=8, adaptive_sims=False,
                   use_tablebase=False, navigator=nav)
mv = sel.select_move(chess.Board())
test("selector returns legal move", mv in chess.Board().legal_moves)
analysis = sel.get_analysis(chess.Board(), top_n=3)
test("analysis returns ranked moves", len(analysis) == 3 and "visits" in analysis[0])

# ── 8. Self-play one game ──
print("\n[8] Self-play")
import self_play
sp_mcts = BatchedMCTS(model, num_simulations=12, batch_size=8, add_root_noise=True,
                      use_tablebase=False)
samples, z = self_play.play_game(sp_mcts, max_moves=6, temp_moves=3)
test("self-play produced samples", len(samples) > 0)
test("self-play value in [-1,1]", all(-1 <= float(v) <= 1 for *_, v in samples))

# ── 9. Data pipeline + full training loss ──
print("\n[9] Data pipeline + training loss")
d = tempfile.mkdtemp()
import random; random.seed(0)
rows, seen = [], set()
for g in range(20):
    bb = chess.Board()
    for _ in range(12):
        lm = list(bb.legal_moves)
        if not lm: break
        bb.push(random.choice(lm))
        key = " ".join(bb.fen().split()[:4])
        if key in seen: continue
        seen.add(key); lm2 = list(bb.legal_moves)
        if not lm2: break
        rows.append((bb.fen(), random.choice(lm2).uci(), 25, random.randint(-300, 300)))
with open(os.path.join(d, "e.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["fen","line","depth","cp","mate"])
    for fen, bm, dep, cp in rows: w.writerow([fen, bm, dep, cp, ""])
from data.convert_lichess_evals import convert_lichess_evals, cp_to_wdl
out = os.path.join(d, "td")
convert_lichess_evals(d, out, max_positions=300, min_depth=20, stratify=False)
test("WDL antisymmetric", all(abs((cp_to_wdl(c)[0]-cp_to_wdl(c)[2])
                                  + (cp_to_wdl(-c)[0]-cp_to_wdl(-c)[2])) < 1e-9
                                  for c in range(0, 600, 50)))
from train import EvalMambaTrainer
tcfg = MambaConfig(d_model=40, n_layer=2, d_state=8, mimo_p=2, vocab_size=vocab,
                   grad_checkpoint=True)
tr = EvalMambaTrainer(out, tcfg, batch_size=8, device="cpu", num_workers=0)
trl, val = tr.create_dataloaders()
loss, parts = tr._compute_loss(next(iter(trl)))
test("training loss finite", math.isfinite(loss.item()))
test("all 6 loss parts + t1 metric present",
     set(parts) == {"policy","wdl","con","strat","av","unc","t1"})
loss.backward()
gsum = sum(p.grad.abs().sum().item() for p in tr.model.parameters() if p.grad is not None)
test("gradients finite & nonzero", math.isfinite(gsum) and gsum > 0)

# ── 10. Serialization ──
print("\n[10] Serialization")
with tempfile.TemporaryDirectory() as tmp:
    p = os.path.join(tmp, "m.pt")
    model.save_pretrained(p)
    loaded = ChessMamba.from_pretrained(p, cfg)
    test("model reload weights match",
         all(torch.equal(a, b) for a, b in zip(model.parameters(), loaded.parameters())))
    sp = os.path.join(tmp, "s.pt")
    sm.save_pretrained(sp)
    test("search config sidecar", os.path.exists(sp.replace(".pt", "_config.json")))
    SearchMamba.from_pretrained(sp)
    test("search reload ok", True)

# ── 11. UCI parsing ──
print("\n[11] UCI parsing")
from uci import UCIEngine
u = UCIEngine()
u._cmd_position(["position", "startpos", "moves", "e2e4", "e7e5"])
test("UCI startpos+moves", u.board.fen().startswith("rnbqkbnr/pppp1ppp"))
u._cmd_setoption(["setoption", "name", "AdaptiveSims", "value", "false"])
test("UCI bool option parsed", u.options["AdaptiveSims"] is False)

# ── 12. Param counts ──
print("\n[12] Param counts (full-size configs)")
fc = MambaConfig(d_model=512, n_layer=16, d_state=64, mimo_p=4, vocab_size=vocab)
fe = ChessMamba(fc); pe = sum(p.numel() for p in fe.parameters())
print(f"  Eval Mamba d=512/L=16: {pe:,} params")
test("eval params in sane range", 40e6 < pe < 130e6, f"{pe:,}")
del fe

print("\n" + "=" * 64)
print(f"Results: {PASS} passed, {FAIL} failed")
print("=" * 64)
sys.exit(1 if FAIL else 0)
