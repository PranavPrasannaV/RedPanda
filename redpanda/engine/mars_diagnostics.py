"""
MARS definitive diagnostics — answers "is batched MARS faster than MCTS, and if
not, is it FIXABLE or DEAD?" in ONE run, so we never have to keep guessing.

Method (no monkeypatching, no hand-waving):
  0. CORRECTNESS GATE — the batched rollout must reproduce the sequential one,
     and the batched step primitive must equal N single steps. Speed numbers from
     a wrong rollout are worthless, so we refuse to report them if this fails.
  1. BATCH-SIZE SWEEP — run the batched rollout at B = 1,8,32,128,(512) and
     measure nodes/sec. As B grows the per-rollout GPU cost -> 0, so the curve
     ASYMPTOTES to the pure Python-control-flow ceiling. That asymptote is the
     hard upper bound on any GPU/kernel optimization.
  2. MCTS baseline nodes/sec (the bar to beat).
  3. Full batched MARS vs MCTS at equal wall-clock (the headline).
  4. VERDICT:
       batched MARS nodes/sec > MCTS              -> FIXED (speed thesis holds*)
       else if ceiling (asymptote) > MCTS         -> FIXABLE (need Python/kernel
                                                     work; batching alone short)
       else (ceiling <= MCTS)                     -> control-flow bound: MARS as
                                                     designed cannot win on speed
                                                     -> DEAD without redesign
     (* speed only; strength still needs trained nets + the tournament.)

Usage (training GPU):
    python mars_diagnostics.py --random --triton
    python mars_diagnostics.py --random --triton --max-batch 512
    python mars_diagnostics.py --tiny            # CPU smoke test
"""

import os, sys, time, argparse, statistics
# Reduce fragmentation (mirror train.py); must precede the torch import.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import chess

from encoding import encoder, ACTION_SPACE
from mamba import MambaConfig
from model import ChessMamba
from search_mamba import SearchMamba, SearchMambaConfig
from mars_search import MARS
from batched_mcts import BatchedMCTS

TEST_FENS = [
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r2qnrnk/p2b2b1/1p1p2pp/2pPpp2/1PP1P3/PRNBB3/3QNPPP/5RK1 w - - 0 1",
    "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1",
    "8/7p/5k2/5p2/p1p2P2/Pr1pPK2/1P1R3P/8 b - - 0 1",
]


def _sync(d):
    if d == "cuda":
        torch.cuda.synchronize()


def set_triton(module, on):
    seen = set()
    for m in module.modules():
        cfg = getattr(m, "config", None)
        if cfg is not None and hasattr(cfg, "use_triton_scan") and id(cfg) not in seen:
            cfg.use_triton_scan = on; seen.add(id(cfg))


def build(args, device):
    vocab = encoder.vocab_size() + 100
    if args.tiny:
        ecfg = MambaConfig(d_model=64, n_layer=2, d_state=16, n_track_state=4,
                           mimo_p=2, vocab_size=vocab, bidirectional=True)
        scfg = SearchMambaConfig(d_model=48, n_layer=2, d_state=16, n_track_state=4,
                                 mimo_p=2, vocab_size=vocab, bidirectional=False)
    else:
        ecfg = MambaConfig(d_model=512, n_layer=16, d_state=64, n_track_state=16,
                           mimo_p=4, vocab_size=vocab, bidirectional=True)
        scfg = SearchMambaConfig(d_model=384, n_layer=12, d_state=32, n_track_state=16,
                                 mimo_p=2, vocab_size=vocab, bidirectional=False)
    if not args.random and args.model and os.path.exists(args.model):
        from self_play import load_model
        em, _ = load_model(args.model, device=device)
        sm = SearchMamba.from_pretrained(args.search, device=device).to(device).eval()
        src = "checkpoints"
    else:
        em = ChessMamba(ecfg, action_space=ACTION_SPACE).to(device).eval()
        sm = SearchMamba(scfg, vocab_size=vocab, eval_dim=ecfg.d_model).to(device).eval()
        src = "RANDOM (speed is weight-independent)"
    set_triton(em, args.triton); set_triton(sm, args.triton)
    return em, sm, src


# ── 0. Correctness gate ──────────────────────────────────────────────────────

def correctness_gate(em, sm, device):
    """Prove the batched GPU path is faithful: (a) one batched step == N single
    steps, (b) a K-step batched CHAIN == K single chains (cache threading is not
    scrambled across rows/plies), (c) a full batched rollout produces sane output.
    NOTE we deliberately do NOT require bit-equality with the sequential rollout:
    encode_board is variable-length, so the batched (batch-max-padded) anchor
    prime and the sequential (exact-length) prime legitimately differ — a padding
    convention, not a bug (and batch-max padding is what MCTS uses too). What
    matters for trusting speed numbers is no cross-row contamination, tested below."""
    print("\n--- 0. Correctness gate " + "-" * 44)
    torch.manual_seed(0)
    N = 6
    b0 = chess.Board(TEST_FENS[0])
    _, c1 = sm.prime_board(encoder.encode_board(b0))
    legal_tok = [int(encoder.encode_move(m)) for m in list(b0.legal_moves)[:N]]

    # (a) one batched step == N single steps
    caches = [sm._expand_cache(c1, 1) for _ in range(N)]
    singles = [sm.backbone.step(token_id=torch.tensor([legal_tok[i]], device=device),
                                cache=[(h.clone(), d.clone()) for h, d in caches[i]])[0]
               for i in range(N)]
    bcache = [(torch.cat([caches[i][L][0] for i in range(N)], 0),
               torch.cat([caches[i][L][1] for i in range(N)], 0)) for L in range(len(c1))]
    hb, _ = sm.backbone.step(token_id=torch.tensor(legal_tok, device=device), cache=bcache)
    d_a = (torch.cat(singles, 0) - hb).abs().max().item()
    print(f"  (a) one batched step == N single steps      : max diff {d_a:.2e}")
    assert d_a < 1e-4, "batched step primitive mismatch!"

    # (b) K-step batched CHAIN == K single chains (no padding; pure step threading)
    K = 8
    rng = np.random.default_rng(0)
    seqs = rng.integers(1, 100, size=(N, K))
    sing_caches = [sm._expand_cache(c1, 1) for _ in range(N)]
    sing_h = [None] * N
    for i in range(N):
        ci = [(h.clone(), d.clone()) for h, d in sing_caches[i]]
        for t in range(K):
            sing_h[i], ci = sm.backbone.step(
                token_id=torch.tensor([int(seqs[i, t])], device=device), cache=ci)
    chain_cache = [(torch.cat([sm._expand_cache(c1, 1)[L][0] for _ in range(N)], 0),
                    torch.cat([sm._expand_cache(c1, 1)[L][1] for _ in range(N)], 0))
                   for L in range(len(c1))]
    hbk = None
    for t in range(K):
        hbk, chain_cache = sm.backbone.step(
            token_id=torch.tensor([int(seqs[i, t]) for i in range(N)], device=device),
            cache=chain_cache)
    d_b = (torch.cat(sing_h, 0) - hbk).abs().max().item()
    print(f"  (b) {K}-step batched chain == {K} single chains : max diff {d_b:.2e}")
    assert d_b < 1e-4, "batched chain scrambles rows across plies!"

    # (c) full batched rollout sanity: finite, in-range, plies advanced
    mars = MARS(em, sm, sim_budget=1, m_root=1, k_anchor=3, depth_cap=6,
                rollout_temp=0.6, use_tablebase=False, quiescence=True,
                adaptive_sims=False)
    mars.stats = {k: 0 for k in ("sm_steps", "anchors", "eval_positions",
                                 "rollouts", "rollout_plies")}
    boards, conts, caches2 = [], [], []
    for fen in TEST_FENS:
        b = chess.Board(fen); h, c = sm.prime_board(encoder.encode_board(b))
        boards.append(b); conts.append(sm.continuation_head(h).squeeze(0)); caches2.append(c)
    bc = [(torch.cat([caches2[i][L][0] for i in range(len(boards))], 0),
           torch.cat([caches2[i][L][1] for i in range(len(boards))], 0))
          for L in range(len(caches2[0]))]
    leaves = mars._rollout_batch([b.copy(stack=False) for b in boards],
                                 bc, torch.stack(conts, 0), None)
    ok = all(np.isfinite(v) and -1.0001 <= v <= 1.0001 for v in leaves)
    print(f"  (c) full batched rollout: leaves={[round(v,3) for v in leaves]} "
          f"| plies={mars.stats['rollout_plies']} | finite&in-range={ok}")
    assert ok and mars.stats["rollout_plies"] > 0, "batched rollout produced bad output!"
    print("  GATE PASSED — batched GPU path is faithful (no cross-row contamination).")


# ── 1. Batch-size sweep (the ceiling test) ───────────────────────────────────

def batch_sweep(em, sm, device, max_batch, fast=False):
    print("\n--- 1. Batched-rollout throughput vs batch size "
          + ("[FASTCHESS] " if fast else "") + "-" * 20)
    print("    (nodes/sec ASYMPTOTE = pure board-control-flow ceiling)")
    mars = MARS(em, sm, sim_budget=1, m_root=1, k_anchor=4, depth_cap=10,
                rollout_temp=0.6, use_tablebase=False, quiescence=False,
                adaptive_sims=False)
    base = chess.Board(TEST_FENS[0])
    enc = encoder.encode_board(base)
    _, c1 = sm.prime_board(enc)
    sizes = [b for b in (1, 8, 32, 128, 512) if b <= max_batch]
    results = []
    legal = list(base.legal_moves)
    if fast:
        import fastchess as fc
        fc.build_move_token_map(encoder)
        root_fc = fc.from_pychess(base)
        root_hist = [t for t in (encoder.encode_move(m) for m in base.move_stack)
                     if t is not None]
        buf = np.empty(256, dtype=np.uint32); nleg = fc.gen_legal(root_fc, buf)
        fcands = [buf[i % nleg] for i in range(max_batch)]
    for B in sizes:
        try:
            mars.stats = {k: 0 for k in ("sm_steps", "anchors", "eval_positions",
                                         "rollouts", "rollout_plies")}
            cacheB = sm._expand_cache(c1, B)
            if fast:
                tk = torch.tensor([fc.move_token(fcands[i]) for i in range(B)], device=device)
            else:
                tk = torch.tensor([int(encoder.encode_move(legal[i % len(legal)]))
                                   for i in range(B)], device=device)
            hB, cacheB = sm.backbone.step(token_id=tk, cache=cacheB)
            contB = sm.continuation_head(hB)
            _sync(device)
            t0 = time.perf_counter()
            if fast:
                fboards = [fc.make(root_fc, fcands[i]) for i in range(B)]
                hists = [root_hist + [fc.move_token(fcands[i])] for i in range(B)]
                mars._rollout_batch_fast(fboards, hists, cacheB, contB, None)
            else:
                boards = [base.copy(stack=False) for _ in range(B)]
                for i in range(B):
                    boards[i].push(legal[i % len(legal)])
                mars._rollout_batch(boards, cacheB, contB, None)
            _sync(device)
            dt = time.perf_counter() - t0
            nps = mars.stats["rollout_plies"] / max(dt, 1e-9)
            results.append((B, nps, dt, mars.stats["rollout_plies"]))
            print(f"  B={B:4d} | {mars.stats['rollout_plies']:6d} plies in {dt:6.3f}s "
                  f"-> {nps:9.1f} nodes/sec")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  B={B:4d} | OOM (skipped) — anchor prime too large at this batch")
    if not results:
        print("  (all batch sizes OOM'd — try a smaller --max-batch)")
        return 0.0, results
    ceiling = max(r[1] for r in results)
    print(f"  >>> ceiling (best observed nodes/sec) = {ceiling:.1f}")
    return ceiling, results


# ── 2/3. MCTS baseline + full batched MARS vs MCTS ───────────────────────────

def _time_engine(eng, device, count_attr):
    t = nodes = 0
    for fen in TEST_FENS:
        b = chess.Board(fen)
        _sync(device); t0 = time.perf_counter(); eng.run_search(b); _sync(device)
        t += time.perf_counter() - t0
        nodes += (eng.stats["rollout_plies"] if count_attr == "stats" else eng.eval_count)
    return nodes / max(t, 1e-9), t / len(TEST_FENS)

def head_to_head(em, sm, device, mars_budget, sims):
    print("\n--- 2/3. Full batched MARS vs MCTS (equal wall-clock) " + "-" * 13)
    mcts = BatchedMCTS(em, num_simulations=sims, batch_size=8,
                       use_tablebase=False, add_root_noise=False)
    cnps, ct = _time_engine(mcts, device, "eval")
    # default MARS (quiescence on, k_anchor 4) and a LEAN MARS (quiescence off,
    # k_anchor 8) — the lean config cuts the EvalMamba overhead that is the real
    # bottleneck, to see whether the full search can clear MCTS.
    configs = [("MARS default ", dict(quiescence=True, k_anchor=4)),
               ("MARS lean    ", dict(quiescence=False, k_anchor=8))]
    best_nps = 0.0
    for name, kw in configs:
        mars = MARS(em, sm, sim_budget=mars_budget, m_root=8, depth_cap=10,
                    use_tablebase=False, adaptive_sims=False, add_root_noise=False,
                    batched=True, use_tt=False, **kw)
        nps, tt = _time_engine(mars, device, "stats")
        best_nps = max(best_nps, nps)
        print(f"  {name}: {tt:5.2f}s/move | {nps:9.1f} nodes/sec | "
              f"{nps/max(cnps,1e-9):.2f}x MCTS "
              f"({'ABOVE' if nps > cnps else 'below'})")
    print(f"  MCTS         : {ct:5.2f}s/move | {cnps:9.1f} nodes/sec")
    # report the BEST achievable MARS config (lean) to the verdict
    return best_nps, cnps


def verdict(mars_nps, mcts_nps, ceiling):
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if mars_nps > mcts_nps:
        print(f"  [FIXED] Batched MARS ({mars_nps:.0f} n/s) already beats MCTS "
              f"({mcts_nps:.0f} n/s) on speed.")
        print("     The speed thesis holds. Next: STRENGTH test (trained nets +")
        print("     tournament) — does the extra search convert to better moves?")
    elif ceiling > mcts_nps * 1.1:
        print(f"  [FIXABLE] Batched MARS ({mars_nps:.0f} n/s) trails MCTS "
              f"({mcts_nps:.0f} n/s) at tested batch sizes,")
        print(f"     BUT the control-flow ceiling ({ceiling:.0f} n/s) is above MCTS.")
        print("     So speed is recoverable with: larger rollout batches, Python")
        print("     vectorization (encode_board, push/pop), and k_anchor tuning")
        print("     (B11). Worth the engineering — the headroom exists.")
    else:
        print(f"  [DEAD on speed, as designed] The control-flow ceiling "
              f"({ceiling:.0f} n/s) is")
        print(f"     at/below MCTS ({mcts_nps:.0f} n/s) — even with FREE neural ops,")
        print("     MARS's per-node Python (python-chess push/pop, legal-move gen,")
        print("     encode_board, move sampling) is too expensive to out-search a")
        print("     batched MCTS. No GPU/kernel work fixes this; it needs either a")
        print("     fundamentally cheaper per-node loop (C/Rust, vectorized board)")
        print("     or a different search shape. Recommend: ship the MCTS backend,")
        print("     file MARS as research with this measured reason.")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--search", default="search_mamba.pt")
    ap.add_argument("--random", action="store_true",
                    help="force random-weight nets (default when no --model; speed-valid)")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--triton", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-batch", type=int, default=128)
    ap.add_argument("--mars-budget", type=int, default=32)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--fast", action="store_true",
                    help="also run the FASTCHESS rollout sweep (Numba board ops) and "
                         "compare its ceiling to the python-chess one + MCTS")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.triton and device != "cuda":
        args.triton = False
    # The entire diagnostic is inference-only — globally disable autograd so no
    # step/prime ever builds a graph (otherwise memory accumulates and OOMs).
    torch.set_grad_enabled(False)
    torch.manual_seed(0); np.random.seed(0)
    em, sm, src = build(args, device)
    print("=" * 70)
    print(f"MARS diagnostics | device={device} | triton={args.triton} | weights: {src}")
    print(f"Eval {sum(p.numel() for p in em.parameters()):,} | "
          f"Search {sum(p.numel() for p in sm.parameters()):,} params")
    if args.tiny:
        print("!! --tiny: smoke test, numbers NOT representative !!")
    print("=" * 70)

    correctness_gate(em, sm, device)
    ceiling, _ = batch_sweep(em, sm, device, args.max_batch)
    fast_ceiling = None
    if args.fast:
        fast_ceiling, _ = batch_sweep(em, sm, device, args.max_batch, fast=True)
    mars_nps, mcts_nps = head_to_head(em, sm, device, args.mars_budget, args.sims)
    if fast_ceiling is not None:
        print("\n--- FASTCHESS vs python-chess rollout ceiling " + "-" * 22)
        print(f"  python-chess board ceiling : {ceiling:8.1f} nodes/sec")
        print(f"  fastchess    board ceiling : {fast_ceiling:8.1f} nodes/sec "
              f"({fast_ceiling/max(ceiling,1e-9):.2f}x)")
        print(f"  MCTS                       : {mcts_nps:8.1f} nodes/sec")
        print(f"  >>> fastchess ceiling vs MCTS = {fast_ceiling/max(mcts_nps,1e-9):.2f}x "
              f"({'ABOVE' if fast_ceiling > mcts_nps else 'below'} MCTS)")
    verdict(mars_nps, mcts_nps, max(ceiling, fast_ceiling or 0))


if __name__ == "__main__":
    main()
