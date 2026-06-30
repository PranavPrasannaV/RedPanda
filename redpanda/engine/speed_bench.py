"""
MARS vs MCTS speed benchmark + per-node Python-overhead profile (B11).

THE point of this tool: MARS's core thesis is SPEED — the O(1) recurrent
step() makes per-node evaluation far cheaper than a full Eval-Mamba forward,
so at equal wall-clock MARS searches more. Speed is WEIGHT-INDEPENDENT
(latency does not care how trained the nets are), so this benchmark gives a
100% valid verdict on the thesis even with random or 1-epoch nets.

What it does NOT measure: playing strength. That needs trained nets and the
tournament (Stage-2). Do not read any strength meaning into this output.

Usage (on the training GPU):
    python speed_bench.py --model chess_mamba.pt --search search_mamba.pt --triton
    python speed_bench.py --random            # no checkpoints needed (full-size nets)
    python speed_bench.py --random --tiny     # CPU smoke test (CI / laptop)

Outputs:
  1. Micro-latencies: python-chess ops, encode_board, zobrist, Eval forward
     (B=1 and B=8), Search-Mamba prime + step. And R = eval_forward / sm_step —
     the raw thesis number.
  2. Real-search throughput: nodes/sec for MARS and MCTS on test positions,
     with an estimated GPU-vs-Python time split per search (the B11 profile).
"""

import os
import sys
import time
import argparse
import statistics

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
from mcgs import zobrist_key

# Mixed test positions (opening / middlegame / tactic / endgame).
TEST_FENS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r2qnrnk/p2b2b1/1p1p2pp/2pPpp2/1PP1P3/PRNBB3/3QNPPP/5RK1 w - - 0 1",
    "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1",
    "8/7p/5k2/5p2/p1p2P2/Pr1pPK2/1P1R3P/8 b - - 0 1",
    "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1",
]


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def timeit(fn, device, warmup=3, reps=10):
    """Median seconds per call (cuda-synchronized)."""
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def set_triton(module, on):
    """Force use_triton_scan on every MambaConfig reachable from `module`."""
    seen = set()
    for m in module.modules():
        cfg = getattr(m, "config", None)
        if cfg is not None and hasattr(cfg, "use_triton_scan") and id(cfg) not in seen:
            cfg.use_triton_scan = on
            seen.add(id(cfg))


def build_models(args, device):
    vocab = encoder.vocab_size() + 100
    if args.random:
        if args.tiny:
            ecfg = MambaConfig(d_model=64, n_layer=2, d_state=16, n_track_state=4,
                               mimo_p=2, vocab_size=vocab, bidirectional=True)
            scfg = SearchMambaConfig(d_model=48, n_layer=2, d_state=16, n_track_state=4,
                                     mimo_p=2, vocab_size=vocab, bidirectional=False)
        else:
            # Full production sizes — what the real engine runs.
            ecfg = MambaConfig(d_model=512, n_layer=16, d_state=64, n_track_state=16,
                               mimo_p=4, vocab_size=vocab, bidirectional=True)
            scfg = SearchMambaConfig(d_model=384, n_layer=12, d_state=32, n_track_state=16,
                                     mimo_p=2, vocab_size=vocab, bidirectional=False)
        eval_model = ChessMamba(ecfg, action_space=ACTION_SPACE).to(device).eval()
        search_model = SearchMamba(scfg, vocab_size=vocab,
                                   eval_dim=ecfg.d_model).to(device).eval()
        src = "RANDOM weights (valid: speed is weight-independent)"
    else:
        from self_play import load_model
        eval_model, _ = load_model(args.model, device=device)
        search_model = SearchMamba.from_pretrained(args.search, device=device)
        search_model.to(device).eval()
        src = f"checkpoints ({args.model}, {args.search})"
    set_triton(eval_model, args.triton)
    set_triton(search_model, args.triton)
    return eval_model, search_model, src


def micro_bench(eval_model, search_model, device):
    print("\n--- 1. Micro-latencies (median) " + "-" * 36)
    board = chess.Board(TEST_FENS[2])

    t_encode = timeit(lambda: encoder.encode_board(board), device, reps=30)
    print(f"  encode_board                    {t_encode*1e3:8.3f} ms")

    moves = list(board.legal_moves)
    def push_pop():
        for m in moves[:10]:
            board.push(m)
            _ = board.is_check()
            board.pop()
    t_pp = timeit(push_pop, device, reps=30) / 10
    print(f"  push + is_check + pop (per op)  {t_pp*1e3:8.3f} ms")

    t_legal = timeit(lambda: list(board.legal_moves), device, reps=30)
    print(f"  legal_moves list                {t_legal*1e3:8.3f} ms")

    t_zob = timeit(lambda: zobrist_key(board), device, reps=30)
    print(f"  zobrist_key                     {t_zob*1e3:8.3f} ms")

    enc = encoder.encode_board(board)
    x1 = torch.tensor([enc], dtype=torch.long, device=device)
    x8 = x1.repeat(8, 1)
    with torch.no_grad():
        t_e1 = timeit(lambda: eval_model(x1, return_dict=True), device)
        t_e8 = timeit(lambda: eval_model(x8, return_dict=True), device)
    print(f"  Eval Mamba forward  B=1         {t_e1*1e3:8.3f} ms")
    print(f"  Eval Mamba forward  B=8         {t_e8*1e3:8.3f} ms  "
          f"({t_e8/8*1e3:.3f} ms/position batched)")

    mid = encoder.encode_move(moves[0])
    with torch.no_grad():
        t_prime = timeit(lambda: search_model.prime_board(enc), device)
        _, cache0 = search_model.prime_board(enc)
        t_step = timeit(lambda: search_model.eval_step(
            [(h.clone(), d.clone()) for (h, d) in cache0], mid), device)
    print(f"  Search Mamba prime_board        {t_prime*1e3:8.3f} ms")
    print(f"  Search Mamba eval_step (O(1))   {t_step*1e3:8.3f} ms")

    R = t_e1 / max(t_step, 1e-9)
    print(f"\n  >>> R = eval_forward / sm_step = {R:.1f}x")
    print("      (MARS's thesis number: each rollout ply costs 1/R of a full eval)")
    return {"t_e1": t_e1, "t_e8": t_e8, "t_step": t_step, "t_prime": t_prime,
            "t_encode": t_encode, "t_pp": t_pp, "R": R}


def search_bench(eval_model, search_model, micro, args, device):
    print("\n--- 2. Real-search throughput " + "-" * 38)
    mars = MARS(eval_model, search_model, sim_budget=args.mars_budget, m_root=8,
                depth_cap=8, use_tablebase=False, adaptive_sims=False,
                add_root_noise=False)
    mcts = BatchedMCTS(eval_model, num_simulations=args.sims, batch_size=8,
                       use_tablebase=False, add_root_noise=False)

    agg = {"mars_t": 0.0, "mars_plies": 0, "mars_steps": 0, "mars_evals": 0,
           "mars_anchors": 0, "mcts_t": 0.0, "mcts_evals": 0}
    for fen in TEST_FENS:
        b = chess.Board(fen)
        if b.is_game_over():
            continue
        # warmup once each on the first position only
        t0 = time.perf_counter(); mars.run_search(b); _sync(device)
        agg["mars_t"] += time.perf_counter() - t0
        s = mars.stats
        agg["mars_plies"] += s["rollout_plies"]
        agg["mars_steps"] += s["sm_steps"]
        agg["mars_evals"] += s["eval_positions"]
        agg["mars_anchors"] += s["anchors"]

        t0 = time.perf_counter(); mcts.run_search(b); _sync(device)
        agg["mcts_t"] += time.perf_counter() - t0
        agg["mcts_evals"] += mcts.eval_count

    n = len(TEST_FENS)
    # MARS: nodes = rollout plies (each ply = one new position examined).
    mars_nps = agg["mars_plies"] / max(agg["mars_t"], 1e-9)
    mcts_nps = agg["mcts_evals"] / max(agg["mcts_t"], 1e-9)
    print(f"  MARS : {agg['mars_t']/n:6.2f}s/move | {agg['mars_plies']} plies "
          f"({agg['mars_steps']} O(1) steps, {agg['mars_anchors']} anchors, "
          f"{agg['mars_evals']} eval-net positions)")
    print(f"         -> {mars_nps:8.1f} nodes/sec")
    print(f"  MCTS : {agg['mcts_t']/n:6.2f}s/move | {agg['mcts_evals']} eval-net positions")
    print(f"         -> {mcts_nps:8.1f} nodes/sec")
    print(f"\n  >>> equal-wall-clock node ratio MARS/MCTS = {mars_nps/max(mcts_nps,1e-9):.2f}x")

    # B11: estimated GPU vs Python split per search.
    mars_gpu = (agg["mars_evals"] * micro["t_e8"] / 8
                + agg["mars_steps"] * micro["t_step"]
                + agg["mars_anchors"] * micro["t_prime"])
    mcts_gpu = agg["mcts_evals"] * micro["t_e8"] / 8
    for name, gpu, tot in (("MARS", mars_gpu, agg["mars_t"]),
                           ("MCTS", mcts_gpu, agg["mcts_t"])):
        py = max(0.0, 1.0 - gpu / max(tot, 1e-9))
        print(f"  {name} time split (estimated): ~{(1-py)*100:4.0f}% GPU / "
              f"~{py*100:4.0f}% Python+overhead")
    print("  (B11: if Python share is large, batching/vectorizing — not the "
          "architecture — is the speed fix.)")
    print("\n  NOTE: this is a SPEED benchmark only. Strength comparisons need "
          "trained nets + the tournament.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="chess_mamba.pt")
    ap.add_argument("--search", default="search_mamba.pt")
    ap.add_argument("--random", action="store_true",
                    help="random-weight nets (no checkpoints needed; speed-valid)")
    ap.add_argument("--tiny", action="store_true",
                    help="tiny configs (CPU smoke test; NOT a real measurement)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--triton", action="store_true",
                    help="use the fused Triton scan for full forwards (CUDA only)")
    ap.add_argument("--sims", type=int, default=100, help="MCTS simulations/move")
    ap.add_argument("--mars-budget", type=int, default=24, help="MARS sim budget/move")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.triton and device != "cuda":
        print("[speed_bench] --triton ignored (no CUDA).")
        args.triton = False
    torch.manual_seed(0)

    eval_model, search_model, src = build_models(args, device)
    ne = sum(p.numel() for p in eval_model.parameters())
    ns = sum(p.numel() for p in search_model.parameters())
    print("=" * 70)
    print(f"ChessMamba speed bench | device={device} | triton={args.triton}")
    print(f"Eval {ne:,} params | Search {ns:,} params | weights: {src}")
    if args.tiny:
        print("!! --tiny is a smoke test: numbers are NOT representative !!")
    print("=" * 70)

    micro = micro_bench(eval_model, search_model, device)
    search_bench(eval_model, search_model, micro, args, device)


if __name__ == "__main__":
    main()
