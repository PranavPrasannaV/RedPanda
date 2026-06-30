"""
Stage-0 search benchmark: MARS vs MCTS with a fixed Stockfish-depth-1 ORACLE
plugged in place of the neural networks. This isolates the SEARCH ALGORITHM
(no trained nets needed) — does MARS's Gumbel + sequential-halving + recurrent-PV
structure pick better moves than MCTS's PUCT at an equal ORACLE-CALL budget?

Fairness: both searches call the SAME oracle; we count oracle evaluations per
position and report solve-rate alongside calls + time, so the comparison is
"strength per oracle call", not raw wall-clock.

Usage:
    python bench.py --stockfish /usr/games/stockfish
    python bench.py --stockfish C:\\path\\to\\stockfish.exe --sims 200 --mars-budget 20
"""

import os
import sys
import time
import argparse
import chess
from oracle import StockfishOracle, OracleMCTS, OracleMARS

# Win-At-Chess tactics: (FEN, best move uci). Each has a single clear best move.
WAC_FENS = [
    ("2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1", "g3g6"),
    ("8/7p/5k2/5p2/p1p2P2/Pr1pPK2/1P1R3P/8 b - - 0 1", "b3b2"),
    ("5rk1/1ppb3p/p1pb4/6q1/3P1p1r/2P1R2P/PP1B1QPN/5R1K w - - 0 1", "e3g3"),
    ("r1bq2k1/2p1b1pp/p1n5/1p1pP2Q/3Pn3/2P5/PP4PP/RNB2RK1 w - - 0 1", "h5f7"),
    ("2r2rk1/1bqnbpp1/1p1ppn1p/pP6/N1P1P3/P2B1N1P/1B2QPP1/R2R2K1 b - - 0 1", "b7e4"),
    ("r1bqk2r/pp2bppp/2p5/3pP3/P2Q1P2/2N1B3/1PP3PP/R4RK1 b kq - 0 1", "f7f6"),
    ("r2qnrnk/p2b2b1/1p1p2pp/2pPpp2/1PP1P3/PRNBB3/3QNPPP/5RK1 w - - 0 1", "f2f4"),
    ("2r3k1/1p2q1pp/2b1pr2/p1pp4/6Q1/1P1PP1R1/P1PN2PP/5RK1 w - - 0 1", "f1f6"),
    ("r1b2rk1/pp1n1ppp/2p1p3/q5B1/2BP4/P1n1PN2/5PPP/R3QRK1 w - - 0 1", "g5e7"),
    ("2rq1rk1/pb1n1ppN/1p2p3/3p4/2pP1P2/b1P1P1P1/PPQN2P1/1K1R3R w - - 0 1", "h7f8"),
]


def run_benchmark(sf_path, sims, mars_budget, depth_cap, sf_depth, mars_temp):
    print(f"Initializing Stockfish oracle (depth {sf_depth})...")
    oracle = StockfishOracle(sf_path, depth=sf_depth)

    mcts = OracleMCTS(oracle, num_simulations=sims, batch_size=8,
                      use_tablebase=False, add_root_noise=False)
    mars = OracleMARS(oracle, sim_budget=mars_budget, m_root=8, depth_cap=depth_cap,
                      rollout_temp=mars_temp, use_tablebase=False)

    res = {"mcts": {"solved": 0, "time": 0.0, "calls": 0},
           "mars": {"solved": 0, "time": 0.0, "calls": 0}}

    print("\n--- Stage 0: MARS vs MCTS (search algorithm only, Stockfish oracle) ---")
    for i, (fen, best) in enumerate(WAC_FENS):
        print(f"\n[{i+1}/{len(WAC_FENS)}] target {best}")
        for name, eng in (("mcts", mcts), ("mars", mars)):
            oracle.calls = 0
            t0 = time.time()
            mv = eng.search(chess.Board(fen))
            dt = time.time() - t0
            ok = mv is not None and mv.uci() == best
            res[name]["solved"] += int(ok)
            res[name]["time"] += dt
            res[name]["calls"] += oracle.calls
            print(f"  {name.upper():4s} {mv.uci() if mv else 'None':6s} "
                  f"{'OK ' if ok else 'xx '} | {oracle.calls:5d} calls | {dt:5.2f}s")

    n = len(WAC_FENS)
    print("\n================= RESULTS (solve-rate per oracle call) =================")
    for name in ("mcts", "mars"):
        r = res[name]
        print(f"  {name.upper():4s}  solved {r['solved']:2d}/{n}  | "
              f"avg {r['calls']//n:5d} oracle-calls/pos | {r['time']/n:5.2f}s/pos")
    print("========================================================================")
    print("Read as: higher solve-rate at FEWER oracle-calls = the better search.")
    oracle.close()


def _find_stockfish(explicit):
    import shutil
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Real files first (cwd-independent), then PATH as a last resort.
    candidates = [
        explicit,
        os.environ.get("STOCKFISH_PATH"),
        os.path.join(repo, "stockfish", "stockfish-windows-x86-64-avx2.exe"),
        os.path.join(repo, "stockfish", "stockfish"),
        "/usr/games/stockfish", "/usr/local/bin/stockfish", "/usr/bin/stockfish",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return shutil.which("stockfish")   # None if not on PATH


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stockfish", default=None, help="path to Stockfish binary (or set STOCKFISH_PATH)")
    ap.add_argument("--sims", type=int, default=100, help="MCTS simulations")
    ap.add_argument("--mars-budget", type=int, default=20, help="MARS root simulation budget")
    ap.add_argument("--depth-cap", type=int, default=8, help="MARS rollout depth cap")
    ap.add_argument("--sf-depth", type=int, default=1, help="Stockfish oracle depth")
    ap.add_argument("--mars-temp", type=float, default=0.6,
                    help="MARS rollout temperature (lower = greedier PV following)")
    args = ap.parse_args()

    sf = _find_stockfish(args.stockfish)
    if sf is None:
        print("Stockfish not found. Pass --stockfish <path> or set STOCKFISH_PATH.")
        sys.exit(1)
    run_benchmark(sf, args.sims, args.mars_budget, args.depth_cap, args.sf_depth, args.mars_temp)
