
"""
Asynchronous multi-game self-play generation (Phase 4.3).

On a single 8 GB GPU you cannot fit many independent model replicas, so the
practical throughput levers are:
    1. MCTS leaf batching (already inside BatchedMCTS - raise --batch-size), and
    2. a small pool of worker processes each generating a slice of games.

This launches `--workers` processes (each runs engine.self_play.generate over a
disjoint shard-id range) so games are produced in parallel and written as .npz
shards into one replay directory. Default workers=1 (GPU); raise to 2 only if
VRAM allows, or run workers on CPU for cheap extra games.

Usage:
    python async_self_play.py --model chess_mamba.pt --games 800 --workers 2 \
        --sims 400 --out selfplay --device cuda
"""

import os
import argparse
import multiprocessing as mp

from self_play import generate


def _worker(args):
    (model, out, games, sims, batch_size, device, seed_offset) = args
    generate(model, out, num_games=games, sims=sims, device=device,
             batch_size=batch_size, seed_offset=seed_offset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="chess_mamba.pt")
    ap.add_argument("--out", default="selfplay")
    ap.add_argument("--games", type=int, default=400, help="total games across workers")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    per = max(1, args.games // args.workers)
    # Disjoint shard-id ranges so workers never overwrite each other's shards.
    tasks = [
        (args.model, args.out, per, args.sims, args.batch_size, args.device,
         w * 100000)
        for w in range(args.workers)
    ]

    if args.workers == 1:
        _worker(tasks[0])
        return

    mp.set_start_method("spawn", force=True)
    procs = [mp.Process(target=_worker, args=(t,)) for t in tasks]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print(f"All {args.workers} workers finished -> {args.out}")


if __name__ == "__main__":
    main()
