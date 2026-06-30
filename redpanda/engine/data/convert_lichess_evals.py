"""
Convert the Lichess pre-evaluated positions dataset -> ChessMamba v3 training
arrays.  This is the ONE supported data path (the Kaggle dataset already carries
Stockfish analysis for every position, so no local engine is needed).

Key v3 changes vs v2:
  - Side-to-move-RELATIVE values (v2 stored White-relative WDL, which silently
    broke MCTS value back-propagation).  cp is negated when Black is to move.
  - Soft policy: consecutive rows sharing a FEN (multi-PV) are merged into a
    top-K move distribution weighted by eval; single-line FENs reduce to one-hot.
  - Compact SPARSE policy on disk: policy_moves.npy (N,K) + policy_probs.npy
    (N,K).  A dense 10Mx8192 float array would be 327 GB - infeasible.
  - Strategy labels (detect_themes) and a phase id are saved per position, so the
    strategy head can be trained and geometric advantage vectors can be bucketed.
  - FEN de-duplication (keeps first/​deepest) and game-phase stratification
    (~30 % opening / 50 % middlegame / 20 % endgame).
  - Streams to fixed-width int16 memmaps -> 10M+ positions fit on a laptop / Kaggle.

Usage (Kaggle CPU session):
    python convert_lichess_evals.py \
        --input /kaggle/input/chess-evaluations/ \
        --output /kaggle/working/training_data \
        --max-positions 5000000 --min-depth 20
"""

import argparse
import os
import sys
import json
import math
import time
import glob
import gc
import shutil
import multiprocessing as mp
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import chess
from encoding import encoder, ACTION_SPACE, PHASE_OPENING, PHASE_MIDDLEGAME, PHASE_ENDGAME
from strategy import detect_themes, themes_to_vector, NUM_THEMES

SEQ_LEN = 160          # fixed padded board-encoding width (max real len ~ 107)
TOP_K = 5              # max moves kept in the soft policy
POLICY_TEMP = 100.0    # centipawn temperature for the soft policy softmax


def cp_to_wdl(cp: float) -> list:
    """
    Side-to-move-relative centipawns -> [W, D, L], ANTISYMMETRIC in cp.

    We model the expected score p = sigmoid(cp/scale) (so p(-cp)=1-p(cp)) and a
    |cp|-decreasing draw rate, then split:  W = p - d/2,  L = (1-p) - d/2.
    This guarantees value = W - L = 2p - 1 flips sign with cp - essential so that
    MCTS's negate-up-the-tree value back-prop is sound.
    """
    p = 1.0 / (1.0 + math.exp(-cp / 350.0))         # expected score
    d = max(0.0, 0.40 * (1.0 - abs(cp) / 600.0))    # draws fade with |cp|
    d = min(d, 2.0 * min(p, 1.0 - p))               # keep W, L >= 0
    win = p - d / 2.0
    loss = (1.0 - p) - d / 2.0
    return [win, d, loss]


def mate_to_cp(mate: int) -> float:
    return (10000 - mate * 10) if mate > 0 else (-10000 - mate * 10)


def _fen_key(board: chess.Board) -> str:
    """Transposition-aware key (ignores move counters)."""
    p = board.fen().split()
    return " ".join(p[:4])


class _Accumulator:
    """Collects consecutive rows of one FEN into a soft-policy position."""

    def __init__(self, board, white_to_move):
        self.board = board
        self.white_to_move = white_to_move
        self.move_cp = {}      # first_move_uci -> best (side-to-move) cp seen
        self.best_pv = []      # PV of the best line so far
        self.best_cp = -1e18

    def add(self, first_move, stm_cp, pv):
        if first_move not in self.move_cp or stm_cp > self.move_cp[first_move]:
            self.move_cp[first_move] = stm_cp
        if stm_cp > self.best_cp:
            self.best_cp = stm_cp
            self.best_pv = pv

    def finalize(self):
        """Return (input_ids, policy_moves, policy_probs, wdl, cp, phase, strat, pv)."""
        moves = list(self.move_cp.items())                  # [(uci, stm_cp)]
        moves.sort(key=lambda x: x[1], reverse=True)
        moves = moves[:TOP_K]

        move_ids, cps = [], []
        for uci, c in moves:
            mid = encoder.encode_move(uci)
            if mid is not None and mid < ACTION_SPACE:
                move_ids.append(mid)
                cps.append(c)
        if not move_ids:
            return None

        cps = np.array(cps, dtype=np.float64)
        w = np.exp((cps - cps.max()) / POLICY_TEMP)
        probs = (w / w.sum()).astype(np.float32)

        try:
            input_ids = encoder.encode_board(self.board)
        except Exception:
            return None

        best_cp = float(cps[0])
        wdl = cp_to_wdl(best_cp)
        phase = encoder.phase_index(self.board)
        strat = themes_to_vector(detect_themes(self.board))
        # FEN is stored so the v4 dynamics trainer can replay the PV and encode
        # each intermediate position for the consistency target.
        return (input_ids, move_ids, probs.tolist(), wdl, best_cp, phase, strat,
                self.best_pv, self.board.fen())


def _phase_targets(max_positions):
    return {
        PHASE_OPENING: int(0.30 * max_positions),
        PHASE_MIDDLEGAME: int(0.55 * max_positions),
        PHASE_ENDGAME: int(0.20 * max_positions),
    }


PARQUET_COLS = ["fen", "line", "depth", "cp", "mate"]


def _iter_parquet(data_file, min_depth):
    """
    Robust parquet reader. Tries, in order:
      1. pyarrow iter_batches (streams data pages; avoids the row-group statistics
         path that raises 'Repetition level histogram size mismatch' on some
         pyarrow versions),
      2. per-row-group read (skipping any group that fails),
      3. pandas (pyarrow then fastparquet engines).
    Whatever rows leak through dedup downstream are harmless (FEN de-dup).
    Unreadable files are SKIPPED with a warning instead of crashing the run.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow", "-q"])
        import pyarrow.parquet as pq

    # Strategy 1: streamed record batches.
    try:
        pf = pq.ParquetFile(data_file)
        for batch in pf.iter_batches(batch_size=20000, columns=PARQUET_COLS):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                yield _parse_row(row, min_depth)
            del df
            gc.collect()
        return
    except Exception as e:
        print(f"    iter_batches failed ({type(e).__name__}: {e}); trying per-row-group...")

    # Strategy 2: per-row-group, skipping the bad ones.
    got = False
    try:
        pf = pq.ParquetFile(data_file)
        for rg in range(pf.metadata.num_row_groups):
            try:
                df = pf.read_row_group(rg, columns=PARQUET_COLS).to_pandas()
            except Exception as e:
                print(f"      skip row-group {rg} ({type(e).__name__})")
                continue
            got = True
            for _, row in df.iterrows():
                yield _parse_row(row, min_depth)
            del df
            gc.collect()
        if got:
            return
    except Exception as e:
        print(f"    per-row-group failed ({type(e).__name__}: {e})")

    # Strategy 3: pandas engines (fastparquet is a different decoder entirely).
    import pandas as pd
    for eng in ("pyarrow", "fastparquet"):
        try:
            df = pd.read_parquet(data_file, columns=PARQUET_COLS, engine=eng)
        except Exception as e:
            print(f"    pandas[{eng}] failed ({type(e).__name__}: {e})")
            continue
        for _, row in df.iterrows():
            yield _parse_row(row, min_depth)
        del df
        gc.collect()
        return
    print(f"    !! SKIPPING unreadable file: {data_file}")


def _row_iter(data_files, min_depth):
    """Yield (fen, first_move_uci, stm_cp, pv_ucis) from all data files."""
    for data_file in data_files:
        fname = os.path.basename(data_file)
        print(f"\n  Reading: {fname}")
        if data_file.endswith(".parquet"):
            for rec in _iter_parquet(data_file, min_depth):
                yield rec
        elif data_file.endswith(".csv"):
            import pandas as pd
            for chunk in pd.read_csv(data_file, chunksize=50000):
                for _, row in chunk.iterrows():
                    yield _parse_row(row, min_depth)
                del chunk
                gc.collect()
        elif data_file.endswith((".jsonl", ".jsonl.zst")):
            for rec in _jsonl_iter(data_file, min_depth):
                yield rec


def _parse_row(row, min_depth):
    try:
        if int(row.get("depth")) < min_depth:
            return None
    except (ValueError, TypeError):
        return None
    cp, mate = row.get("cp"), row.get("mate")
    has_cp = cp is not None and not (isinstance(cp, float) and math.isnan(cp))
    has_mate = mate is not None and not (isinstance(mate, float) and math.isnan(mate))
    if not has_cp and not has_mate:
        return None
    fen = str(row.get("fen"))
    parts = fen.split()
    if len(parts) == 4:
        fen += " 0 1"
    try:
        board = chess.Board(fen)
    except Exception:
        return None
    line = row.get("line")
    pv = str(line).split() if line and str(line) != "nan" else []
    if not pv:
        return None
    white_cp = float(cp) if has_cp else mate_to_cp(int(mate))
    stm_cp = white_cp if board.turn == chess.WHITE else -white_cp
    return (board, _fen_key(board), pv[0], stm_cp, pv[:12])


def _jsonl_iter(path, min_depth):
    """Native Lichess eval JSONL: one record per FEN with multi-PV `evals`."""
    if path.endswith(".zst"):
        import zstandard as zstd, io
        fh = open(path, "rb")
        reader = zstd.ZstdDecompressor().stream_reader(fh)
        stream = io.TextIOWrapper(reader, encoding="utf-8")
    else:
        stream = open(path, "r", encoding="utf-8")
    try:
        for raw in stream:
            try:
                rec = json.loads(raw)
            except Exception:
                continue
            fen = rec.get("fen")
            if not fen:
                continue
            try:
                board = chess.Board(fen if len(fen.split()) == 6 else fen + " 0 1")
            except Exception:
                continue
            best = None
            for ev in rec.get("evals", []):
                if ev.get("depth", 0) < min_depth:
                    continue
                if best is None or ev.get("depth", 0) > best.get("depth", 0):
                    best = ev
            if not best:
                continue
            key = _fen_key(board)
            for pv in best.get("pvs", [])[:TOP_K]:
                line = pv.get("line", "")
                moves = line.split()
                if not moves:
                    continue
                if "cp" in pv:
                    white_cp = float(pv["cp"])
                elif "mate" in pv:
                    white_cp = mate_to_cp(int(pv["mate"]))
                else:
                    continue
                stm_cp = white_cp if board.turn == chess.WHITE else -white_cp
                yield (board, key, moves[0], stm_cp, moves[:12])
    finally:
        stream.close()


# Temp-memmap spec: name -> (filename, dtype, per-row trailing shape)
_TMP = {
    "inputs": ("_inputs_tmp.npy", np.int16, (SEQ_LEN,)),
    "pmoves": ("_pmoves_tmp.npy", np.int32, (TOP_K,)),
    "pprobs": ("_pprobs_tmp.npy", np.float32, (TOP_K,)),
    "wdl": ("_wdl_tmp.npy", np.float32, (3,)),
    "cps": ("_cps_tmp.npy", np.float32, ()),
    "phase": ("_phase_tmp.npy", np.int8, ()),
    "strat": ("_strat_tmp.npy", np.float32, (NUM_THEMES,)),
}
_FINAL = {
    "inputs": "inputs.npy", "pmoves": "policy_moves.npy", "pprobs": "policy_probs.npy",
    "wdl": "wdl.npy", "cps": "centipawns.npy", "phase": "phases.npy",
    "strat": "strategy_labels.npy",
}


def _cleanup_tmp(out_dir):
    for fname, _, _ in _TMP.values():
        p = os.path.join(out_dir, fname)
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def _iter_records(data_files, min_depth):
    """
    Yield finalized 9-tuple records (one per FEN), grouping CONSECUTIVE same-FEN
    rows (multi-PV) into a single soft-policy position. De-duplication and
    stratification are applied by the writer, not here.
    """
    acc, acc_key = None, None
    for parsed in _row_iter(data_files, min_depth):
        if parsed is None:
            continue
        board, key, first_move, stm_cp, pv = parsed
        if key != acc_key:
            if acc is not None:
                rec = acc.finalize()
                if rec is not None:
                    yield rec
            acc = _Accumulator(board, board.turn == chess.WHITE)
            acc_key = key
        acc.add(first_move, stm_cp, pv)
    if acc is not None:
        rec = acc.finalize()
        if rec is not None:
            yield rec


def _write_dataset(records, out_dir, cap, stratify=True, dedup=True,
                   label="convert", write_meta=True, min_depth=20):
    """
    Consume `records` into a ChessMamba dataset folder (the final arrays), applying
    optional FEN de-dup + phase stratification, capped at `cap` positions.
    Returns (num_positions, phase_counts).
    """
    os.makedirs(out_dir, exist_ok=True)
    mm = {k: np.lib.format.open_memmap(os.path.join(out_dir, fn), mode="w+",
                                       dtype=dt, shape=(cap,) + sh)
          for k, (fn, dt, sh) in _TMP.items()}
    pv_lines, fens = [], []
    phase_caps = _phase_targets(cap) if stratify else None
    phase_counts = {PHASE_OPENING: 0, PHASE_MIDDLEGAME: 0, PHASE_ENDGAME: 0}
    seen = set() if dedup else None
    n = skipped = last = 0
    start = time.time()

    for rec in records:
        if n >= cap:
            break
        if rec is None:
            skipped += 1
            continue
        input_ids, move_ids, probs, wdl, cp_val, phase, strat, pv, fen = rec
        phase = int(phase)
        if seen is not None:
            fkey = " ".join(str(fen).split()[:4])
            if fkey in seen:
                skipped += 1
                continue
            seen.add(fkey)
        if phase_caps is not None and phase_counts[phase] >= phase_caps[phase]:
            continue
        row = np.zeros(SEQ_LEN, dtype=np.int16)
        L = min(len(input_ids), SEQ_LEN)
        row[:L] = np.asarray(input_ids[:L], dtype=np.int16)
        mm["inputs"][n] = row
        pm = np.full(TOP_K, -1, dtype=np.int32)
        pp = np.zeros(TOP_K, dtype=np.float32)
        k = min(len(move_ids), TOP_K)
        pm[:k] = np.asarray(move_ids[:k], dtype=np.int32)
        pp[:min(len(probs), TOP_K)] = np.asarray(probs[:TOP_K], dtype=np.float32)
        mm["pmoves"][n] = pm
        mm["pprobs"][n] = pp
        mm["wdl"][n] = wdl
        mm["cps"][n] = cp_val
        mm["phase"][n] = phase
        mm["strat"][n] = strat
        pv_lines.append(pv)
        fens.append(str(fen))
        phase_counts[phase] += 1
        n += 1
        if n % 100000 == 0 and n != last:
            last = n
            el = time.time() - start
            r = n / el if el else 0
            print(f"    [{label}] {n:,}/{cap:,} | {r:.0f} pos/s | "
                  f"O/M/E {phase_counts[0]:,}/{phase_counts[1]:,}/{phase_counts[2]:,} | "
                  f"skipped {skipped:,}", flush=True)

    if n == 0:
        del mm
        gc.collect()
        _cleanup_tmp(out_dir)
        return 0, phase_counts

    for k, fname in _FINAL.items():
        dt = _TMP[k][1]
        _finalize(out_dir, fname, mm[k], n, np.int8 if k == "phase" else dt)
    best = np.array(mm["pmoves"][:n, 0], dtype=np.int32, copy=True)
    np.save(os.path.join(out_dir, "best_moves.npy"), best)
    json.dump(pv_lines, open(os.path.join(out_dir, "pv_lines.json"), "w"))
    json.dump(fens, open(os.path.join(out_dir, "start_fens.json"), "w"))

    del mm
    gc.collect()
    _cleanup_tmp(out_dir)

    if write_meta:
        json.dump({
            "num_positions": n, "seq_len": SEQ_LEN, "action_space": ACTION_SPACE,
            "top_k": TOP_K, "num_themes": NUM_THEMES,
            "values_relative_to": "side_to_move",
            "phase_counts": {"opening": phase_counts[PHASE_OPENING],
                             "middlegame": phase_counts[PHASE_MIDDLEGAME],
                             "endgame": phase_counts[PHASE_ENDGAME]},
            "min_depth": min_depth, "source": "lichess-evaluations",
        }, open(os.path.join(out_dir, "metadata.json"), "w"), indent=2)
    return n, phase_counts


def _iter_shard_records(shard_dirs, cleanup=False):
    """
    Yield records reconstructed from worker shard datasets (for the merge).
    If cleanup, each shard dir is deleted once consumed — this frees its disk
    as the merge progresses, roughly halving peak disk usage.
    """
    for sd in shard_dirs:
        try:
            inputs = np.load(os.path.join(sd, "inputs.npy"), mmap_mode="r")
            pmoves = np.load(os.path.join(sd, "policy_moves.npy"), mmap_mode="r")
            pprobs = np.load(os.path.join(sd, "policy_probs.npy"), mmap_mode="r")
            wdl = np.load(os.path.join(sd, "wdl.npy"), mmap_mode="r")
            cps = np.load(os.path.join(sd, "centipawns.npy"), mmap_mode="r")
            phase = np.load(os.path.join(sd, "phases.npy"), mmap_mode="r")
            strat = np.load(os.path.join(sd, "strategy_labels.npy"), mmap_mode="r")
            pv = json.load(open(os.path.join(sd, "pv_lines.json")))
            fens = json.load(open(os.path.join(sd, "start_fens.json")))
        except FileNotFoundError:
            continue
        for i in range(len(inputs)):
            pm = pmoves[i]
            mids = pm[pm >= 0].tolist()
            probs = pprobs[i][:len(mids)].tolist()
            yield (inputs[i], mids, probs, wdl[i], float(cps[i]),
                   int(phase[i]), strat[i], pv[i], fens[i])
        del inputs, pmoves, pprobs, wdl, cps, phase, strat
        gc.collect()
        if cleanup:
            shutil.rmtree(sd, ignore_errors=True)


def _convert_worker(args):
    wid, files, shard_dir, min_depth, cap = args
    n, _ = _write_dataset(_iter_records(files, min_depth), shard_dir, cap,
                          stratify=False, dedup=True, label=f"w{wid}",
                          write_meta=False, min_depth=min_depth)
    return wid, n


def convert_lichess_evals(input_dir, output_dir, max_positions=5_000_000,
                          min_depth=20, stratify=True, workers=1):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 64)
    print("ChessMamba v4 - Lichess Eval Converter")
    print("=" * 64)
    print(f"  Input:   {input_dir}")
    print(f"  Output:  {output_dir}")
    print(f"  Target:  {max_positions:,} positions | min_depth={min_depth} | "
          f"stratify={stratify} | workers={workers}")

    # OneDrive's sync filter breaks memory-mapped file creation (OSError 22).
    if "onedrive" in os.path.abspath(output_dir).lower():
        print("\n  !! WARNING: output is inside a OneDrive folder. Memory-mapped\n"
              "     writes there often fail with '[Errno 22] Invalid argument'.\n"
              "     Use a local non-synced path, e.g. --output C:\\chessdata\\training\n")

    data_files = []
    for ext in ("*.parquet", "*.jsonl", "*.jsonl.zst", "*.csv"):
        data_files.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))
    if not data_files:
        print("ERROR: no data files found!")
        return
    data_files = sorted(data_files)
    print(f"  Found {len(data_files)} data files")
    start = time.time()

    if workers <= 1 or len(data_files) == 1:
        n, pc = _write_dataset(_iter_records(data_files, min_depth), output_dir,
                               max_positions, stratify=stratify, dedup=True,
                               label="convert", min_depth=min_depth)
    else:
        # ── Multicore: each worker converts a file subset into a shard, then a
        #    single merge pass does the global FEN de-dup + phase stratification. ──
        workers = min(workers, len(data_files))
        per_worker = math.ceil(max_positions * 1.5 / workers)  # slack for dedup/strat
        buckets = [[] for _ in range(workers)]
        for i, f in enumerate(data_files):
            buckets[i % workers].append(f)
        shard_dirs = [os.path.join(output_dir, f"_shard{w}") for w in range(workers)]
        for sd in shard_dirs:
            shutil.rmtree(sd, ignore_errors=True)
        tasks = [(w, buckets[w], shard_dirs[w], min_depth, per_worker)
                 for w in range(workers)]
        print(f"  Launching {workers} workers (cap {per_worker:,}/worker)...")
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            results = pool.map(_convert_worker, tasks)
        produced = sum(r[1] for r in results)
        print(f"\n  Workers produced {produced:,} records; merging "
              f"(global de-dup + stratify)...")
        n, pc = _write_dataset(_iter_shard_records(shard_dirs, cleanup=True),
                               output_dir, max_positions, stratify=stratify,
                               dedup=True, label="merge", min_depth=min_depth)
        for sd in shard_dirs:
            shutil.rmtree(sd, ignore_errors=True)

    if n == 0:
        print("ERROR: no positions converted!")
        return
    size_mb = sum(os.path.getsize(os.path.join(output_dir, f))
                  for f in os.listdir(output_dir)
                  if os.path.isfile(os.path.join(output_dir, f))) / 1e6
    dt = time.time() - start
    print("=" * 64)
    print(f"Done: {n:,} positions | {dt/60:.1f} min | {size_mb:.0f} MB on disk")
    print(f"  phases O/M/E: {pc[PHASE_OPENING]:,} / {pc[PHASE_MIDDLEGAME]:,} / "
          f"{pc[PHASE_ENDGAME]:,}")
    print("=" * 64)


def _finalize(output_dir, name, tmp_mm, n, dtype):
    """Copy the first n rows of an oversized temp memmap into a tight .npy."""
    shape = (n,) if tmp_mm.ndim == 1 else (n, tmp_mm.shape[1])
    out = np.lib.format.open_memmap(
        os.path.join(output_dir, name), mode="w+", dtype=dtype, shape=shape)
    CH = 200000
    for s in range(0, n, CH):
        e = min(s + CH, n)
        out[s:e] = tmp_mm[s:e]
    out.flush()
    del out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="/kaggle/working/training_data")
    parser.add_argument("--max-positions", type=int, default=5_000_000)
    parser.add_argument("--min-depth", type=int, default=20)
    parser.add_argument("--no-stratify", action="store_true")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel worker processes (0 = cpu_count-1). "
                             "Each converts a file subset; results are merged.")
    args = parser.parse_args()

    workers = args.workers
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 2) - 1)

    convert_lichess_evals(
        input_dir=args.input,
        output_dir=args.output,
        max_positions=args.max_positions,
        min_depth=args.min_depth,
        stratify=not args.no_stratify,
        workers=workers,
    )
