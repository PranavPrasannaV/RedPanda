
"""
Self-play reinforcement learning (Phase 4).

Policy-iteration loop:
    PLAY   - the current model plays games against itself with MCTS. Each move
             records (position, MCTS visit distribution π, MCTS root value).
    LABEL  - at game end, every position gets a value target blending the MCTS
             value with the actual game outcome z (side-to-move relative):
                 target_v = 0.75 * mcts_value + 0.25 * z
    TRAIN  - fine-tune the network toward π (policy) and target_v (value).
    REPEAT - a stronger model generates better games -> stronger model.

Design choices (match the plan):
    - Temperature schedule: τ=1.0 for the first `temp_moves` plies (exploration),
      then τ->0 (exploitation). Root Dirichlet noise diversifies openings.
    - Resign: if P(loss) > resign_thresh for `resign_streak` plies, resign early.
    - Replay shards saved as .npz, consumable by SelfPlayDataset / train_iteration.

CLI:
    python self_play.py generate --model chess_mamba.pt --games 200 --sims 400 --out selfplay
    python self_play.py train    --model chess_mamba.pt --replay selfplay --epochs 2
"""

import os
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import chess

from encoding import encoder, ACTION_SPACE
from mamba import MambaConfig
from model import ChessMamba
from batched_mcts import BatchedMCTS

SEQ_LEN = 160
TOP_K = 16          # self-play policy keeps more moves than supervised data


# ─── Model loading ───────────────────────────────────────────────────────────

def load_model(model_path, config_path=None, device="cpu"):
    config_path = config_path or model_path.replace(".pt", "_config.json")
    if os.path.exists(config_path):
        # Use the saved (already vocab-padded) config verbatim - overriding
        # vocab_size here would un-pad it and break the checkpoint load.
        cd = json.load(open(config_path))
        cfg = MambaConfig(**{k: v for k, v in cd.items()
                             if k in MambaConfig.__dataclass_fields__})
    else:
        cfg = MambaConfig(d_model=512, n_layer=16, d_state=64, mimo_p=4,
                          vocab_size=encoder.vocab_size() + 100)
    model = ChessMamba(cfg, action_space=ACTION_SPACE).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model, cfg


# ─── One self-play game ──────────────────────────────────────────────────────

def play_game(mcts: BatchedMCTS, max_moves=200, temp_moves=30,
              resign_thresh=0.95, resign_streak=3, opening_random_plies=8):
    board = chess.Board()
    samples = []           # (input_ids, π_moves, π_probs, mcts_value, stm_is_white)
    resign_count = 0
    result = None

    for ply in range(max_moves):
        if board.is_game_over(claim_draw=True):
            break
        root = mcts.run_search(board)
        if not root.children:
            break

        tau = 1.0 if ply < temp_moves else 0.0
        pi = mcts.get_policy_target(root, temperature=max(tau, 1e-3))
        ids = np.zeros(SEQ_LEN, dtype=np.int16)
        enc = encoder.encode_board(board)
        ids[: min(len(enc), SEQ_LEN)] = enc[:SEQ_LEN]
        nz = np.nonzero(pi)[0]
        order = nz[np.argsort(pi[nz])[::-1]][:TOP_K]
        pm = np.full(TOP_K, -1, dtype=np.int32); pp = np.zeros(TOP_K, dtype=np.float32)
        pm[: len(order)] = order
        pp[: len(order)] = pi[order]
        samples.append((ids, pm, pp, mcts.root_value(root), board.turn == chess.WHITE))

        # Resign check from the root WDL (side-to-move loss prob).
        # `result` is stored WHITE-perspective: the side to move is the loser.
        if root.wdl is not None and float(root.wdl[2]) > resign_thresh:
            resign_count += 1
            if resign_count >= resign_streak:
                result = -1.0 if board.turn == chess.WHITE else 1.0
                break
        else:
            resign_count = 0

        # Move selection: sample early (diversify), argmax late.
        if ply < opening_random_plies or ply < temp_moves:
            move = mcts.search(board, temperature=1.0)
        else:
            move = max(root.children, key=lambda m: root.children[m].visit_count)
        if move is None:
            break
        board.push(move)

    # Final result (White perspective): +1 white win, -1 black win, 0 draw.
    if result is None:
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            z_white = 0.0
        else:
            z_white = 1.0 if outcome.winner == chess.WHITE else -1.0
    else:
        z_white = result   # already stored White-perspective at resign time

    out = []
    for ids, pm, pp, mcts_v, stm_white in samples:
        z = z_white if stm_white else -z_white
        target_v = 0.75 * float(mcts_v) + 0.25 * z
        out.append((ids, pm, pp, np.float32(np.clip(target_v, -1, 1))))
    return out, z_white


# ─── Generation ──────────────────────────────────────────────────────────────

def load_search(path, device="cpu"):
    from search_mamba import SearchMamba
    m = SearchMamba.from_pretrained(path, device=device)
    return m.to(device).eval()


def play_game_mars(mars, max_moves=200, temp_moves=30,
                   resign_thresh=0.95, resign_streak=3, opening_random_plies=8):
    """One self-play game driven by MARS; records value-equivalence targets."""
    import numpy as np
    board = chess.Board()
    samples = []
    resign_count = 0
    result = None
    for ply in range(max_moves):
        if board.is_game_over(claim_draw=True):
            break
        res = mars.run_search(board)
        if res["best"] is None:
            break
        tau = 1.0 if ply < temp_moves else 0.0
        pi = mars.get_policy_target(res, temperature=max(tau, 1e-3))
        ids = np.zeros(SEQ_LEN, dtype=np.int16)
        enc = encoder.encode_board(board)
        ids[: min(len(enc), SEQ_LEN)] = enc[:SEQ_LEN]
        nz = np.nonzero(pi)[0]
        order = nz[np.argsort(pi[nz])[::-1]][:TOP_K]
        pm = np.full(TOP_K, -1, dtype=np.int32); pp = np.zeros(TOP_K, dtype=np.float32)
        pm[: len(order)] = order; pp[: len(order)] = pi[order]
        samples.append((ids, pm, pp, float(res["root_value"]), board.turn == chess.WHITE))

        wdl = res["wdl"]
        if wdl is not None and float(wdl[2]) > resign_thresh:
            resign_count += 1
            if resign_count >= resign_streak:
                result = -1.0 if board.turn == chess.WHITE else 1.0
                break
        else:
            resign_count = 0

        # Move: sample by visits early (diversify), best late.
        edges = res["edges"]
        moves = list(edges.keys())
        if ply < max(opening_random_plies, temp_moves):
            visits = np.array([edges[m][0] for m in moves], dtype=np.float64)
            p = visits / visits.sum() if visits.sum() > 0 else None
            move = moves[int(np.random.choice(len(moves), p=p))] if p is not None else res["best"]
        else:
            move = res["best"]
        board.push(move)

    if result is None:
        outcome = board.outcome(claim_draw=True)
        z_white = 0.0 if (outcome is None or outcome.winner is None) else \
            (1.0 if outcome.winner == chess.WHITE else -1.0)
    else:
        z_white = result
    out = []
    for ids, pm, pp, mcts_v, stm_white in samples:
        z = z_white if stm_white else -z_white
        out.append((ids, pm, pp, np.float32(np.clip(0.75 * mcts_v + 0.25 * z, -1, 1))))
    return out, z_white


def generate(model_path, out_dir, num_games=200, sims=400, device="cpu",
             batch_size=16, c_puct=1.5, shard_size=2000, seed_offset=0,
             backend="mcts", search_path=None):
    os.makedirs(out_dir, exist_ok=True)
    model, _ = load_model(model_path, device=device)
    use_mars = backend == "mars" and search_path and os.path.exists(search_path)
    if use_mars:
        from mars_search import MARS
        mars = MARS(model, load_search(search_path, device), sim_budget=sims,
                    add_root_noise=True, adaptive_sims=False, use_tablebase=True)
        game_fn = lambda: play_game_mars(mars)
    else:
        mcts = BatchedMCTS(model, num_simulations=sims, c_puct=c_puct,
                           batch_size=batch_size, add_root_noise=True,
                           adaptive_sims=False, use_tablebase=True)
        game_fn = lambda: play_game(mcts)

    buf_ids, buf_pm, buf_pp, buf_v = [], [], [], []
    shard = seed_offset
    results = {"w": 0, "b": 0, "d": 0}

    def flush():
        nonlocal shard, buf_ids, buf_pm, buf_pp, buf_v
        if not buf_ids:
            return
        path = os.path.join(out_dir, f"shard_{shard:05d}.npz")
        np.savez_compressed(
            path,
            inputs=np.stack(buf_ids), policy_moves=np.stack(buf_pm),
            policy_probs=np.stack(buf_pp), value=np.array(buf_v, dtype=np.float32))
        print(f"  wrote {path} ({len(buf_ids)} positions)")
        shard += 1
        buf_ids, buf_pm, buf_pp, buf_v = [], [], [], []

    for g in range(num_games):
        samples, z = game_fn()
        for ids, pm, pp, v in samples:
            buf_ids.append(ids); buf_pm.append(pm); buf_pp.append(pp); buf_v.append(v)
        results["w" if z > 0 else "b" if z < 0 else "d"] += 1
        if (g + 1) % 10 == 0:
            print(f"  game {g+1}/{num_games} | positions {len(buf_ids)} | "
                  f"W/D/B {results['w']}/{results['d']}/{results['b']}")
        if len(buf_ids) >= shard_size:
            flush()
    flush()
    print(f"Generated {num_games} games -> {out_dir} | W/D/B "
          f"{results['w']}/{results['d']}/{results['b']}")


# ─── Replay dataset + training iteration ─────────────────────────────────────

class SelfPlayDataset(torch.utils.data.Dataset):
    def __init__(self, replay_dir):
        self.files = sorted(glob.glob(os.path.join(replay_dir, "*.npz")))
        self.index = []   # (file_idx, row)
        self._cache = {}
        for fi, f in enumerate(self.files):
            n = int(np.load(f)["value"].shape[0])
            self.index.extend((fi, r) for r in range(n))
        print(f"SelfPlayDataset: {len(self.index):,} positions from {len(self.files)} shards")

    def _load(self, fi):
        if fi not in self._cache:
            if len(self._cache) > 4:
                self._cache.clear()
            self._cache[fi] = np.load(self.files[fi])
        return self._cache[fi]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fi, r = self.index[i]
        d = self._load(fi)
        return {
            "input_ids": torch.as_tensor(d["inputs"][r], dtype=torch.long),
            "policy_moves": torch.as_tensor(d["policy_moves"][r].astype(np.int64)),
            "policy_probs": torch.as_tensor(d["policy_probs"][r].copy(), dtype=torch.float32),
            "value": torch.tensor(float(d["value"][r]), dtype=torch.float32),
        }


def train_iteration(model_path, replay_dir, epochs=2, batch_size=48, lr=1e-4,
                    device="cpu", out_path=None):
    from data.dataset import build_dense_policy
    out_path = out_path or model_path
    model, cfg = load_model(model_path, device=device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ds = SelfPlayDataset(replay_dir)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                         num_workers=2, pin_memory=use_amp, drop_last=True)
    for ep in range(1, epochs + 1):
        tot = pol = val = 0.0; nb = 0
        for batch in loader:
            ids = batch["input_ids"].to(device)
            pm = batch["policy_moves"].to(device); pp = batch["policy_probs"].to(device)
            target_v = batch["value"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(ids, return_dict=True)
                pi = build_dense_policy(pm, pp, ACTION_SPACE).clamp(min=1e-9)
                policy_loss = F.kl_div(F.log_softmax(out["policy"], -1), pi, reduction="batchmean")
                v_pred = out["wdl"][:, 0] - out["wdl"][:, 2]
                value_loss = F.mse_loss(v_pred, target_v)
                loss = policy_loss + value_loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item(); pol += policy_loss.item(); val += value_loss.item(); nb += 1
        print(f"  iter epoch {ep}/{epochs} | loss {tot/max(nb,1):.4f} "
              f"(policy {pol/max(nb,1):.4f} value {val/max(nb,1):.4f})")
    torch.save(model.state_dict(), out_path)
    json.dump({k: v for k, v in cfg.__dict__.items()},
              open(out_path.replace(".pt", "_config.json"), "w"), indent=2)
    print(f"[OK] self-play iteration saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", default="chess_mamba.pt")
    g.add_argument("--out", default="selfplay")
    g.add_argument("--games", type=int, default=200)
    g.add_argument("--sims", type=int, default=400)
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--device", default=None)
    g.add_argument("--seed-offset", type=int, default=0)
    g.add_argument("--backend", choices=["mcts", "mars"], default="mars")
    g.add_argument("--search", default="search_mamba.pt", help="Search Mamba for MARS")
    t = sub.add_parser("train")
    t.add_argument("--model", default="chess_mamba.pt")
    t.add_argument("--replay", default="selfplay")
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--batch-size", type=int, default=48)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--device", default=None)
    t.add_argument("--out", default=None)
    a = ap.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if a.cmd == "generate":
        generate(a.model, a.out, num_games=a.games, sims=a.sims, device=device,
                 batch_size=a.batch_size, seed_offset=a.seed_offset,
                 backend=a.backend, search_path=a.search)
    else:
        train_iteration(a.model, a.replay, epochs=a.epochs, batch_size=a.batch_size,
                        lr=a.lr, device=device, out_path=a.out)


if __name__ == "__main__":
    main()
