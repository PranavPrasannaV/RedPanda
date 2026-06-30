# ChessMamba — How to Train & Test (single RTX 3060 Ti)

> **v4 / MARS is implemented** (see `../CHESSMAMBA_V4_PLAN.md`). The training flow
> is the same three phases below, plus the value-equivalent dynamics model and the
> MARS search. Quick map:
>
> ```bash
> cd engine
> # backbone upgrades are ON by default (bidirectional + negative-eigenvalue tracking)
> python train.py --phase eval    --data data/training --epochs 12 --batch-size 48
> python train.py --phase vectors --data data/training
> python train.py --phase search  --data data/training --epochs 8   # value-equivalent dynamics (consistency loss)
>
> # Phase-C gate: does MARS out-play the MCTS yardstick on the same net?
> python tournament.py --opponent self --backend mars --opponent-backend mcts --games 40 --sims 64
>
> # Self-play RL with MARS (Phase D):
> python self_play.py generate --backend mars --model chess_mamba.pt \
>     --search search_mamba.pt --games 2000 --sims 64 --device cuda
> python self_play.py train --model chess_mamba.pt --replay selfplay --epochs 2 --device cuda
>
> # Play / test with MARS (auto-selected when search_mamba.pt is present):
> python uci.py
> python tournament.py --opponent stockfish --stockfish <path> --sf-depth 6 --games 40 --sims 64
> ```
> `python test_v4.py` runs the 26-test integration suite. The rest of this doc is
> the shared v3 detail (data format, scaling, schedule) and still applies.

---

## (v3 detail) Train & Test on a single RTX 3060 Ti

Everything below assumes you are in the repo root and the engine lives in
`engine/`. All commands run on one 8 GB GPU. AMP + gradient checkpointing are on
by default so the model fits; raise `--grad-accum` to grow the effective batch
without more VRAM.

```
pip install -r requirements.txt
```

---

## 0. Get the data (once)

The supported data path is the **Lichess pre-evaluated positions** dataset
(every position already has Stockfish `cp`/`mate`/`line`/`depth`). Two options:

**A. Convert on Kaggle (recommended — CPU, fast), download `training_data/`.**
Open `chessmamba_v3_kaggle.ipynb`, run *Phase 0*, Save Version, download the
`training_data/` output, drop it in `engine/data/training/`.

**B. Convert locally** if you have the dataset on disk:
```
python engine/data/convert_lichess_evals.py \
    --input  /path/to/lichess-evaluations \
    --output engine/data/training \
    --max-positions 3000000 --min-depth 20
```
Output is compact (sparse soft policy + int16 encodings): ~3M positions ≈ 1–2 GB.
Values are stored **side-to-move relative** (required for correct MCTS backup).

What the converter produces: `inputs.npy`, `policy_moves.npy`, `policy_probs.npy`,
`best_moves.npy`, `wdl.npy`, `centipawns.npy`, `phases.npy`, `strategy_labels.npy`,
`pv_lines.json`, `metadata.json`.

---

## 1. Supervised pre-training (the foundation)

```
cd engine
python train.py --phase eval    --data data/training --epochs 12 --batch-size 48
python train.py --phase vectors --data data/training          # geometric vectors
python train.py --phase search  --data data/training --epochs 8
```

- `eval` trains policy (KL to soft Stockfish target) + WDL + **eval-bucket
  contrastive** + **strategy BCE** + **action-value MSE** + uncertainty.
- `vectors` builds the 3×5 phase×context advantage vectors from embeddings.
- `search` trains the Search Mamba look-ahead evaluator on PV lines.

Outputs: `chess_mamba_best.pt` (+ `_config.json`), `advantage_vectors.pt`,
`search_mamba.pt` (+ `_config.json`). Rename `chess_mamba_best.pt` →
`chess_mamba.pt` for inference loaders.

**Targets after Phase 1:** policy top-1 ≈ 35 %+, top-5 ≈ 65 %+, WDL loss < 0.8.

### Scaling the model
Default is `d=512 / 16 layers` (~63 M, fits 8 GB comfortably). To scale toward
the Phase-6 target on the same card:
```
python train.py --phase eval --data data/training \
    --d-model 768 --n-layer 24 --batch-size 16 --grad-accum 4 --epochs 12
```

---

## 2. Verify the engine

```
# Unit / integration tests (CPU, seconds):
python test_v3.py

# Play a move from a FEN with the full MCTS pipeline:
python -c "import torch,chess,json; from mamba import MambaConfig; \
from model import ChessMamba; from batched_mcts import BatchedMCTS; from encoding import ACTION_SPACE; \
cfg=MambaConfig(**{k:v for k,v in json.load(open('chess_mamba_config.json')).items() if k in MambaConfig.__dataclass_fields__}); \
m=ChessMamba(cfg,action_space=ACTION_SPACE).eval(); m.load_state_dict(torch.load('chess_mamba.pt',weights_only=True)); \
b=chess.Board(); print(b.san(BatchedMCTS(m,num_simulations=400,adaptive_sims=True).search(b)))"

# UCI engine (load into CuteChess / Arena / a Lichess bot):
python uci.py
```

### Strength vs Stockfish
```
python tournament.py --opponent stockfish --stockfish <path-to-stockfish> \
    --sf-depth 6 --games 40 --sims 400
```
Reports score + an Elo-difference estimate. Use `--sf-skill 0..20` or
`--sf-movetime` to calibrate the opponent. Regression-test against an older net
with `--opponent self --opponent-model old.pt`.

### Endgames
Download Syzygy tablebases for perfect ≤7-piece play (auto-probed in MCTS):
```
python download_tablebases.py        # places files in engine/syzygy/
```

---

## 3. Self-play reinforcement (break the supervised ceiling)

Each iteration: generate games with the current net, then train toward the MCTS
visit distribution + blended value. Repeat 10–20× (each ≈ +50–100 Elo).

```
# Generate (single GPU; raise --batch-size for throughput):
python self_play.py generate --model chess_mamba.pt --out selfplay \
    --games 2000 --sims 400 --device cuda
# Or multi-process:
python async_self_play.py --model chess_mamba.pt --out selfplay \
    --games 2000 --workers 2 --sims 400 --device cuda

# Train one iteration on the replay buffer:
python self_play.py train --model chess_mamba.pt --replay selfplay \
    --epochs 2 --device cuda --out chess_mamba_iter1.pt

# Re-run vectors, then loop: gen -> train -> gen ...
```

Suggested schedule on a 3060 Ti:

| Step | ~Time | Notes |
|------|-------|-------|
| Supervised pretrain (3M pos, 12 ep) | 1–2 days | Phase 1 |
| Self-play gen #1 (50k games @ 400 sims) | ~1 day | raise sims each cycle |
| Train iteration #1 | a few hrs | |
| repeat 10–20 cycles | 2–4 weeks | each ≈ +50–100 Elo |

---

## 4. Optional speedups

- `from torch_compile_wrapper import maybe_compile` then wrap the model
  (`maybe_compile(model, mode="reduce-overhead")`) for 1.5–3× on recent CUDA.
- Larger MCTS `--batch-size` improves GPU utilisation during self-play/eval.

---

## Housekeeping

The large archives in the repo root are **not** needed to train or run the
engine and can be deleted to reclaim ~100 GB:
`engine.zip`, `chessmamba-engine-v2.zip`, `lichess_db_standard_rated_2023-03.pgn.zst`.
(They are kept only because deleting user data is destructive — remove them
yourself when you're sure.)
