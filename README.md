# ChessMamba ♟️

**A chess engine on a dual Mamba-3 state-space backbone, with MARS — a novel recurrent search that replaces alpha-beta and Monte-Carlo Tree Search.**

---

## The idea

Every top chess engine pays a fixed neural-network cost at *every node* it searches — Stockfish runs NNUE, Leela/AlphaZero run a full network forward pass. **MARS (Mamba Adversarial Recurrent Search)** carries a state-space hidden state *down each line of play* and updates it one move at a time with an **O(1) recurrent step**, paying the full network cost only at the root and at periodic re-anchor points.

The payoff: dramatically cheaper per-node evaluation, and therefore **more search at equal compute**.

> **Validated result:** MARS searches **1.55× more positions per second than batched MCTS** at equal wall-clock, on full-size networks, on an A100-class GPU.

This is the same efficiency principle that makes NNUE fast (incremental evaluation), reached through a completely different mechanism — a recurrent state-space model.

---

## Key results (all measured, not estimated)

| Component | Result |
|---|---|
| **MARS search throughput vs batched MCTS** | **1.55× faster** at equal wall-clock, full-size networks |
| **Custom Triton scan kernel** | **bit-exact** (forward ~1e-6, gradients ~1e-8), **~30× training speedup** (14.8 → ~0.5 s/batch) |
| **O(1) recurrent step vs parallel scan** | bit-exact (verified across sequence lengths) |
| **From-scratch bitboard move generator** | **perft-validated** + 59,598 positions cross-checked vs python-chess, **0 errors** |

---

## Architecture

### Dual Mamba-3 backbone

- **Eval Mamba** (~63M params) — the heavy, accurate evaluator. **Bidirectional** scan for 2D board awareness; **negative-eigenvalue tracking channels** to represent discrete state (castling rights, repetition). Runs at the root and at re-anchor points. Outputs policy (8,192-move space), WDL, action-values, uncertainty, and a position embedding.
- **Search Mamba** (~23M params) — the cheap recurrent dynamics model. **Causal** (so its `step()` is provably bit-exact); a value-equivalent model trained with an **EfficientZero consistency loss** that keeps deep rollouts grounded. Drives every rollout node via an **O(1) step** instead of a full forward pass.

### Mamba-3 innovations (in `mamba.py`)
- **Exponential-trapezoidal discretization** — 2nd-order accurate (sees current *and* previous input)
- **Complex-valued state transitions** — RoPE-like rotational dynamics
- **MIMO** state mixing + **BCNorm** (RMSNorm on B/C projections)
- **Negative-eigenvalue tracking channels** — eigenvalues `λ = tanh(θ) ∈ (−1, 1)` that flip sign each step, representing parity/discrete state that decay-only SSMs *provably cannot*

### MARS — the search (per move)

```
1. Eval Mamba at the ROOT        →  policy prior, value, uncertainty (sets the budget)
2. Gumbel top-k                  →  ≤ m_root candidate root moves
3. Sequential halving            →  allocate the simulation budget across candidates
4. Each simulation = a stochastic principal-variation ROLLOUT:
     · move ordering from the Search Mamba's learned continuation head
     · each ply advanced by the O(1) eval_step()         (cheap)
     · re-anchored to the Eval Mamba every k_anchor plies (bounds value drift)
     · captures/checks quiescence at the leaf
5. MCGS transposition DAG (Zobrist)  →  share position values across rollouts
6. Negamax backup (side-to-move-relative)  →  pick the Gumbel-improved best move
```

**Correctness is never the network's job:** legality, transitions, terminals, and Syzygy come from exact rules (python-chess). The learned model *only* makes per-node evaluation cheap; value drift is bounded by re-anchoring.

---

## Technical highlights

- **Custom fused Triton GPU kernel** (`triton_scan.py`) for the Mamba-3 trapezoidal selective scan — forward + analytic reverse-scan backward, **bit-exact** with the reference, and self-verifying on the GPU before any training is allowed. Took training from 14.8 → ~0.5 s/batch (~30×).
- **O(1) recurrent step** — carry the SSM state down a line of play and update per move; bit-exact with the full parallel scan.
- **Batched lockstep rollouts** — B simulations advanced together with every neural operation batched into a single GPU call (correctness-verified, no cross-rollout contamination).
- **Rigorous measurement harness** (`mars_diagnostics.py`) — correctness gate, throughput sweep, and a MARS-vs-MCTS head-to-head that isolated the real bottleneck through controlled experiments rather than guesswork.

---

## Repository structure

```
engine/
  mamba.py             Mamba-3 SSM backbone (complex eigenvalues, tracking channels)
  triton_scan.py       fused bit-exact Triton kernel for the selective scan
  search_mamba.py      value-equivalent recurrent dynamics model (the rollout engine)
  mars_search.py       the MARS search algorithm
  mcgs.py              Monte-Carlo Graph Search transposition table (Zobrist)
  batched_mcts.py      batched MCTS baseline (the comparison)
  model.py             evaluation network + multi-task heads
  train.py             training pipeline: Eval Mamba → advantage vectors → Search Mamba
  fastchess.py         perft-validated Numba bitboard chess core
  mars_diagnostics.py  the MARS-vs-MCTS measurement harness
  encoding.py          board/move tokenization
  tournament.py        Elo evaluation vs Stockfish
  test_*.py            integration + correctness suites (perft, bit-exactness, etc.)
CHESSMAMBA_V4_PLAN.md  full architecture specification
FUTURE_IMPROVEMENTS.md engine audit + measured-Elo roadmap
requirements.txt
```

> **Note:** the trained weights (~hundreds of MB), the 5M-position training set (~3 GB), and Syzygy tablebases (~70 GB) are intentionally not committed. Datasets regenerate via `engine/data/convert_lichess_evals.py`; tablebases via `engine/download_tablebases.py`.

---

## Status & roadmap

- ✅ **Architecture built and validated** — dual Mamba-3 backbone, bit-exact Triton kernel, the MARS search, and a perft-validated chess core.
- ✅ **Speed validated** — MARS out-searches batched MCTS 1.55× at equal compute.
- 🔄 **Supervised training in progress** — the next milestone is the playing-strength head-to-head (MARS vs MCTS with trained networks).
- 🧭 **Beyond:** reaching the very top is a *compute-scaling* problem (self-play), not an architecture one — and MARS's search efficiency makes that scaling cheaper.

---

## Foundations

Grounded in MuZero, Gumbel MuZero ([mctx](https://github.com/google-deepmind/mctx)), EfficientZero, DeepMind's *Grandmaster-Level Chess Without Search* ([arXiv:2402.04494](https://arxiv.org/abs/2402.04494)), and negative-eigenvalue linear-RNN state-tracking results.

---

*ChessMamba is an independent research project exploring whether a recurrent state-space model can search more efficiently than the alpha-beta and MCTS algorithms behind today's strongest engines. The MARS search and dual-Mamba architecture are original.*
