# ChessMamba v4 — The MARS Architecture (full implementation plan)

A novel, fully neural chess engine built to contend with Stockfish/LC0 **without
alpha-beta, without classical MCTS/PUCT tree pruning, and without NNUE.** It keeps
the existing dual-Mamba foundation and adds four upgrades you specified:

1. **Negative-eigenvalue / Gated-DeltaNet state tracking** in the eval backbone
2. **Bidirectional (multi-directional) board scan** in the eval backbone
3. **Monte-Carlo Graph Search (MCGS)** — a transposition DAG instead of a tree
4. **MARS** — *Mamba Adversarial Recurrent Search*: a Gumbel best-first search
   whose node evaluations come from O(1) recurrent steps of the Search Mamba,
   periodically re-anchored to the Eval Mamba.

> **Honest framing (read once).** "Beat Stockfish straight out of training" =
> *architecture (this doc) + the self-play phase run to convergence.* The
> supervised stage alone yields a strong-GM engine; surpassing full-strength
> Stockfish requires the self-play loop (Phase D) and, realistically, large
> compute. Every mechanism below is either already proven in the literature or
> cheaply falsifiable; the one genuinely novel bet (an SSM as the value model for
> chess search) is de-risked by an explicit A/B gate before any scaling. With
> hardware off the table, this is the complete, coherent plan — not a guarantee,
> but the strongest grounded vehicle for the goal.

---

## BUILD STATUS — the architecture is IMPLEMENTED (training is the remaining work)

All code below is built and tested on CPU (`engine/test_v4.py`: 26/26;
`engine/test_v3.py` regression: 34/34). What remains is *running the training*,
which needs the dataset + GPU time (Phases below map to `engine/NEXT_STEPS.md`).

| Plan phase | Code | Where |
|---|---|---|
| A. Backbone (neg-eigenvalue tracking + bidirectional) | ✅ done | `mamba.py`, `model.py` |
| B. Search Mamba value-equivalent dynamics (+ consistency) | ✅ done | `search_mamba.py`, `train.py:DynamicsTrainer`, `data/dataset.py:DynamicsDataset` |
| C. MCGS + MARS + selector backend | ✅ done | `mcgs.py`, `mars_search.py`, `move_selector.py` |
| D. Self-play with MARS | ✅ done | `self_play.py:play_game_mars` |
| Benchmark MARS vs MCTS (A/B) | ✅ done | `tournament.py --opponent self --backend mars --opponent-backend mcts` |
| **Train to strength** | ⏳ run it | supervised -> dynamics -> self-play (NEXT_STEPS) |

---

## 0. The central design insight (why this can out-search MCTS/alpha-beta)

We **know the rules of chess** — so, unlike MuZero, we do **not** ask a learned
model to predict legality or transitions. python-chess gives exact legal moves,
transitions, transposition keys, terminals, and Syzygy. The learned model is used
**only to make per-node *evaluation* cheap.**

- **MCTS/LC0** pays a *full network forward pass* at every node it evaluates.
- **alpha-beta/Stockfish** pays a fast hand-crafted/NNUE eval but searches millions
  of nodes via pruning heuristics.
- **MARS** pays **one O(1) recurrent `step()`** of the small Search Mamba per node
  (carrying SSM state down the line), and only runs the heavy Eval Mamba at the
  root and at periodic *re-anchor* points.

Net effect: many more node-evaluations per second than MCTS at equal accuracy,
with all rules-correctness kept exact and free. The only error source is the
*value/policy estimate*, which is bounded by re-anchoring. That is a far smaller
risk surface than a full learned-dynamics model.

---

## 1. Architecture overview

```
                         Position (python-chess Board)  ──► exact rules, transpositions, Syzygy
                                       │
        ┌──────────────────────────────┴───────────────────────────────┐
        │                                                                │
  EVAL MAMBA  (heavy, bidirectional, negative-eigenvalue)         SEARCH MAMBA (light, causal)
  encode full board  →  policy | WDL | action-value(Q)            value-equivalent recurrent
                         uncertainty | strategy | embedding         dynamics: step(state, move)
        │   used at: ROOT  +  every K-ply RE-ANCHOR                    → value, continuation-policy
        │                                                                │  used at: every node (O(1))
        └───────────────────────────────┬────────────────────────────────┘
                                         ▼
                        MARS  =  Gumbel best-first search
                          • Gumbel + Sequential Halving at root (few-sim policy improvement)
                          • completed-Q interior selection
                          • recurrent rollouts (Search Mamba step) for cheap depth
                          • re-anchor to Eval Mamba every K plies (drift control)
                          • MCGS transposition DAG (share value/visits across move orders)
                          • captures/checks quiescence at leaves (tactical safety, NOT alpha-beta)
                          • Syzygy exact value at <=7 pieces
                                         ▼
                        Geometric tie-break (15 phase x context vectors)
                                         ▼
                                     best move
```

**Eliminated:** alpha-beta, PUCT tree + classical pruning, NNUE, Transformers.

---

## 2. Component specifications

### 2.1 Encoding (`encoding.py`) — minor additions
- Keep the v3 token encoding (history + 64-square snapshot + metadata). `ACTION_SPACE=8192`.
- **Add** `zobrist_key(board)` = `chess.polyglot.zobrist_hash(board)` for MCGS transposition keys.
- **Add** a *causal move-stream* note: the Search Mamba consumes `encode(root) + [move tokens]`; this is append-only and is what makes `step()` exact. The Eval Mamba consumes a full snapshot (no append-only requirement) → free to be bidirectional.

### 2.2 Eval Mamba backbone upgrades (`mamba.py`, `model.py`)

**(A) Negative-eigenvalue / state-tracking spectrum.** Standard SSMs constrain the
state transition `a ∈ (0,1)` (decay only) and *cannot* track parity / discrete
state (Merrill et al. 2024, "Illusion of State in SSMs"). Fix (Grazzi et al.,
ICLR'25): allow `a ∈ (-1,1)`. Our Mamba-3 already has a real gate
`dA = exp(Δ·A_real)·cos(Δ·A_imag) ∈ (-1,1)` — a phase near π yields a *negative*
(sign-flipping) eigenvalue — but it isn't initialized or allocated for tracking.

Concrete change to `Mamba3Block`:
- Partition the `d_state` dimension into two groups: `N_rot` **rotational** channels
  (complex, as today) and `N_track` **tracking** channels with a *real, signed*
  input-independent eigenvalue `λ = tanh(θ) ∈ (-1,1)` per (channel, state),
  `θ` learnable, **initialized spread across (-1,1) including near −1** so some
  channels start as parity trackers.
- The tracking channels use the same parallel scan (it already supports negative
  gates — verified). Add config: `n_track_state: int` (default `d_state // 4`).
- *(Stretch alternative)* a Gated-DeltaNet mixer (delta-rule associative memory)
  for even stronger tracking; higher risk, keep behind a flag.

**(B) Bidirectional / multi-directional scan.** The 64-square block has 2D
structure a single forward scan underuses (Vision-Mamba / Vim, 2024).
Concrete change: add `bidirectional: bool` to the **Eval Mamba only**. In
`Mamba3Block.forward`, run the SSM on `x` and on `x.flip(1)`, merge by
`out = out_fwd + out_bwd` (or concat→linear). Optionally add a second board
ordering (file-major) as a third scan. Cost: 2–3× the scan, still linear.

> **Critical constraint:** bidirectional is **Eval-Mamba-only**. The Search Mamba
> MUST stay strictly **causal/unidirectional** or `step()` is no longer exact and
> MARS rollouts break. Enforce in code: `SearchMambaConfig.bidirectional=False`,
> asserted.

`model.py`: `ChessMamba` keeps its 6 heads; backbone built with
`bidirectional=True, n_track_state=...`. Everything else unchanged.

### 2.3 Search Mamba → value-equivalent recurrent dynamics model (`search_mamba.py`)

Today it scores lines. For MARS it must, at **every recurrent step**, output the
quantities planning needs (value-equivalence principle, Grimm et al. 2020):
- `value_head(h_t) → v̂_t` ∈ [−1,1] (side-to-move value after the move)
- `continuation_head(h_t) → π̂_t` (already exists) — learned move ordering / priors
- `consistency_proj(h_t)` — for the EfficientZero consistency loss

`step()` already verified bit-exact and append-only. Add `value_head`,
`consistency_proj`; keep causal. Add a `rollout(cache, move_tokens)` returning the
per-step `(v̂, π̂, h)` sequence for training and search.

### 2.4 MCGS — transposition graph (`mcgs.py`, NEW)

- `TranspositionTable`: `dict[zobrist_key → NodeStats]` where
  `NodeStats = {N (visits), W (value sum), Q, P (prior policy), terminal, wdl}`.
- The search references nodes by **board key**, so positions reached by different
  move orders **share value/visit statistics** (huge in chess).
- **Recurrent state is path-dependent**, so it is **not** stored in the shared node;
  it is carried along the *current rollout path* only. Re-anchoring (2.5) corrects
  any path-induced value drift, so sharing statistics by board key stays sound.
- Backup: standard MCGS update along the selected path (Czech et al. 2020).

### 2.5 MARS search loop (`mars_search.py`, NEW) — the core

```
search(root_board, sim_budget, m=16, beam=4, K_anchor=4, depth_cap=12):
    # ---- Root anchor (heavy net, once) ----
    e = EvalMamba(encode(root_board))            # prior P, value V, action-value Q, uncertainty U
    sim_budget = adapt_by_uncertainty(U)         # uncertainty head sets the budget
    root = TT.get_or_create(zobrist(root_board), P=e.policy, Q_init=e.action_value, V=e.value)
    sm_root_cache = SearchMamba.prime(encode(root_board))

    # ---- Gumbel root selection via Sequential Halving ----
    cand = gumbel_top_k(root.P, m)               # g_a ~ Gumbel; pick m by g_a + logit
    for round in sequential_halving_rounds(sim_budget, len(cand)):
        for a in cand:
            run_rollout(root_board, sm_root_cache, first_move=a, K_anchor, depth_cap, beam)
        cand = keep_top_half(cand, key=lambda a: g_a + logit(a) + sigma(completed_Q(a)))
    return argmax_or_sample(cand by completed_Q),  visit/value stats   # Gumbel-improved move

run_rollout(board0, sm_cache0, first_move, K, depth_cap, beam):
    board = board0.copy(); sm_cache = clone(sm_cache0)
    path = []; ply = 0; move = first_move
    while True:
        board.push(move)                         # EXACT rules (python-chess)
        node = TT.get_or_create(zobrist(board))
        path.append(node)
        # terminal / tablebase = exact, free
        if board.is_game_over(): v = terminal_value(board); break
        if can_probe_syzygy(board): v = syzygy_value(board); break
        # cheap O(1) eval via Search Mamba step
        h, sm_cache = SearchMamba.step(sm_cache, token(move))
        v_hat, pi_hat = value_head(h), policy_head(h)
        node.P = node.P or pi_hat                 # set priors lazily from the model
        ply += 1
        # ---- periodic re-anchor (drift control) ----
        if ply % K == 0 or ply >= depth_cap:
            e = EvalMamba(encode(board))          # heavy, accurate refresh
            v = optional_quiescence(board, e.value)   # captures/checks only
            sm_cache = SearchMamba.prime(encode(board))  # reset latent to truth
            if ply >= depth_cap: break
            v_hat = e.value
        # ---- interior selection: Full-Gumbel deterministic + completed-Q over a beam ----
        move = gumbel_interior_select(node, beam, completed_Q)
        if move is None: v = v_hat; break
    negamax_backup(path, v)                       # flip sign each ply; MCGS update
```

Key points:
- **Rules-correctness is exact and free** (python-chess); the model only supplies
  cheap value/priors.
- **completed-Q-values** (mctx default) give a valid policy improvement from few
  visits — Gumbel MuZero "learns reliably even with 2 simulations."
- **Re-anchor every K plies** resets the latent to a true Eval-Mamba encoding,
  bounding value drift — the failure mode the literature warns about.
- **Quiescence** (captures+checks only, depth ≤ ~4) at re-anchor/leaf is the single
  sliver of exact tactical calc we allow; it is **not** alpha-beta and is optional.

### 2.6 Move selector integration (`move_selector.py`)
- **The shipped engine runs MARS. MCTS is NOT a component of the product.**
- The existing v3 MCTS is kept *temporarily and only* as a **development benchmark**
  (the Phase-C A/B control): it lets us prove MARS is better on the same network.
  Once MARS passes the gate, the MCTS code (`batched_mcts.py`, `mcts.py`) is
  **deleted** from the engine.
- Optional alternative if you want MCTS gone even as a yardstick: benchmark MARS
  against (i) the **searchless net** (does search help at all?) and (ii)
  **Stockfish at fixed depth** (absolute strength) — neither requires MCTS.
- Phase 0 tablebase shortcut and Phase B geometric tie-break unchanged.

---

## 3. Training pipeline

### 3.1 Data (`data/convert_lichess_evals.py`) — add per-ply values
- Keep v3 outputs (side-to-move antisymmetric WDL, soft policy, strategy, phases).
- **Add** `pv_values`: for each stored PV line `m1..mK`, a per-ply value sequence.
  Supervised bootstrap: `v_t = (−1)^t · value(B0)` (eval is ~stable along a PV;
  sign alternates by side to move). Cheap and good enough to pretrain the dynamics
  model; real targets come from self-play (3.4).

### 3.2 Supervised pre-training (`train.py`)
**Eval Mamba** (bidirectional + negative-eigenvalue backbone), v3 loss unchanged:
`L_eval = KL(policy) + CE(WDL) + MSE(action_value@best) + BCE(strategy) + SupCon(embedding) + MSE(uncertainty)`

**Search Mamba dynamics** (new objective), per PV line unrolled with `step()`:
```
L_dyn = Σ_t [ MSE(v̂_t, v_t)                      # value-equivalent value
            + CE(π̂_t, m_{t+1})                    # continuation / move ordering
            + λ_c · (1 − cos(consistency_proj(h_t), stopgrad(EvalEmbed(B_t)))) ]  # EfficientZero consistency
```
`B_t` reconstructed by replaying `m1..mt` from the FEN; `EvalEmbed` = Eval Mamba
embedding (frozen). The **consistency term is the ingredient** that keeps deep
recurrent rollouts accurate (EfficientZero's biggest single contributor).

### 3.3 Geometric advantage vectors (`train.py --phase vectors`) — unchanged (v3).

### 3.4 Self-play value-equivalence RL (`self_play.py`, `async_self_play.py`)
The phase that pushes past the supervised ceiling toward engine strength.
- Generate games with **MARS** (Gumbel root, low sim budget).
- Record per position: **Gumbel-improved policy** π (the search's improved
  distribution, *not* raw net policy), **backed-up value** z (blend of MARS root
  value and game outcome), and the **realized next positions** along the chosen
  line (for the dynamics consistency target — now *real* search trajectories, not
  the sign-alternated bootstrap).
- Train Eval + Search jointly: `KL(net_policy, π) + MSE(net_value, z) + L_dyn(real)`.
- Re-compute geometric vectors periodically. Loop: gen → train → gen …

### 3.5 Loss summary
| Model | Supervised | Self-play |
|---|---|---|
| Eval Mamba | policy KL + WDL + AV + strategy + contrastive + uncertainty | KL(π_search) + MSE(z) |
| Search Mamba | value + continuation + consistency (bootstrap targets) | value + continuation + consistency (real targets) |

---

## 4. File-by-file change map

**Modify**
| File | Change |
|---|---|
| `mamba.py` | negative-eigenvalue tracking-channel group; `bidirectional` option (eval only); assert search stays causal |
| `model.py` | build Eval backbone with `bidirectional=True, n_track_state` |
| `search_mamba.py` | add `value_head`, `consistency_proj`, `rollout()`; keep causal; verify `step()` exactness preserved |
| `encoding.py` | `zobrist_key()` helper |
| `move_selector.py` | `backend="mcts"|"mars"` switch |
| `train.py` | Search-Mamba dynamics objective (value+continuation+consistency); self-play value-equivalence targets |
| `self_play.py` | drive games with MARS; record π_search, z, real next-states for consistency |
| `data/convert_lichess_evals.py` | emit `pv_values` per PV line |
| `tournament.py` | MARS-vs-MCTS A/B mode (equal node budget) |

**New**
| File | Purpose |
|---|---|
| `mcgs.py` | transposition table + graph node stats + MCGS backup |
| `mars_search.py` | the MARS loop (Gumbel sequential-halving, recurrent rollouts, re-anchor, quiescence, syzygy) |
| `test_v4.py` | exactness + integration tests for all of the above |

---

## 5. Phased build order — each phase has a hard verification gate

| Phase | Build | **Gate (must pass before next phase)** |
|---|---|---|
| **A. Backbone** | negative-eigenvalue + bidirectional Eval Mamba; causal Search Mamba unchanged | `test_v4`: step() still bit-exact; eval top-1/top-5 on val set ≥ v3 baseline; a synthetic parity-tracking probe improves with tracking channels |
| **B. Dynamics** | Search Mamba value+continuation+consistency training | rollout value error vs Eval-Mamba re-anchor **decreases** with the consistency loss (ablation); continuation top-1 ≥ X% |
| **C. MARS + MCGS** | `mcgs.py`, `mars_search.py`, MARS selector | **THE bet:** MARS **out-plays the v3 MCTS at equal node budget** (dev-time A/B control only — MCTS is the *yardstick*, not a shipped component). If MARS loses, **fix MARS** (more frequent re-anchor, better dynamics training, deeper quiescence) and re-iterate B — do **not** ship MCTS. Once MARS wins, delete the MCTS code. |
| **D. Self-play RL** | value-equivalence loop with MARS | Elo vs fixed Stockfish-depth ladder **rises** across iterations; no collapse |
| **E. Scale** | bigger nets (d=768/24), torch.compile, more self-play | monotone Elo gains; periodic regression vs previous best |

This ordering means the **novel, risky piece (C) is validated cheaply and early**,
before any expensive scaling — you never bet compute on an unproven search.

---

## 6. Hyperparameters

**3060 Ti / 8 GB defaults** (fit-first): Eval d=512/16 (bidirectional → ~1.5× cost,
keep batch ~32 + grad-accum), Search d=384/12 causal, `n_track_state=16`,
MARS `sim_budget` adaptive 8–64, `m=16`, `beam=4`, `K_anchor=4`, `depth_cap=12`,
`λ_c=0.5`. AMP + gradient checkpointing on.

**Scale config (hardware off the table):** Eval d=768/24, Search d=512/16,
`sim_budget` 64–512, `depth_cap` 20+, large-batch self-play across GPUs. Same code,
bigger numbers.

---

## 7. Risk register & honest expectations

| Risk | Severity | Mitigation |
|---|---|---|
| **Recurrent value drift** in deep rollouts (the core bet) | High | Re-anchor every K plies to the true Eval Mamba; EfficientZero consistency loss; **Phase-C A/B gate** kills the design early if it doesn't beat MCTS |
| Tactical blindspot (neural eval misses forced lines) | Med | captures/checks quiescence at leaves; Syzygy; action-value Q-init; deeper sims |
| Bidirectional breaks step() if misapplied | Med | hard assert: Eval-only bidirectional; Search strictly causal |
| MCGS unsoundness from path-dependent state | Med | share *statistics* by board key only; keep latent per-path; re-anchor |
| Self-play instability / collapse | Med | value-equivalence targets, replay buffer, regression gating in Phase E |
| Beating *full* Stockfish needs scale | High (compute) | the explicit purpose of Phase D/E; with hardware unconstrained this is feasible, not guaranteed |

**Expected trajectory:** end of Phase B–C → strong club/master baseline; end of a
modest Phase D → strong-GM and beats time-limited Stockfish; full Phase D/E at
scale → genuine engine-level contention. "Straight out of training" means *after
Phase D converges*, not after supervised pretraining alone.

---

## 8. What makes this novel (one line)

A learned **best-first Gumbel search over a transposition DAG, where every node is
evaluated by an O(1) recurrent step of a value-equivalent SSM and re-anchored to a
negative-eigenvalue, bidirectional SSM evaluator** — keeping chess's exact rules
free while making neural evaluation cheap enough to out-search MCTS per FLOP.
Nothing here depends on an unsolved scientific result; the only unproven
combination is gated by Phase C before any scaling.
