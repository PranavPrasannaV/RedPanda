# ChessMamba — Future Improvements & Engine Audit

Rules for this file:
- NOTHING here changes the novel search/eval architecture (dual Mamba-3 + MARS).
  These are additions, fixes, and tuning on top of it.
- Every item is a PROVEN technique (used in Stockfish/LC0/AlphaZero lineage) or a
  concrete bug-class fix found by code audit. No speculative ideas.
- Elo estimates are from engine-development literature, scaled honestly; treat as
  rough priorities, not promises. Verify every change with SPRT (see B4).
- Code locations are given so each item is actionable when we pick it up.

---

## PART A — Adopted from search literature (agreed 2026-06)

### A1. Check & singular extensions in MARS rollouts  [~30-80 Elo, cheap]
Where: `mars_search.py` `_rollout` (the `ply >= self.depth_cap` cutoff).
- Check extension: when `board.is_check()`, do not count that ply against
  `depth_cap` — forcing sequences must resolve before truncation (horizon effect).
- Singular extension: at a node where one candidate's backed-up value dominates
  all siblings by a margin, extend its rollout depth. Stockfish ~+30 Elo.
- Both are search-budget reallocation; the novel rollout mechanism is untouched.

### A2. Static Exchange Evaluation (SEE)  [~30-60 Elo, cheap, pure rules]
Where: new `see.py`; used in `mars_search.py` `_quiescence` and candidate ordering.
- SEE statically resolves capture sequences on one square (gain/loss in material)
  with zero NN calls — perfectly aligned with "rules are free, net evaluates".
- Use 1: quiescence — only consider SEE >= 0 captures (prune losing captures);
  current `_quiescence` considers ALL captures+checks up to `q_max`, wasting
  Eval-Mamba batch slots on obviously losing exchanges.
- Use 2: candidate move ordering — demote SEE-losing captures before Gumbel
  sampling so root candidates aren't wasted.
- python-chess has no built-in SEE; implement the standard swap algorithm
  (~60 lines, well documented in CPW "SEE - The Swap Algorithm").

### A3. Online move-ordering priors (history / killer / countermove)  [~20-50 Elo]
Where: `mars_search.py` (root candidate scoring) + `mcgs.py` (per-search stats).
- Track moves that produced good backups elsewhere in this search; add a small
  bonus to their prior when they appear as candidates in sibling positions.
- Supplements (never replaces) the neural policy prior. Smaller win than in
  alpha-beta engines because we already have a learned prior — keep the weight low.

### A4. SPRT testing methodology  [not Elo — but makes every other Elo claim real]
Where: new `sprt.py` + extend `tournament.py`.
- Sequential Probability Ratio Test: play games until statistical confidence that
  change A > change B (standard bounds: elo0=0, elo1=5, alpha=beta=0.05).
- Every item in this file gets merged ONLY if it passes SPRT vs the previous
  build. This is how Stockfish avoids self-deception; 20-game eyeball matches
  are noise. Also exactly what a paper reviewer expects for MARS-vs-MCTS claims.

---

## PART B — Engine audit findings (concrete gaps found in OUR code, 2026-06)

### B1. Mate-distance scoring  [HIGH PRIORITY — converts wins into wins]
Found: `mars_search.py` `_terminal_value` returns ±1.0 for ANY mate; MCTS
backups likewise carry no distance information.
Problem: the engine cannot prefer mate-in-1 over mate-in-9. In won positions it
shuffles (all winning lines look identical), risking 50-move and threefold draws
from completely won games. This is a classic engine bug class.
Proven fix: ply-aware mate scores — terminal win returns `1.0 - eps*ply`
(e.g. eps=1e-3) so shorter mates back up strictly higher; symmetrically
`-1.0 + eps*ply` for losses (prefer the longest defense). Keep |v| <= 1.
Also applies to tablebase WDL probes (prefer DTZ-decreasing moves when winning).
Cost: a few lines in `_terminal_value`, `_rollout` backup, and MCTS terminal
handling. Zero architecture impact.

### B2. Search reuse between moves  [~30-100 Elo at fast time controls]
Found: no subtree/TT reuse anywhere — `BatchedMCTS.search` and `MARS.run_search`
start cold every move; the MCGS transposition table is rebuilt per move.
Proven fix (LC0-standard):
- MCTS: after playing move m, re-root the tree at child(m) and keep its subtree
  statistics (visits/values) for the next search.
- MARS: persist the MCGS `TranspositionTable` across moves within a game
  (entries keyed by zobrist remain valid; just age them). Also persist across
  ponder/opponent moves.
Where: `batched_mcts.py`, `mcgs.py`, `move_selector.py` (owns per-game lifetime).

### B3. Real time management in UCI  [~30-70 Elo in clocked play]
Found: `uci.py` uses a flat rule (`my_time // 20 + inc/2`) and a fixed sims
budget; the search itself has no time-based stopping.
Proven fixes:
- Convert sims budget into a soft time budget with periodic clock checks;
  stop cleanly at ~90-95% of allocation (iterative anytime search — MCTS and
  MARS are both naturally anytime).
- Allocate MORE time when the root is unstable (best move changed recently or
  top-2 values are close) and LESS when one move dominates — "easy move" cutoff.
- Never flag: hard cap = remaining_time / 2 regardless of allocation.
Where: `uci.py` `_go`, plus a `time_budget_ms` parameter on both searches.

### B4. Pondering (think on opponent's time)  [~20-40 Elo in real matches]
Found: absent. UCI `go ponder` ignored.
Proven fix: after moving, keep searching from the expected reply position
(`ponderhit` = continue; `pondermiss` = restart). Combines with B2 (the work is
kept in the reused tree/TT). Lower priority until B2/B3 exist.

### B5. Deterministic match configuration  [free Elo — avoid self-sabotage]
Found (good): `tournament.py` `build_selector` already sets
`add_root_noise=False, adaptive_sims=False` — correct for match play.
Remaining: ensure temperature=0 (argmax) for ALL match/UCI play (noise and
temperature are for self-play exploration ONLY), and double-check `uci.py`
defaults match `tournament.py` (one config source, not two).

### B6. Inference speed = direct Elo (more sims per second)
Found: match/search inference runs the eval net in fp32, layer-by-layer, no
compile; MARS rollout steps are per-node Python calls.
Proven fixes (in order of value/effort):
- bf16 (or fp16) inference for Eval+Search Mamba in `move_selector.py` /
  `mars_search.py` — ~1.5-2x throughput on Ampere; doubling NPS ~ +50-70 Elo
  equivalent in engine literature. Verify move-choice agreement on a position
  suite vs fp32 first (same bar as the Triton gate).
- Batch MARS rollouts: run M rollouts lockstep, batching their per-ply
  `eval_step` calls into one GPU call (the Search Mamba state is (B,...) ready).
  This is the MARS-side analogue of `batched_mcts.py` and the biggest MARS
  speed lever besides the kernel.
- `torch.compile` the inference-only step path once stable.

### B7. Syzygy at the root: DTZ-aware move selection  [correctness in endgames]
Found: `tablebase.py` provides WDL probing for evaluation; root move selection
does not consult DTZ.
Problem: WDL-only play in tablebase-won positions can wander (same shuffling
class as B1) and even forfeit wins via the 50-move rule in DTZ-critical
positions.
Proven fix: at the root, if the position is in tablebase range, pick the
DTZ-optimal move directly (python-chess `probe_dtz`), bypassing search
entirely. Standard practice; ~free Elo in endgames and prevents catastrophes.

### B8. Opening diversity for evaluation matches  [methodology, like A4]
Found: `tournament.py` always starts from the initial position.
Problem: engine pairs repeat games from startpos (deterministic play -> the
same game over and over), making match results statistically meaningless.
Proven fix: play each match from a suite of balanced openings (e.g. the
standard 8-move UHO books, or even the converter's `start_fens.json`), each
opening played twice with colors swapped. Required for SPRT (A4) to be valid.

### B9. PUCT/FPU tuning sweep  [~20-60 Elo, zero code]
Found: `cpuct` and first-play-urgency style defaults in `batched_mcts.py` are
literature defaults, never tuned for OUR net's value scale.
Proven fix: once nets are trained, grid/SPRT-sweep cpuct (and Gumbel/m_root/
k_anchor/depth_cap for MARS) at fixed nodes. LC0 gained large Elo from exactly
this. Cheap to run overnight with A4+B8 infrastructure.

### B11. Python overhead in the search loop  [potentially the largest single item]
Context: classical engines lose ~350-450 Elo in Python (50-100x slower move
gen/tree code; +50-70 Elo per NPS doubling). For a GPU-neural engine the loss
is bounded by the fraction of wall-clock spent OUTSIDE the GPU — python-chess
push/pop, legal_moves, encode_board, zobrist, node bookkeeping. Honest range
for us: ~30-170 Elo depending on that fraction.
CRITICAL for MARS: the cheaper the O(1) GPU step, the larger the Python share
per node — overhead can eat the exact speed advantage MARS exists to prove
(the "speed ratio ~ 1" failure mode). A Stage-2 loss from this is an
ENGINEERING result, not a dead thesis — diagnose before concluding.
Plan, strictly in this order:
  1. MEASURE: speed bench must profile per-node time split
     (python-chess ops vs encode_board vs GPU call). Decide from data.
  2. BATCH: lockstep-batch MARS rollouts (with B6) — amortizes Python over a
     GPU batch; usually recovers most of the loss without leaving Python.
  3. VECTORIZE: encode_board is a Python loop per token — rewrite with
     precomputed tables / numpy.
  4. PORT (post-validation only): hot loop (move gen + encode) to C/Rust;
     end-state for a serious engine is the LC0 shape — C++ UCI shell driving
     GPU nets. Never port before the architecture is validated.
Training is unaffected (GPU-bound via the Triton kernel) — search/play only.

### B10. Known strengths (audited — do NOT "fix", they are already right)
- Encoding includes halfmove-clock bucket AND repetition flags (rule-50/rep
  awareness most hobby engines lack) — `encoding.py:163-217`.
- Values are side-to-move relative and WDL-antisymmetric (tested in test_v3).
- MCTS is batched with virtual loss (`batched_mcts.py`).
- Terminal/legality/transposition facts come from python-chess + zobrist, not
  the net ("rules are free").
- Syzygy WDL already integrated into evaluation (`tablebase.py`).

---

## Suggested order of attack (after MARS validation)

1. B1 mate-distance + B7 root DTZ        (correctness: stop drawing won games)
2. A4 SPRT + B8 opening suite            (measurement: make every claim real)
3. B3 time management + B2 reuse         (the big practical match-play Elo)
4. A2 SEE + A1 extensions                (search quality inside MARS)
5. B6 inference speed (bf16, batched rollouts)
6. B9 parameter sweep                    (last: tune the whole stack)
7. A3 ordering priors + B4 pondering     (polish)
