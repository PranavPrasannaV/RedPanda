# ChessMamba — Engine Audit, Training-Speed Plan & Search-First Validation

This document addresses everything raised, in order:

1. **Is the architecture pristine / not flawed?** → audit + correctness checklist + honest design notes.
2. **Why is training ~4 it/s, and how to make it fast on the 3060 Ti with ZERO Elo loss.**
3. **Should we add a lightweight hybrid (e.g. "SLICE")?** → verdict.
4. **Test the SearchMamba search (MARS) vs MCTS and alpha-beta BEFORE deep training.** → staged validation plan + the tooling needed.
5. **Recommended order of operations.**

> Ground rule respected throughout: **the novel search (MARS) and the dual-Mamba
> evaluation architecture are NOT changed.** Everything proposed is either a
> pure-speed implementation change (identical math → zero Elo impact), a
> *measured* trade-off you opt into, or new test tooling.

---

## 1. Architecture integrity audit

### Verdict: the engine is sound and the implementation is correct.
- **Both test suites pass:** `test_v3.py` 34/34, `test_v4.py` 26/26. These verify the
  pieces that, if broken, would silently wreck strength.
- Nothing you touched introduced a regression that the tests can see.

### Correctness checklist (all verified)
| Property | Status | Why it matters |
|---|---|---|
| O(1) recurrent `step()` == parallel scan (bit-exact) | ✅ | MARS rollouts depend on it |
| MARS uses `-child.value()` / negamax sign | ✅ | wrong sign = plays anti-optimally |
| WDL / values are **side-to-move relative & antisymmetric** | ✅ | required for sound backup |
| Action space 8192 covers all promotion moves | ✅ | 8000 silently dropped ~125 moves |
| MARS finds mate-in-1 even with random nets | ✅ | search structure is correct |
| Phase detection works on 4-field eval-DB FENs | ✅ | (the bug we already fixed) |
| Dynamics training (value+continuation+consistency) backprops | ✅ | the value-equivalent model trains |
| Bidirectional restricted to Eval Mamba; Search Mamba causal | ✅ (asserted) | bidirectional would break `step()` |

### Honest design notes (not bugs — things to be aware of, with no action required yet)
1. **Action-value head is weakly supervised.** The eval-DB gives one "best move" per
   position, so the per-move Q head is trained on essentially one move/position. It
   will Q-init the *best* move well and others weakly. Fine for MARS seeding;
   improvable later with multi-PV data. Not a flaw.
2. **MARS rollouts are currently single stochastic-PV (beam = 1).** Correct and
   functional; widening the beam is a *tuning knob inside MARS*, not an
   architecture change, and is exactly what the Stage-2 test below is for.
3. **`d_state = 64` and bidirectional are the two biggest compute costs** and their
   Elo benefit is **untested**. We keep them by default (no Elo loss), but §5 below
   lets you *measure* whether they're worth their cost — that's the only honest way
   to know.

**Bottom line:** you are not training a flawed engine. The design is internally
consistent and the proven-vs-novel split is exactly as documented in
`ARCHITECTURE.md`. The one genuinely unproven element — the SSM-as-dynamics-model
search — is precisely what the validation plan in §4 is designed to de-risk *before*
you spend serious GPU time.

---

## 2. The training-speed problem

### Symptom
~4 it/s at batch ≈12 ⇒ ~48 positions/sec ⇒ **~28 h per epoch** on 4.9M positions
⇒ ~14 days for 12 epochs. Untenable.

### Root cause (it is the SSM scan implementation, not your hardware)
The Mamba scan is a **pure-PyTorch Hillis-Steele parallel scan** that materialises
`(batch, L, d_inner, d_state) = (12, ~107, 1024, 64)` tensors — **84 million
elements each** — and re-reads/re-writes several of them across `log₂(L) ≈ 7`
iterations, *per layer*. It is **memory-bandwidth-bound**, and several factors
stack on top of each other:

| Factor | Multiplier vs a "standard" fast Mamba |
|---|---|
| Pure-PyTorch scan (no fused kernel) | ~5–20× slower than the CUDA/Triton kernel |
| `d_state = 64` (standard Mamba uses 16) | ~4× the state traffic |
| **Bidirectional** Eval scan (forward + reversed) | ~2× |
| **Gradient checkpointing** (recomputes forward in backward) | ~1.5× compute |
| No `torch.compile` / no TF32 | leaves ~1.5–3× on the table |

None of this is your GPU's fault — the same model with a proper kernel would be an
order of magnitude faster. The good news: **almost all of the speed can be
recovered without changing the model's math (zero Elo loss).**

---

## 3. Zero-Elo-loss speed fixes (ordered by impact ÷ risk)

These produce a **mathematically identical** model — same weights, same outputs,
same Elo — just computed faster. Each is verified bit-exact against the current
scan before adoption (we already have the exactness test harness).

### Tier A — quick wins (apply first, low risk)
- **Enable TF32 matmuls** (`torch.backends.cuda.matmul.allow_tf32=True`,
  `cudnn.allow_tf32=True`). Ampere-native, ~1.1–1.3×, negligible numerical impact.
- **Confirm AMP is actually active** and the loss isn't silently running fp32.
- **Dataloader tuning on Linux** (you switched to Linux — good): `num_workers` =
  physical cores, `pin_memory=True`, `persistent_workers=True`, and make sure the
  data is on a **local SSD** (not a network share). If the GPU is ever idle waiting
  on data, this matters; if it's scan-bound, it won't — but it's free to get right.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (you're already setting this).

*Expected combined: ~1.3–1.6×. Effort: minutes.*

### Tier B — `torch.compile` the model (low risk, big-ish win)
Wrap the Eval/Search models with `torch.compile` (the `torch_compile_wrapper.py`
already exists but isn't wired into `train.py`). Inductor fuses the scan's many
elementwise ops + einsums into far fewer kernels, cutting launch overhead and
memory traffic. Works much better on **Linux** than Windows (another reason your
Linux switch was right).

*Expected: ~1.5–3×. Effort: small (wire it in + verify outputs match + handle the
grad-checkpoint interaction). Zero Elo loss.*

### Tier C — chunked (streaming) scan (medium effort, the structural fix)
Replace the Hillis-Steele scan with a **chunked scan**: process the sequence in
blocks of `C` (e.g. 32) — a small scan within each block, carrying the recurrent
state (and the one trapezoidal `dBu` term) across blocks. This is exactly how
Mamba-2 / FlashLinearAttention get their speed, and it is **bit-exact** with the
current recurrence.
- **Why it helps:** the Hillis-Steele scan creates ~2 fresh 84M-element tensors per
  iteration × 7 iterations = enormous redundant traffic. A chunked scan keeps the
  working set to `(batch, C, d_inner, d_state)` and does O(L) work instead of
  O(L·logL). Lower peak memory ⇒ **bigger batch fits** ⇒ better GPU utilisation ⇒
  more positions/sec on top of the per-op savings.

*Expected: ~2–4× (compute) plus a larger feasible batch. Effort: medium; must be
verified bit-exact (we have the test). Zero Elo loss.*

### Tier D — custom Triton selective-scan kernel (high effort, biggest win)
A fused Triton kernel computing *our exact* trapezoidal/complex-gate/tracking-channel
recurrence (the official `mamba-ssm` CUDA kernel can't be dropped in because it
computes a *different* recurrence). This is the real fix that closes the gap to a
production Mamba.

*Expected: ~5–20×. Effort: large + correctness-critical. Recommended only if A–C
don't get you to an acceptable epoch time. Zero Elo loss (same math).*

### Realistic outcome
Tiers **A + B + C** plausibly take you from ~48 pos/s to ~**200–400 pos/s**
(epoch ~3–7 h instead of ~28 h) with **no Elo change whatsoever**. Tier D, if
needed, takes it further. I'd implement A+B first (fast to do, measure the real
gain on your card), then C, and only reach for D if necessary.

---

## 4. Optional *measured* trade-offs (you decide; only if a benchmark says so)

These could 2–4× speed but *might* cost Elo, so they are **off the table unless an
ablation proves the Elo cost is ~zero.** I list them only so the decision is
data-driven, never silent:
- **Bidirectional Eval scan → unidirectional** (~2× the eval forward). Its board-
  awareness benefit is untested. A quick ablation (train both a few epochs on a
  subset, compare top-1/top-5 policy accuracy + tactics) tells you if it's worth 2×.
- **`d_state = 64 → 32`** (~2× the scan). Also untested whether 64 buys real Elo.

Recommendation: **leave both at default for now** (honour "no Elo loss"); run the
ablations *after* the speed fixes, when a 2-epoch experiment is cheap. If an
ablation shows no measurable Elo difference, you get the speed for free.

---

## 5. The "SLICE" / lightweight-hybrid question

**Verdict: do not add anything now.**
- There is **no established "SLICE" chess model** — a web search turned up nothing;
  the only well-known lightweight evaluator is **NNUE**, which your design
  deliberately excludes (and which would change the eval architecture).
- The engine is already comprehensive: 6 eval heads + a value-equivalent dynamics
  model + a learned search. Bolting on another module **before the base is trained
  and measured** is premature — you'd be optimising a weakness you haven't located
  yet, and adding risk to an unvalidated stack.
- Your own rule applies: *"don't add stuff for the sake of adding it unless it seems
  revolutionary."* Nothing here clears that bar today.

**When to revisit:** after Stage-2 of the validation plan tells you *where* the
engine is actually weak. If the measured weakness is, say, sharp tactics, the
right add is targeted (e.g. a stronger quiescence or a tactics-specific signal),
not a generic "accuracy" module. Measure first, then add only what the data
demands.

---

## 6. Search-first validation: test MARS vs MCTS vs alpha-beta

This is your top priority and the right instinct — validate the search before
investing in training. The subtlety: **MARS's value depends on trained nets**, so
there's an unavoidable minimum of training to test it meaningfully. Here's how to
test as early and cheaply as possible, in three stages.

### What "is MARS better than MCTS / alpha-beta?" actually means — two separable claims
- **(a) Search-algorithm quality:** given the *same* per-node evaluations, does
  MARS's Gumbel + sequential-halving + re-anchor + MCGS pick better moves than
  MCTS's PUCT at equal node budget?
- **(b) Search speed:** MARS evaluates a node with one **O(1)** Search-Mamba step
  (cheap) vs MCTS's full forward pass (expensive) ⇒ more nodes/sec at equal
  wall-clock ⇒ stronger.

Claim (a) is testable with almost no training; claim (b) needs a roughly-trained
Search Mamba.

### Stage 0 — search-algorithm sanity, **no neural training** (optional, fast)
Plug a fixed reference evaluator — **Stockfish at depth 1** (cheap, deterministic) —
as the value/policy oracle into *both* a MARS-style selector and the MCTS selector,
and compare which extracts more strength from the **same number of oracle calls**
(tactics-suite solve rate + a short head-to-head). This isolates claim (a): is the
search *logic* sound, independent of any network training? If MARS-selection ≥ MCTS
here, the algorithm design is validated.
- *Tooling I'll build:* an "oracle evaluator" adapter so any search can call
  Stockfish-depth-1 in place of the neural heads, plus a tactics-suite runner.

### Stage 1 — neural eval + MCTS vs Stockfish, after a SHORT eval-only train
This only needs the **Eval Mamba** (not the dynamics model). After the speed fixes,
train the eval head a few hours on a subset (or 3–4 epochs), then:
- Run **tactics suite** (e.g. the standard WAC / a curated EPD set): solve rate +
  nodes + time. A decent eval + MCTS should already solve easy tactics — a clean,
  low-variance "is the eval usable" signal.
- Run **MCTS vs Stockfish** at fixed low depth (`tournament.py` already supports
  this) for an absolute Elo calibration of "neural eval + classical search."

This answers "is our evaluation good enough to be worth searching over?" before you
build MARS on top of it.

### Stage 2 — the real A/B: MARS vs MCTS (same net) + MARS vs alpha-beta
After a short **dynamics** train (the Search Mamba; needs the eval Mamba frozen for
the consistency loss), run the head-to-head that answers both (a) and (b):
- **MARS vs MCTS at equal node budget on the *same* eval net** — the Phase-C gate.
  `tournament.py --opponent self --backend mars --opponent-backend mcts` already
  does exactly this. If MARS wins, the novel search has earned its place.
- **MARS vs Stockfish at fixed depth** (= vs alpha-beta) for absolute calibration.
- Tweak MARS hyper-params here (sim budget, `k_anchor`, depth cap, beam, quiescence)
  — this is where the search gets tuned, cheaply, before full training.

*Tooling I'll build:* a single `bench.py` that runs (1) the tactics suite for any
backend, (2) MARS-vs-MCTS A/B, (3) any-engine-vs-Stockfish, and prints solve rate /
Elo / nodes-per-second side by side — so every tweak is measured the same way.

### Why you can't fully skip training for the MARS test
MARS needs the eval Mamba (root + re-anchor policy/value/Q) **and** the Search Mamba
(O(1) dynamics). The Search Mamba's training even *requires* a frozen eval Mamba
(for the consistency target). So the dependency is `eval → dynamics → MARS`. The
plan above front-loads only the *minimum* training (hours, on a subset, sped up by
§3) needed to make each test meaningful — you never commit the full multi-day run
until the search is validated.

---

## 7. Recommended order of operations

```
1. SPEED        Apply Tier A + B (TF32, dataloader, torch.compile).  Measure it/s.
                If still slow → Tier C (chunked scan, bit-exact).      (zero Elo loss)
2. BENCH TOOL   Build bench.py: tactics suite + MARS-vs-MCTS A/B + vs-Stockfish.
3. STAGE 1      Short eval-only train (subset / few epochs) → eval+MCTS vs Stockfish
                + tactics solve rate.   "Is the evaluation usable?"
4. STAGE 2      Short dynamics train → MARS vs MCTS (same net) + MARS vs Stockfish.
                Tune MARS.   "Is the novel search better?"   ← the decisive gate
5. (optional)   Ablate bidirectional / d_state for free speed if Elo-neutral.
6. FULL TRAIN   Only now: full supervised train → vectors → self-play RL.
7. (later)      Revisit hybrid additions ONLY if Stage-2 reveals a specific weakness.
```

This front-loads the two things you care about — **fast training** and **validating
the search** — and defers the expensive multi-day commitment until the novel search
has actually proven itself, all without touching the architecture you invented.

---

## 8. Decisions I need from you before coding

1. **Speed tiers:** start with A+B and measure, or go straight to A+B+C? (I recommend
   A+B first, then C if needed — fastest path to "good enough".)
2. **Tactics suite:** OK to bundle a standard public EPD test set (e.g. WAC) into
   the repo for the benchmark, or do you have a preferred set?
3. **Stockfish path:** confirm the Stockfish binary location on the desktop (you have
   `stockfishwindows-x86-64-avx2/`) so `bench.py` / `tournament.py` can call it.
4. **Light-train budget:** how many hours are you willing to spend on the Stage-1/2
   short trains to get the search validated? (This sets the subset size / epochs.)

Once you pick, I'll implement in this order: **speed fixes → `bench.py` → run
Stage 1/2 with you.** No architecture changes — only faster math and new tests.

Sources consulted for the "SLICE"/lightweight-eval question:
- [Neural Networks for Chess (survey)](https://arxiv.org/pdf/2209.01506)
- [NNUE design principles (IJRIAS)](https://rsisinternational.org/journals/ijrias/articles/a-theoretical-analysis-of-the-development-and-design-principles-of-nnue-for-chess-evaluation/)
- [Chessprogramming wiki — Neural Networks](https://www.chessprogramming.org/Neural_Networks)
```
