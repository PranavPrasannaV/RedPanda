
"""
ChessMamba v3 - Supervised training pipeline (single-GPU / RTX 3060 Ti).

Phases:
    eval     Train the Evaluation Mamba (policy KL + WDL CE + eval-bucket
             contrastive + strategy BCE + action-value MSE + uncertainty aux)
    vectors  Compute phasexcontext geometric advantage vectors from embeddings
    search   Train the Search Mamba on PV lines (value MSE + ranking)
    all      eval -> vectors -> search

Single 8 GB GPU notes:
    - AMP (fp16 autocast) + gradient checkpointing are ON by default.
    - Default model d=512/L=16 (~55 M) trains comfortably; scale to d=768/L=24
      with --d-model 768 --n-layer 24 (uses more VRAM - drop --batch-size).
    - Use --grad-accum to grow the effective batch without more VRAM.

Usage:
    python train.py --phase eval --data data/training --epochs 12 --batch-size 48
    python train.py --phase all  --data data/training
"""

import os
# Reduce CUDA fragmentation (set BEFORE the allocator initialises). Helps the
# big (B,L,d_inner,d_state) scan tensors fit on an 8 GB card.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import time
import json
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import asdict
import numpy as np

from mamba import MambaConfig
from model import ChessMamba
from search_mamba import SearchMamba, SearchMambaConfig
from encoding import encoder, ACTION_SPACE
from data.dataset import build_dense_policy
from torch_compile_wrapper import maybe_compile

# Live progress bar (shows it/s + ETA per batch). Falls back to the plain
# iterable if tqdm isn't installed, so training never depends on it.
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _progress(iterable, **kw):
    return tqdm(iterable, **kw) if tqdm is not None else iterable


def _make_adamw(params, lr, weight_decay=0.01, betas=(0.9, 0.98), device="cuda"):
    """AdamW, fused on CUDA when available (one kernel for the whole step —
    identical update formula, just faster). Falls back transparently."""
    if device == "cuda":
        try:
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay,
                               betas=betas, fused=True)
        except (RuntimeError, TypeError, ValueError):
            pass
    return optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)

# Tier A Speed Fix: Enable TF32 for fast Ampere matmuls
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# torch.compile mode, set by main() from --compile. None/"off" => no compile.
# NOTE: "max-autotune"/"reduce-overhead" enable CUDA graphs, which PIN a memory
# pool and OOM on an 8 GB card (and max-autotune-gemm needs more SMs than a 3060
# Ti has). So compile is OFF by default; when on we use plain "default" (Inductor
# fusion, NO cuda graphs). It does not change the model's math (zero Elo impact).
COMPILE_MODE = None


def _compile(model):
    if COMPILE_MODE in (None, "off"):
        return model
    return maybe_compile(model, mode=COMPILE_MODE)


def eval_config_for(args, vocab_size):
    # Eval Mamba: bidirectional encoder + negative-eigenvalue tracking channels (v4).
    return MambaConfig(
        d_model=args.d_model, n_layer=args.n_layer, d_state=args.d_state,
        n_track_state=args.n_track_state, bidirectional=not args.no_bidirectional,
        mimo_p=args.mimo_p, use_complex=True, use_bcnorm=True,
        vocab_size=vocab_size,
        # BOTH checkpointing levels ON by default — they are complementary, not
        # substitutes:
        #   layer ckpt -> frees activations ACROSS the 16-layer depth (the big
        #                 forward-pass win; without it all layers' activations
        #                 stay alive at once and OOM before backward even starts)
        #   scan ckpt  -> bounds the peak WITHIN one scan during recompute (fixes
        #                 the backward OOM)
        scan_checkpoint=not args.no_checkpoint,
        grad_checkpoint=not (args.no_checkpoint or args.no_layer_ckpt),
        use_triton_scan=args.triton,
    )


# ─── Phase 1: Evaluation Mamba ───────────────────────────────────────────────

class EvalMambaTrainer:
    def __init__(self, data_dir, config: MambaConfig, batch_size=48, lr=3e-4,
                 device=None, grad_accum=1, num_workers=4, bf16=False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.num_workers = num_workers
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.config = config
        self.model = ChessMamba(config, action_space=ACTION_SPACE).to(self.device)
        self.model = _compile(self.model)
        self.optimizer = _make_adamw(self.model.parameters(), lr, device=self.device)
        self.use_amp = self.device == "cuda"
        # bf16: fp32's exponent RANGE at half precision — the fp16 overflow
        # class (values > 65504 -> inf -> nan) cannot occur, and no loss
        # scaling is needed (GradScaler disabled becomes a pass-through).
        self.amp_dtype = torch.bfloat16 if bf16 else torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)

        # Loss weights
        self.w_policy, self.w_wdl = 1.0, 1.0
        self.w_contrastive, self.w_strategy = 0.5, 0.3
        self.w_action_value, self.w_uncertainty = 0.5, 0.1
        self.temperature = 0.1
        # eval-bucket boundaries (side-to-move value in [-1,1])
        self.value_buckets = torch.tensor([-0.6, -0.2, 0.2, 0.6])

        self.best_val_loss = float("inf")
        self.nan_skips = 0
        self.model_path = "chess_mamba.pt"
        self.config_path = "chess_mamba_config.json"
        self.progress_file = "training_progress.json"
        amp_desc = ("bf16" if self.amp_dtype == torch.bfloat16 else "fp16") \
            if self.use_amp else "off"
        print(f"Eval Mamba: {sum(p.numel() for p in self.model.parameters()):,} params "
              f"| device={self.device} | amp={amp_desc} | "
              f"ckpt(scan={getattr(config, 'scan_checkpoint', True)},"
              f"layer={config.grad_checkpoint})")

    def create_dataloaders(self):
        from data.dataset import create_dataloaders
        return create_dataloaders(self.data_dir, batch_size=self.batch_size,
                                  num_workers=self.num_workers, val_split=0.02)

    # ── Loss ──

    @staticmethod
    def _trim_pad(inputs):
        """Drop trailing all-PAD(0) columns — boards are padded to a fixed width
        (160) but real content is ~107 tokens, so this cuts scan memory ~30%."""
        nonpad = (inputs != 0).any(dim=0)
        if bool(nonpad.any()):
            L = int(nonpad.nonzero().max().item()) + 1
            return inputs[:, :L]
        return inputs

    def _compute_loss(self, batch):
        inputs = self._trim_pad(batch["input_ids"].to(self.device, non_blocking=True))
        wdl_t = batch["wdl"].to(self.device, non_blocking=True)
        value_t = batch["value"].to(self.device, non_blocking=True)
        pmoves = batch["policy_moves"].to(self.device, non_blocking=True)
        pprobs = batch["policy_probs"].to(self.device, non_blocking=True)
        best = batch["best_move"].to(self.device, non_blocking=True)
        phase = batch["phase"].to(self.device, non_blocking=True)
        strat_t = batch.get("strategy")
        if strat_t is not None:
            strat_t = strat_t.to(self.device, non_blocking=True)

        out = self.model(inputs, return_dict=True)

        # 1. Policy KL vs soft target
        target = build_dense_policy(pmoves, pprobs, ACTION_SPACE).clamp(min=1e-9)
        logp = F.log_softmax(out["policy"], dim=-1)
        policy_loss = F.kl_div(logp, target, reduction="batchmean")

        # 2. WDL cross-entropy (soft)
        wdl_loss = -(wdl_t * F.log_softmax(out["wdl_raw"], dim=-1)).sum(-1).mean()

        # 3. Eval-bucket + phase contrastive
        con_loss = self._supervised_contrastive(out["embedding"], value_t, phase)

        # 4. Strategy BCE (auxiliary, weight 0.3). The OLD torch.logit(clamp(p))
        # round-trip produced +inf->NaN AND an exploding gradient (d/dp logit =
        # 1/(p(1-p))) into the SHARED backbone whenever the strategy sigmoid
        # saturated — a likely cause of stalled policy learning. Use plain fp32
        # BCE, and if the head output is non-finite (saturated/corrupted) drop
        # ONLY this term instead of NaN-ing the whole batch and halting ALL
        # training (which is what froze the run).
        if strat_t is not None and torch.isfinite(out["strategy"]).all():
            # Manual BCE: F.binary_cross_entropy is BANNED inside autocast
            # (PyTorch raises). All-fp32 ops + the clamp keep every log finite.
            strat_p = out["strategy"].float().clamp(1e-5, 1 - 1e-5)
            st = strat_t.float()
            strategy_loss = -(st * strat_p.log()
                              + (1 - st) * (1 - strat_p).log()).mean()
        else:
            strategy_loss = torch.zeros((), device=self.device)

        # 5. Action-value MSE on the best move (only labelled move)
        av = out["action_value"]
        av_best = av.gather(1, best.unsqueeze(1)).squeeze(1)
        action_value_loss = F.mse_loss(av_best, value_t.clamp(-1, 1))

        # 6. Uncertainty aux: predict 1 - top1 soft-policy mass (many good moves -> uncertain)
        with torch.no_grad():
            top1 = target.max(dim=-1).values
            unc_target = (1.0 - top1).clamp(0, 1)
        unc_loss = F.mse_loss(out["uncertainty"].squeeze(-1), unc_target)

        total = (self.w_policy * policy_loss + self.w_wdl * wdl_loss
                 + self.w_contrastive * con_loss + self.w_strategy * strategy_loss
                 + self.w_action_value * action_value_loss
                 + self.w_uncertainty * unc_loss)
        parts = {"policy": policy_loss.item(), "wdl": wdl_loss.item(),
                 "con": con_loss.item(), "strat": strategy_loss.item(),
                 "av": action_value_loss.item(), "unc": unc_loss.item()}
        # Live train top1 (metric only, ~free: logits already computed). Gives
        # real-time signal instead of waiting an epoch for validation.
        with torch.no_grad():
            parts["t1"] = (out["policy"].argmax(-1) == best).float().mean().item()
        return total, parts

    def _supervised_contrastive(self, emb, value, phase):
        B = emb.shape[0]
        if B < 4:
            return torch.zeros((), device=emb.device)
        emb = F.normalize(emb, dim=-1)
        sim = (emb @ emb.t()) / self.temperature
        bucket = torch.bucketize(value, self.value_buckets.to(value.device))
        same_bucket = bucket.unsqueeze(0) == bucket.unsqueeze(1)
        same_phase = phase.unsqueeze(0) == phase.unsqueeze(1)
        pos = (same_bucket & same_phase).float()
        eye = torch.eye(B, device=emb.device)
        pos = pos * (1 - eye)
        npos = pos.sum(-1)
        has = npos > 0
        if not has.any():
            return torch.zeros((), device=emb.device)
        sim_max = sim.max(dim=-1, keepdim=True).values
        exp_sim = torch.exp(sim - sim_max) * (1 - eye)
        log_den = torch.log(exp_sim.sum(-1, keepdim=True) + 1e-8) + sim_max
        log_prob = sim - log_den
        loss = -(pos * log_prob).sum(-1)[has] / npos[has]
        return loss.mean()

    # ── Loop ──

    def train_epoch(self, loader, epoch):
        self.model.train()
        agg = {"loss": 0, "policy": 0, "wdl": 0, "con": 0, "strat": 0, "av": 0,
               "unc": 0, "t1": 0}
        nb = 0
        self.optimizer.zero_grad(set_to_none=True)
        pbar = _progress(loader, total=len(loader), desc=f"E{epoch}",
                         unit="batch", dynamic_ncols=True)
        for i, batch in enumerate(pbar):
            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                loss, parts = self._compute_loss(batch)
                loss = loss / self.grad_accum
            cur = loss.item() * self.grad_accum
            # Non-finite guard: one fp16-overflow batch must not poison the
            # epoch stats or (via a partial accumulation) the next step. Skip
            # it, report WHICH component blew up, and keep training.
            if not math.isfinite(cur):
                self.nan_skips += 1
                if self.nan_skips <= 10:
                    msg = (f"  [warn] non-finite loss at batch {i+1} "
                           f"(parts: {parts}) — batch skipped "
                           f"[{self.nan_skips} total]")
                    (tqdm.write(msg) if tqdm is not None else print(msg))
                # MUST free this batch's autograd graph (normally backward does
                # it) or it stays alive through the next forward -> OOM.
                del loss
                self.optimizer.zero_grad(set_to_none=True)
                continue
            self.scaler.scale(loss).backward()
            if (i + 1) % self.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            agg["loss"] += cur
            for k in parts:
                agg[k] += parts[k]
            nb += 1
            # Live loss on the bar (tqdm shows it/s + ETA itself).
            if hasattr(pbar, "set_postfix") and (i + 1) % 10 == 0:
                pbar.set_postfix(loss=f"{agg['loss']/nb:.4f}",
                                 cur=f"{cur:.3f}", refresh=False)
            # Full component breakdown every 200 batches, without breaking the bar.
            if (i + 1) % 200 == 0:
                msg = (f"  E{epoch} {i+1}/{len(loader)} | loss {agg['loss']/nb:.4f} "
                       f"(P {agg['policy']/nb:.3f} W {agg['wdl']/nb:.3f} "
                       f"C {agg['con']/nb:.3f} S {agg['strat']/nb:.3f} "
                       f"AV {agg['av']/nb:.3f} U {agg['unc']/nb:.3f}) "
                       f"| top1 {agg['t1']/nb:.1%}")
                (tqdm.write(msg) if tqdm is not None else print(msg))
        return agg["loss"] / max(nb, 1)

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        tot, nb, c1, c5, npos = 0, 0, 0, 0, 0
        for batch in _progress(loader, total=len(loader), desc="  val",
                               unit="batch", dynamic_ncols=True, leave=False):
            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                loss, _ = self._compute_loss(batch)
            # Skip non-finite val batches: one must not poison the whole val
            # number (it breaks best-ckpt tracking and the stop decision).
            lv = loss.item()
            if math.isfinite(lv):
                tot += lv; nb += 1
            inputs = self._trim_pad(batch["input_ids"].to(self.device))
            best = batch["best_move"].to(self.device)
            logits = self.model(inputs)[0]
            top5 = logits.topk(5, dim=-1).indices
            c1 += (logits.argmax(-1) == best).sum().item()
            c5 += (top5 == best.unsqueeze(-1)).any(-1).sum().item()
            npos += inputs.shape[0]
        return tot / max(nb, 1), c1 / max(npos, 1), c5 / max(npos, 1)

    def save_checkpoint(self, suffix=""):
        path = self.model_path if not suffix else self.model_path.replace(".pt", f"_{suffix}.pt")
        torch.save(self.model.state_dict(), path)
        with open(path.replace(".pt", "_config.json"), "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        # canonical config for inference loaders
        with open(self.config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)

    def load_checkpoint(self, path=None):
        path = path or self.model_path
        if os.path.exists(path):
            sd = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(sd)
            print(f"  resumed from {path}")
            # Revive any non-finite weights baked into the checkpoint (e.g. a
            # strategy head that overflowed under the old logit-roundtrip loss).
            # Zeroing them lets that head retrain cleanly; no-op when healthy.
            nbad = 0
            with torch.no_grad():
                for p in self.model.parameters():
                    bad = ~torch.isfinite(p)
                    if bad.any():
                        nbad += int(bad.sum().item())
                        p[bad] = 0.0
            if nbad:
                print(f"  sanitized {nbad} non-finite weights from checkpoint")
            return True
        return False

    def train(self, num_epochs=12, resume=False):
        print("\n" + "=" * 60 + f"\nPhase 1: Eval Mamba ({num_epochs} epochs)\n" + "=" * 60)
        if resume:
            self.load_checkpoint()
        train_loader, val_loader = self.create_dataloaders()
        sched = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)
        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            tr = self.train_epoch(train_loader, epoch)
            vl, a1, a5 = self.validate(val_loader)
            sched.step()
            print(f"\nEpoch {epoch}/{num_epochs} | train {tr:.4f} | val {vl:.4f} | "
                  f"top1 {a1:.1%} | top5 {a5:.1%} | {time.time()-t0:.0f}s")
            if math.isfinite(vl) and vl < self.best_val_loss:
                self.best_val_loss = vl
                self.save_checkpoint("best")
                print(f"  * new best (val {vl:.4f})")
            self.save_checkpoint()
            json.dump({"phase": "eval", "epoch": epoch, "total_epochs": num_epochs,
                       "train_loss": tr, "val_loss": vl, "top1": a1, "top5": a5,
                       "best_val_loss": self.best_val_loss},
                      open(self.progress_file, "w"), indent=2)
        print(f"\n[OK] Phase 1 done. best val {self.best_val_loss:.4f}")
        return self.model


# ─── Phase 2: Advantage vectors ──────────────────────────────────────────────

def compute_and_save_advantage_vectors(model, data_dir, device="cpu"):
    from geometric import compute_advantage_vectors, save_advantage_vectors
    print("\n" + "=" * 60 + "\nPhase 2: Advantage vectors\n" + "=" * 60)
    inputs = np.load(os.path.join(data_dir, "inputs.npy"), mmap_mode="r")
    wdl = np.load(os.path.join(data_dir, "wdl.npy"), mmap_mode="r")
    phases = np.load(os.path.join(data_dir, "phases.npy"), mmap_mode="r")
    sp = os.path.join(data_dir, "strategy_labels.npy")
    strat = np.load(sp, mmap_mode="r") if os.path.exists(sp) else None
    vectors = compute_advantage_vectors(model, inputs, wdl, phases, strat, device=device)
    save_advantage_vectors(vectors, "advantage_vectors.pt")
    print("[OK] Phase 2 done.")
    return vectors


# ─── Phase 3: Search Mamba ───────────────────────────────────────────────────

class SearchMambaTrainer:
    def __init__(self, data_dir, config: SearchMambaConfig, batch_size=64,
                 lr=1e-4, device=None, num_workers=4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.model = SearchMamba(config, vocab_size=config.vocab_size).to(self.device)
        self.model = _compile(self.model)
        self.optimizer = _make_adamw(self.model.parameters(), lr,
                                     betas=(0.9, 0.999), device=self.device)
        self.use_amp = self.device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.model_path = "search_mamba.pt"
        print(f"Search Mamba: {sum(p.numel() for p in self.model.parameters()):,} params")

    def create_dataloader(self):
        from data.dataset import PVLineDataset
        from torch.utils.data import DataLoader, random_split
        ds = PVLineDataset(self.data_dir, max_pv_length=8)
        val = max(1, int(len(ds) * 0.02))
        tr, va = random_split(ds, [len(ds) - val, val])
        mk = lambda s, sh: DataLoader(s, batch_size=self.batch_size, shuffle=sh,
                                      num_workers=self.num_workers, pin_memory=True)
        return mk(tr, True), mk(va, False)

    def _loss(self, board_enc, move_seq, quality):
        board_enc = board_enc.to(self.device); move_seq = move_seq.to(self.device)
        quality = quality.to(self.device)
        pred = torch.sigmoid(self.model(board_enc, move_seq))
        value_loss = F.mse_loss(pred, quality)
        # Ranking: real PV should outscore a shuffled (mismatched) line.
        neg = self.model(board_enc, move_seq.roll(1, dims=0))
        rank_loss = F.relu(0.1 - (self.model(board_enc, move_seq) - neg)).mean()
        return value_loss + 0.5 * rank_loss

    def train(self, num_epochs=8):
        print("\n" + "=" * 60 + "\nPhase 3: Search Mamba\n" + "=" * 60)
        tr_loader, va_loader = self.create_dataloader()
        sched = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)
        best = float("inf")
        for epoch in range(1, num_epochs + 1):
            self.model.train(); t0 = time.time(); tot = 0; nb = 0
            for i, (b, m, q) in enumerate(tr_loader):
                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    loss = self._loss(b, m, q)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer); self.scaler.update()
                tot += loss.item(); nb += 1
                if (i + 1) % 200 == 0:
                    print(f"  E{epoch} {i+1}/{len(tr_loader)} | loss {tot/nb:.4f}")
            self.model.eval(); vt = 0; vb = 0
            with torch.no_grad():
                for b, m, q in va_loader:
                    with torch.amp.autocast("cuda", enabled=self.use_amp):
                        vt += self._loss(b, m, q).item(); vb += 1
            vl = vt / max(vb, 1)
            print(f"Epoch {epoch}/{num_epochs} | train {tot/max(nb,1):.4f} | "
                  f"val {vl:.4f} | {time.time()-t0:.0f}s")
            if vl < best:
                best = vl
                self.model.save_pretrained(self.model_path)
                print(f"  * saved (val {vl:.4f})")
        print(f"\n[OK] Phase 3 done. best val {best:.4f}")
        return self.model


# ─── Phase 3b: Value-equivalent dynamics (v4, for MARS) ──────────────────────

class DynamicsTrainer:
    """
    Train the Search Mamba as a value-equivalent recurrent dynamics model:
        L = MSE(value) + CE(continuation) + lambda_c * (1 - cos) consistency.
    The consistency term (EfficientZero) pulls each rollout hidden state toward the
    frozen Eval Mamba's embedding of the TRUE next position -- the ingredient that
    keeps MARS's deep recurrent rollouts accurate.
    """

    def __init__(self, data_dir, eval_ckpt, eval_cfg_path, search_config,
                 batch_size=64, lr=1e-4, device=None, num_workers=4,
                 lambda_c=0.5, max_pv=8, seq_len=160, bf16=False, triton=False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_pv = max_pv
        self.seq_len = seq_len
        self.lambda_c = lambda_c
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Frozen, trained Eval Mamba -> consistency target embeddings. This is
        # the DOMINANT cost of dynamics training (B x T board re-encodings per
        # batch), so it gets the fused kernel too. Triton is set EXPLICITLY
        # from the flag (never trusted from the saved config json).
        cd = json.load(open(eval_cfg_path))
        ecfg = MambaConfig(**{k: v for k, v in cd.items()
                              if k in MambaConfig.__dataclass_fields__})
        ecfg.use_triton_scan = triton
        self.eval_model = ChessMamba(ecfg, action_space=ACTION_SPACE).to(self.device)
        self.eval_model.load_state_dict(torch.load(eval_ckpt, map_location=self.device,
                                                   weights_only=True))
        self.eval_model.eval()
        self.eval_model = _compile(self.eval_model)
        for p in self.eval_model.parameters():
            p.requires_grad_(False)

        self.search = SearchMamba(search_config, vocab_size=search_config.vocab_size,
                                  eval_dim=ecfg.d_model).to(self.device)
        self.search = _compile(self.search)
        self.optimizer = _make_adamw(self.search.parameters(), lr,
                                     betas=(0.9, 0.999), device=self.device)
        self.use_amp = self.device == "cuda"
        # Same precision policy as the Eval trainer: bf16 kills the fp16
        # overflow class; GradScaler disabled under bf16 (pass-through).
        self.amp_dtype = torch.bfloat16 if bf16 else torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)
        self.model_path = "search_mamba.pt"
        self.nan_skips = 0
        amp_desc = ("bf16" if self.amp_dtype == torch.bfloat16 else "fp16") \
            if self.use_amp else "off"
        print(f"Dynamics (Search Mamba): {sum(p.numel() for p in self.search.parameters()):,} params "
              f"| eval_dim={ecfg.d_model} | amp={amp_desc} | triton={triton}")

    def create_dataloader(self):
        from data.dataset import DynamicsDataset
        from torch.utils.data import DataLoader, random_split
        ds = DynamicsDataset(self.data_dir, max_pv=self.max_pv, seq_len=self.seq_len)
        val = max(1, int(len(ds) * 0.02))
        tr, va = random_split(ds, [len(ds) - val, val])
        pin = self.device == "cuda"
        mk = lambda s, sh: DataLoader(s, batch_size=self.batch_size, shuffle=sh,
                                      num_workers=self.num_workers, pin_memory=pin,
                                      drop_last=sh)
        return mk(tr, True), mk(va, False)

    def _loss(self, batch):
        dev = self.device
        board = batch["board_enc"].to(dev)
        moves = batch["move_tokens"].to(dev)
        vtgt = batch["value_tgt"].to(dev)
        nmove = batch["next_move"].to(dev)
        inter = batch["inter_enc"].to(dev)
        vmask = batch["vmask"].to(dev)
        cmask = batch["cmask"].to(dev)
        B, T = moves.shape

        v_hat, cont, feats = self.search.dynamics_rollout(board, moves)

        value_loss = (vmask * (v_hat - vtgt) ** 2).sum() / vmask.sum().clamp(min=1)

        ce = F.cross_entropy(cont.reshape(B * T, -1), nmove.reshape(B * T),
                             reduction="none").reshape(B, T)
        cont_loss = (cmask * ce).sum() / cmask.sum().clamp(min=1)

        with torch.no_grad():
            emb = self.eval_model.get_embedding(inter.reshape(B * T, -1)).reshape(B, T, -1)
        cos = F.cosine_similarity(feats, emb, dim=-1)
        consist_loss = (vmask * (1.0 - cos)).sum() / vmask.sum().clamp(min=1)

        total = value_loss + cont_loss + self.lambda_c * consist_loss
        return total, {"value": value_loss.item(), "cont": cont_loss.item(),
                       "consist": consist_loss.item()}

    def train(self, num_epochs=8, max_steps=0, save_every=2000):
        """
        max_steps:  stop after this many optimizer steps (0 = run all epochs).
                    Enables a quick 'good-enough Search Mamba' for early MARS
                    validation without committing to a full multi-day epoch.
        save_every: ALSO checkpoint every N steps mid-epoch (the epoch-end save
                    remains), so a partial run is never wasted.
        """
        print("\n" + "=" * 60 + "\nPhase 3: Search Mamba (value-equivalent dynamics)\n" + "=" * 60)
        if max_steps:
            print(f"  (quick mode: stopping after {max_steps} steps; "
                  f"checkpoint every {save_every})")
        tr_loader, va_loader = self.create_dataloader()
        sched = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)
        best = float("inf")
        gstep = 0
        for epoch in range(1, num_epochs + 1):
            self.search.train(); t0 = time.time(); tot = 0; nb = 0
            pbar = _progress(tr_loader, total=len(tr_loader), desc=f"E{epoch}",
                             unit="batch", dynamic_ncols=True)
            for i, batch in enumerate(pbar):
                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=self.use_amp,
                                        dtype=self.amp_dtype):
                    loss, parts = self._loss(batch)
                cur = loss.item()
                if not math.isfinite(cur):
                    self.nan_skips += 1
                    if self.nan_skips <= 10:
                        msg = (f"  [warn] non-finite dynamics loss at step {gstep+1} "
                               f"(parts: {parts}) — batch skipped [{self.nan_skips} total]")
                        (tqdm.write(msg) if tqdm is not None else print(msg))
                    del loss
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.search.parameters(), 1.0)
                self.scaler.step(self.optimizer); self.scaler.update()
                tot += cur; nb += 1; gstep += 1
                if hasattr(pbar, "set_postfix") and (i + 1) % 10 == 0:
                    pbar.set_postfix(loss=f"{tot/nb:.4f}", cur=f"{cur:.3f}",
                                     refresh=False)
                if (i + 1) % 200 == 0:
                    msg = (f"  E{epoch} {i+1}/{len(tr_loader)} | loss {tot/nb:.4f} "
                           f"(V {parts['value']:.3f} C {parts['cont']:.3f} K {parts['consist']:.3f})")
                    (tqdm.write(msg) if tqdm is not None else print(msg))
                if save_every and gstep % save_every == 0:
                    self.search.save_pretrained(self.model_path)
                    msg = f"  [ckpt] saved {self.model_path} at step {gstep}"
                    (tqdm.write(msg) if tqdm is not None else print(msg))
                if max_steps and gstep >= max_steps:
                    self.search.save_pretrained(self.model_path)
                    print(f"\n[OK] Phase 3 quick mode: {gstep} steps done, "
                          f"avg loss {tot/max(nb,1):.4f}, saved {self.model_path}")
                    return self.search
            self.search.eval(); vt = 0; vb = 0
            with torch.no_grad():
                for batch in va_loader:
                    with torch.amp.autocast("cuda", enabled=self.use_amp,
                                            dtype=self.amp_dtype):
                        lv = self._loss(batch)[0].item()
                    if math.isfinite(lv):
                        vt += lv; vb += 1
            vl = vt / max(vb, 1); sched.step()
            print(f"Epoch {epoch}/{num_epochs} | train {tot/max(nb,1):.4f} | val {vl:.4f} | {time.time()-t0:.0f}s")
            if math.isfinite(vl) and vl < best:
                best = vl
                self.search.save_pretrained(self.model_path)
                print(f"  * saved (val {vl:.4f})")
        print(f"\nPhase 3 done. best val {best:.4f}")
        return self.search


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="ChessMamba v3 training")
    p.add_argument("--phase", choices=["eval", "vectors", "search", "all"], default="all")
    p.add_argument("--data", default="data/training")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16,
                   help="8GB tip: bidirectional needs ~8-12; use --grad-accum to "
                        "raise the effective batch")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layer", type=int, default=16)
    p.add_argument("--d-state", type=int, default=64)
    p.add_argument("--mimo-p", type=int, default=4)
    p.add_argument("--no-checkpoint", action="store_true",
                   help="disable ALL gradient checkpointing (fastest, max VRAM - big GPUs only)")
    p.add_argument("--no-layer-ckpt", action="store_true",
                   help="keep scan ckpt but drop LAYER ckpt (more speed, more VRAM - needs headroom)")
    p.add_argument("--triton", action="store_true",
                   help="use the fused Triton scan kernel (verifies correctness on this GPU "
                        "before training starts; aborts on any mismatch)")
    p.add_argument("--bf16", action="store_true",
                   help="autocast in bfloat16 instead of float16: same speed on Ampere+, "
                        "fp32 exponent range, so fp16-overflow NaNs cannot occur")
    p.add_argument("--max-steps", type=int, default=0,
                   help="(phase search) stop dynamics training after N steps - quick "
                        "'good-enough Search Mamba' for early MARS validation (0 = full)")
    p.add_argument("--save-every", type=int, default=2000,
                   help="(phase search) also checkpoint every N steps mid-epoch")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default=None)
    # v4 backbone
    p.add_argument("--n-track-state", type=int, default=16,
                   help="negative-eigenvalue state-tracking channels (Eval+Search)")
    p.add_argument("--no-bidirectional", action="store_true",
                   help="disable the Eval Mamba's bidirectional board scan")
    p.add_argument("--lambda-consist", type=float, default=0.5,
                   help="weight of the EfficientZero consistency loss (dynamics)")
    # search-mamba scaling
    p.add_argument("--search-d-model", type=int, default=384)
    p.add_argument("--search-n-layer", type=int, default=12)
    p.add_argument("--compile", default="off",
                   choices=["off", "default", "reduce-overhead", "max-autotune"],
                   help="torch.compile mode. 'off' (default) is safest on 8 GB — "
                        "the cuda-graph modes (reduce-overhead/max-autotune) pin a "
                        "memory pool and OOM. If you try compile, use 'default'.")
    args = p.parse_args()

    global COMPILE_MODE
    COMPILE_MODE = args.compile
    if COMPILE_MODE not in (None, "off"):
        print(f"[train] torch.compile mode = {COMPILE_MODE} (experimental on 8 GB)")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    vocab = encoder.vocab_size() + 100

    if args.triton:
        # Correctness gate: the fused kernel must reproduce the reference
        # path's forward AND gradients on THIS GPU, or we refuse to train.
        from triton_scan import HAS_TRITON, verify as verify_triton
        if not HAS_TRITON or device != "cuda":
            sys.exit("[train] --triton requires CUDA + the triton package. Aborting.")
        print("[train] verifying Triton scan kernel against the reference path...")
        verify_triton(device=device, thorough=False, verbose=False)
        print("[train] Triton scan verified (forward + gradients match). Using fused kernel.")

    if args.phase in ("eval", "all"):
        trainer = EvalMambaTrainer(
            args.data, eval_config_for(args, vocab), batch_size=args.batch_size,
            lr=args.lr, device=device, grad_accum=args.grad_accum,
            num_workers=args.num_workers, bf16=args.bf16)
        model = trainer.train(num_epochs=args.epochs, resume=args.resume)

    if args.phase in ("vectors", "all"):
        if args.phase == "vectors":
            cfg = eval_config_for(args, vocab)
            cpath = "chess_mamba_config.json"
            if os.path.exists(cpath):
                cd = json.load(open(cpath))
                cfg = MambaConfig(**{k: v for k, v in cd.items()
                                     if k in MambaConfig.__dataclass_fields__})
                cfg.vocab_size = vocab
            model = ChessMamba(cfg, action_space=ACTION_SPACE).to(device)
            model.load_state_dict(torch.load("chess_mamba.pt", map_location=device,
                                              weights_only=True))
        model.eval()
        compute_and_save_advantage_vectors(model, args.data, device=device)

    if args.phase in ("search", "all"):
        scfg = SearchMambaConfig(
            d_model=args.search_d_model, n_layer=args.search_n_layer,
            d_state=32, n_track_state=min(args.n_track_state, 31), mimo_p=2,
            vocab_size=vocab, bidirectional=False,   # MUST stay causal for MARS
            scan_checkpoint=not args.no_checkpoint,
            grad_checkpoint=not (args.no_checkpoint or args.no_layer_ckpt),
            use_triton_scan=args.triton)
        DynamicsTrainer(
            args.data, eval_ckpt="chess_mamba.pt",
            eval_cfg_path="chess_mamba_config.json", search_config=scfg,
            batch_size=max(32, args.batch_size), device=device,
            num_workers=args.num_workers, lambda_c=args.lambda_consist,
            bf16=args.bf16, triton=args.triton).train(
                num_epochs=min(args.epochs, 8),
                max_steps=args.max_steps, save_every=args.save_every)

    print("\nAll requested phases complete.")


if __name__ == "__main__":
    main()
