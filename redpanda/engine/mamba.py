
"""
Mamba-3 SSM: Pure PyTorch implementation for ChessMamba.

Key upgrades from Mamba-1:
  1. Exponential-Trapezoidal discretization (2nd-order accurate)
  2. Complex-valued state spaces (rotational dynamics via RoPE-like rotations)
  3. MIMO formulation (cross-feature interaction in state updates)
  4. BCNorm (RMSNorm on B,C projections for training stability)
  5. No Conv1d (replaced by biases in B/C projections)

Reference: arXiv:2603.15569 (Mamba-3, March 2026)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from dataclasses import dataclass, field
from einops import rearrange, repeat


@dataclass
class MambaConfig:
    d_model: int = 512
    n_layer: int = 16
    vocab_size: int = 8192
    d_state: int = 64           # Mamba-3 efficient with larger N
    expand: int = 2
    dt_rank: int = 'auto'
    pad_vocab_size_multiple: int = 8
    bias: bool = False

    # ── Mamba-3 specific ──
    use_complex: bool = True    # Complex-valued state transitions
    mimo_p: int = 4             # MIMO input/output channels
    use_bcnorm: bool = True     # RMSNorm on B,C projections

    # ── v4 backbone upgrades ──
    # Negative-eigenvalue state-tracking channels: a slice of the state dim uses a
    # REAL, signed, time-invariant eigenvalue lambda = tanh(theta) in (-1, 1).
    # Eigenvalues near -1 flip sign each step -> they can track parity / discrete
    # state (castling rights, repetition) that decay-only SSMs provably cannot.
    n_track_state: int = 0
    # Bidirectional (forward + reversed) scan for 2D board awareness. MUST stay
    # False for any model that uses step()/prime_cache (the Search Mamba), or the
    # O(1) recurrence is no longer exact. Safe (and recommended) for the Eval Mamba.
    bidirectional: bool = False

    # ── Training-efficiency knobs (critical for single 8GB GPU) ──
    grad_checkpoint: bool = False  # Recompute each layer in backward to save VRAM
    # Recompute the selective scan CHUNK-BY-CHUNK in backward. The scan's
    # (B, chunk, d_inner, d_state) intermediates dominate training memory; with
    # this on, only ONE chunk's working set is ever alive in backward instead of
    # all of them (~L/SCAN_CHUNK x less scan memory). Same math, exact gradients.
    scan_checkpoint: bool = True
    # Fused Triton kernel for the scan (training speed: ~50x less memory
    # traffic; same recurrence, analytic backward). OFF by default — enable
    # only after triton_scan.verify() passes on the GPU (train.py --triton
    # runs that gate automatically).
    use_triton_scan: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        assert 0 <= self.n_track_state < self.d_state, "n_track_state must be < d_state"
        self.n_rot_state = self.d_state - self.n_track_state

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )


# ─── Utility Modules ─────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def einsum(a, b, pattern):
    return torch.einsum(pattern.replace(" ", ""), a, b)


# ─── Parallel Scan ───────────────────────────────────────────────────────────

# Chunk length for the streaming selective scan. Smaller = less peak VRAM (the
# (B, chunk, d_inner, d_state) working set) but more Python-level chunk steps.
# 32 is a good balance for L~107 on an 8 GB card. Same math at any value.
SCAN_CHUNK = 32


def parallel_scan(gates, tokens):
    """
    GPU-parallel inclusive prefix scan for the linear recurrence:
        h[t] = gates[t] * h[t-1] + tokens[t],   h[-1] = 0

    Hillis-Steele algorithm: O(log L) sequential steps, each fully
    parallelised across batch, hidden dim, and state dim.

    Supports both real and complex-valued tensors.

    Args:
        gates:  (B, L, ...) multiplicative coefficients
        tokens: (B, L, ...) additive inputs

    Returns:
        (B, L, ...) hidden states h[0..L-1]
    """
    a = gates
    b = tokens

    stride = 1
    while stride < a.shape[1]:
        b = torch.cat([
            b[:, :stride],
            a[:, stride:] * b[:, :-stride] + b[:, stride:]
        ], dim=1)
        a = torch.cat([
            a[:, :stride],
            a[:, stride:] * a[:, :-stride]
        ], dim=1)
        stride *= 2

    return b


# ─── Mamba-3 Block ───────────────────────────────────────────────────────────

class Mamba3Block(nn.Module):
    """
    Single Mamba-3 layer with:
      - Exponential-trapezoidal discretization
      - Complex-valued state transitions (via RoPE-like rotation)
      - MIMO (Multi-Input Multi-Output) SSM formulation
      - BCNorm (RMSNorm on B/C projections)
      - No causal Conv1d (biases in B/C replace it)
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        d_inner = config.d_inner
        n = config.d_state                  # total state dim
        n_rot = config.n_rot_state          # rotational/decay channels
        n_track = config.n_track_state      # negative-eigenvalue tracking channels
        p = config.mimo_p

        # ── Gate + value projection ──
        self.in_proj = nn.Linear(config.d_model, d_inner * 2, bias=config.bias)

        # ── SSM parameter projection ──
        # Output: dt_rank + B(n*p) + C(n*p)  (B/C cover ALL n states)
        # Bias=True replaces the removed Conv1d
        ssm_proj_size = config.dt_rank + n * p * 2
        self.x_proj = nn.Linear(d_inner, ssm_proj_size, bias=True)

        # ── dt projection ──
        self.dt_proj = nn.Linear(config.dt_rank, d_inner, bias=True)
        # Initialise dt bias for reasonable initial timescales
        with torch.no_grad():
            dt_init = torch.exp(
                torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            )
            inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
            self.dt_proj.bias.copy_(inv_dt)
            self.dt_proj.bias._no_reinit = True

        # ── A parameter (rotational/decay channels: first n_rot states) ──
        if n_rot > 0:
            if config.use_complex:
                # Rotational dynamics: A = r * exp(i*theta)
                log_A_mag = torch.log(
                    repeat(torch.arange(1, n_rot + 1, dtype=torch.float32), 'n -> d n', d=d_inner)
                )
                self.A_log_mag = nn.Parameter(log_A_mag)
                self.A_phase = nn.Parameter(torch.zeros(d_inner, n_rot))
            else:
                A = repeat(torch.arange(1, n_rot + 1, dtype=torch.float32), 'n -> d n', d=d_inner)
                self.A_log = nn.Parameter(torch.log(A))

        # ── Tracking channels (last n_track states): real signed eigenvalue ──
        # lambda = tanh(theta) in (-1, 1); init spread across the range incl. near
        # -1 so some channels begin as parity / sign-flip trackers.
        if n_track > 0:
            theta = torch.atanh(torch.linspace(-0.95, 0.95, n_track).clamp(-0.999, 0.999))
            self.track_theta = nn.Parameter(theta.unsqueeze(0).repeat(d_inner, 1))

        # ── BCNorm ──
        if config.use_bcnorm:
            self.b_norm = RMSNorm(n * p)
            self.c_norm = RMSNorm(n * p)

        # ── Skip connection & output ──
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, config.d_model, bias=config.bias)

    # ── Shared A computation (used by both scan and O(1) step) ──

    def _compute_A(self):
        """
        Return (A_real, A_imag) for the ROTATIONAL channels, shape (d_inner, n_rot),
        or (None, None) if there are no rotational channels.

        Complex: A = mag * exp(i*phase) with mag = -exp(A_log_mag) (decaying).
        Real:    A = -exp(A_log), A_imag = None.
        """
        if self.config.n_rot_state == 0:
            return None, None
        if self.config.use_complex:
            mag = -torch.exp(self.A_log_mag.float())     # (d_inner, n_rot)
            phase = self.A_phase.float()
            return mag * torch.cos(phase), mag * torch.sin(phase)
        return -torch.exp(self.A_log.float()), None

    def _track_gate(self):
        """Real signed eigenvalue for the tracking channels: (d_inner, n_track)."""
        return torch.tanh(self.track_theta.float())

    def forward(self, x):
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model)
        """
        # Gate split
        xz = self.in_proj(x)                     # (B, L, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)        # each (B, L, d_inner)

        # No Conv1d in Mamba-3 - direct SiLU activation
        x_branch = F.silu(x_branch)

        # SSM (forward; + reversed pass for the Eval Mamba's 2D board awareness)
        if not self.config.bidirectional:
            y = self.ssm(x_branch)
        else:
            # Per-token projections COMMUTE with time reversal: projecting the
            # flipped sequence yields exactly the flipped projections. So
            # project ONCE and flip the results — bit-identical to the naive
            # ssm(flip(x)) at half the projection cost (the scan itself must
            # still run once per direction).
            delta, B, C = self._ssm_proj(x_branch)
            A_real, A_imag = self._compute_A()
            y = self._selective_scan_trapezoidal(
                x_branch, delta, A_real, A_imag, B, C, self.D)
            y = y + self._selective_scan_trapezoidal(
                x_branch.flip(1), delta.flip(1), A_real, A_imag,
                B.flip(1), C.flip(1), self.D).flip(1)

        # Gated output
        z = F.silu(z)
        output = y * z
        output = self.out_proj(output)

        return output

    def forward_with_state(self, x):
        """
        Like forward() but also returns the final recurrent state (h, dBu_last)
        so a caller can continue the sequence via step(). Used to prime an MCTS
        / line-evaluation cache from a full board encoding.

        Returns: (output_seq (B,L,d_model), (h_last, dBu_last))
        """
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)
        x_branch = F.silu(x_branch)
        y, state = self.ssm(x_branch, return_state=True)
        z = F.silu(z)
        return self.out_proj(y * z), state

    def _ssm_proj(self, u):
        """Per-token SSM parameter projections: u -> (delta, B, C)."""
        config = self.config
        batch, L, _ = u.shape
        n = config.d_state
        p = config.mimo_p

        x_dbl = self.x_proj(u)  # (B, L, dt_rank + 2*n*p)
        dt_raw, B_raw, C_raw = x_dbl.split(
            [config.dt_rank, n * p, n * p], dim=-1
        )
        if config.use_bcnorm:
            B_raw = self.b_norm(B_raw)
            C_raw = self.c_norm(C_raw)
        B = B_raw.view(batch, L, n, p)      # MIMO: (B, L, n, p)
        C = C_raw.view(batch, L, n, p)
        delta = F.softplus(self.dt_proj(dt_raw))  # (B, L, d_inner)
        return delta, B, C

    def ssm(self, u, return_state=False):
        """
        Selective SSM with Mamba-3 innovations.

        Exponential-trapezoidal discretization:
            h[t] = dA[t] * h[t-1] + 0.5 * (dB[t]*u[t] + dA[t]*dB[t-1]*u[t-1])

        This 2nd-order rule sees both current AND previous inputs.
        """
        delta, B, C = self._ssm_proj(u)
        A_real, A_imag = self._compute_A()
        return self._selective_scan_trapezoidal(
            u, delta, A_real, A_imag, B, C, self.D, return_state=return_state
        )

    def _selective_scan_trapezoidal(self, u, delta, A_real, A_imag, B, C, D,
                                    return_state=False):
        """
        Exponential-trapezoidal selective scan.

        Euler (Mamba-1):
            h[t] = dA * h[t-1] + dB * u[t]

        Trapezoidal (Mamba-3):
            h[t] = dA * h[t-1] + 0.5*(dB_t*u_t + dA_t * dB_{t-1}*u_{t-1})

        The trapezoidal correction gives 2nd-order accuracy by blending
        current and previous input contributions.
        """
        batch, L, d_inner = u.shape
        config = self.config

        # Small (B,L,N) projections — never the (B,L,D,N) blowup.
        B_contracted = B.sum(dim=-1)            # (B, L, N)
        C_contracted = C.sum(dim=-1)            # (B, L, N)
        track = self._track_gate() if config.n_track_state > 0 else None  # (D, n_track)

        # ── Fused Triton path (identical recurrence, ~50x less memory traffic) ──
        # Gated: flag on + CUDA + triton importable + not the return_state
        # (inference priming) path, which stays on the reference implementation.
        if getattr(config, "use_triton_scan", False) and u.is_cuda and not return_state:
            from triton_scan import HAS_TRITON, triton_trapezoidal_scan
            if HAS_TRITON:
                y = triton_trapezoidal_scan(
                    u=u, delta=delta, B_contracted=B_contracted,
                    C_contracted=C_contracted, A_real=A_real, A_imag=A_imag,
                    track=track, n_rot=config.n_rot_state, n_state=config.d_state)
                return y + u * D

        # ── Chunk-streaming scan ──
        # Mathematically identical to a single full-length scan, but only a
        # (B, CHUNK, d_inner, d_state) working set is ever materialised per chunk.
        # During training, each chunk is additionally gradient-checkpointed so
        # backward holds ONE chunk's intermediates at a time instead of all of
        # them — this is what makes 8 GB training fit. Exact same gradients.
        use_ckpt = (config.scan_checkpoint and self.training
                    and torch.is_grad_enabled())
        h = torch.zeros((batch, d_inner, config.d_state),
                        device=u.device, dtype=torch.float32)   # carry state
        dBu_prev_last = None                                     # carry for trapezoid
        ys = []
        for s in range(0, L, SCAN_CHUNK):
            e = min(s + SCAN_CHUNK, L)
            args = (delta[:, s:e], u[:, s:e], B_contracted[:, s:e],
                    C_contracted[:, s:e], h, dBu_prev_last, A_real, A_imag, track)
            if use_ckpt:
                y_c, h, dBu_prev_last = torch.utils.checkpoint.checkpoint(
                    self._scan_chunk, *args, use_reentrant=False)
            else:
                y_c, h, dBu_prev_last = self._scan_chunk(*args)
            ys.append(y_c)

        y = torch.cat(ys, dim=1) + u * D

        if return_state:
            # (h_last, dBu_last) - the exact state step() expects to continue from.
            return y, (h, dBu_prev_last)
        return y

    def _scan_chunk(self, delta_c, u_c, B_c, C_c, h, dBu_prev, A_real, A_imag, track):
        """
        One chunk of the streaming trapezoidal scan:
            h[t] = g[t]*h[t-1] + 0.5*(dBu[t] + g[t]*dBu[t-1])
        starting from carry state `h` (and `dBu_prev`, the previous token's input
        contribution; None means zeros for the very first chunk).

        Standalone so training can recompute it chunk-by-chunk in backward
        (torch.utils.checkpoint) — see _selective_scan_trapezoidal.

        Returns: (y_c (B,c,d_inner), h_last (B,d_inner,N), dBu_last (B,d_inner,N))
        """
        batch, c, d_inner = u_c.shape

        # gates_c : (B, c, D, N)
        groups = []
        if A_real is not None:
            dec = torch.exp(einsum(delta_c, A_real, 'b l d, d n -> b l d n'))
            if A_imag is not None:
                dec = dec * torch.cos(einsum(delta_c, A_imag, 'b l d, d n -> b l d n'))
            groups.append(dec)
        if track is not None:
            groups.append(track.view(1, 1, d_inner, -1).expand(batch, c, d_inner, -1))
        gates_c = groups[0] if len(groups) == 1 else torch.cat(groups, dim=-1)

        # dBu_c : (B, c, D, N)
        dBu_c = einsum(delta_c, B_c, 'b l d, b l n -> b l d n') * u_c.unsqueeze(-1)
        if dBu_prev is None:
            dBu_prev = torch.zeros_like(dBu_c[:, 0])
        dBu_prev_c = torch.cat([dBu_prev.unsqueeze(1), dBu_c[:, :-1]], dim=1)
        tokens_c = 0.5 * dBu_c + 0.5 * gates_c * dBu_prev_c

        # Fold the carried state into the FIRST token: the recurrence
        # h[t] = g[t]*h[t-1] + x[t] with h[-1] = carry is exactly the same
        # recurrence with h[-1] = 0 and x[0] += g[0]*carry. This avoids
        # materialising a gate-cumprod tensor (big autograd memory win).
        tok0 = tokens_c[:, 0] + gates_c[:, 0] * h.to(gates_c.dtype)
        tokens_c = torch.cat([tok0.unsqueeze(1), tokens_c[:, 1:]], dim=1)

        h_c = parallel_scan(gates_c, tokens_c)                # (B, c, D, N)
        y_c = torch.einsum('bldn,bln->bld', h_c, C_c)
        return y_c, h_c[:, -1], dBu_c[:, -1]

    # ── O(1) recurrent step (Phase 3: killer feature) ──

    def step(self, x_t, state=None):
        """
        Single-token recurrent update. O(1) in sequence length.

        Reproduces EXACTLY the recurrence the parallel scan computes, so a
        cached state can be extended one move at a time without reprocessing
        the whole sequence - the foundation of fast MCTS / line evaluation.

        Args:
            x_t:   (B, d_model) - one token embedding
            state: (h, dBu_prev) from a previous step, or None to start fresh.
                   h:        (B, d_inner, n)  - compressed SSM state
                   dBu_prev: (B, d_inner, n)  - previous input contribution
                             (needed for the trapezoidal blend)
        Returns:
            out:       (B, d_model)
            new_state: (h, dBu_prev)
        """
        config = self.config
        n = config.d_state
        p = config.mimo_p
        B = x_t.shape[0]

        xz = self.in_proj(x_t)                         # (B, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)
        x_branch = F.silu(x_branch)                    # u_t : (B, d_inner)

        x_dbl = self.x_proj(x_branch)                  # (B, dt_rank + 2*n*p)
        dt_raw, B_raw, C_raw = x_dbl.split(
            [config.dt_rank, n * p, n * p], dim=-1
        )
        if config.use_bcnorm:
            B_raw = self.b_norm(B_raw)
            C_raw = self.c_norm(C_raw)

        B_mat = B_raw.view(B, n, p)
        C_mat = C_raw.view(B, n, p)
        delta = F.softplus(self.dt_proj(dt_raw))       # (B, d_inner)

        A_real, A_imag = self._compute_A()             # (d_inner, n_rot) or None

        # Real scan gate per state group, matching the parallel scan exactly.
        gate_groups = []
        if A_real is not None:
            dA_decay = torch.exp(delta.unsqueeze(-1) * A_real.unsqueeze(0))  # (B, d_inner, n_rot)
            if A_imag is not None:
                dA_decay = dA_decay * torch.cos(delta.unsqueeze(-1) * A_imag.unsqueeze(0))
            gate_groups.append(dA_decay)
        if config.n_track_state > 0:
            tg = self._track_gate()                    # (d_inner, n_track)
            gate_groups.append(tg.unsqueeze(0).expand(B, config.d_inner, -1))
        gates_t = gate_groups[0] if len(gate_groups) == 1 else torch.cat(gate_groups, dim=-1)

        B_contracted = B_mat.sum(dim=-1)               # (B, n)
        dBu_t = (delta * x_branch).unsqueeze(-1) * B_contracted.unsqueeze(1)  # (B, d_inner, n)

        if state is None:
            h = torch.zeros(B, config.d_inner, n, device=x_t.device, dtype=dBu_t.dtype)
            dBu_prev = torch.zeros_like(dBu_t)
        else:
            h, dBu_prev = state

        tokens_t = 0.5 * dBu_t + 0.5 * gates_t * dBu_prev
        h = gates_t * h + tokens_t                     # (B, d_inner, n)

        C_contracted = C_mat.sum(dim=-1)               # (B, n)
        y = torch.einsum('bdn,bn->bd', h, C_contracted)  # (B, d_inner)
        y = y + x_branch * self.D

        z = F.silu(z)
        out = self.out_proj(y * z)
        return out, (h, dBu_t)


# ─── Full Mamba-3 Model ──────────────────────────────────────────────────────

class Mamba(nn.Module):
    """
    Full Mamba-3 backbone: embedding + N layers of Mamba3Block + final norm.
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.grad_checkpoint = getattr(config, "grad_checkpoint", False)

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mixer": Mamba3Block(config),
                "norm": RMSNorm(config.d_model),
            })
            for _ in range(config.n_layer)
        ])
        self.norm_f = RMSNorm(config.d_model)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, input_embeds=None):
        """
        Args:
            input_ids:    (B, L) token indices (ignored if input_embeds given)
            input_embeds: (B, L, d_model) optional precomputed embeddings
        Returns:
            (B, L, d_model) hidden states
        """
        x = input_embeds if input_embeds is not None else self.embedding(input_ids)

        for layer in self.layers:
            mixer, norm = layer["mixer"], layer["norm"]
            if self.grad_checkpoint and self.training and x.requires_grad:
                # Recompute the (memory-heavy) mixer in the backward pass.
                x = x + torch.utils.checkpoint.checkpoint(
                    lambda inp, m=mixer, nm=norm: m(nm(inp)),
                    x, use_reentrant=False,
                )
            else:
                x = x + mixer(norm(x))

        x = self.norm_f(x)
        return x

    # ── O(1) recurrent inference ──────────────────────────────────────────

    def init_cache(self, batch_size, device):
        """Create an empty per-layer SSM cache."""
        return [None] * len(self.layers)

    def step(self, token_id=None, cache=None, input_embed=None, apply_final_norm=True):
        """
        Process ONE token, returning its hidden state and the updated cache.

        Requires a CAUSAL model (config.bidirectional=False); a bidirectional scan
        has no valid left-to-right recurrence.

        Args:
            token_id:    (B,) or (B,1) token indices (ignored if input_embed given)
            cache:       list of per-layer states (from a previous step) or None
            input_embed: (B, d_model) optional precomputed token embedding
            apply_final_norm: apply norm_f to the returned hidden state
        Returns:
            hidden:    (B, d_model)
            new_cache: updated per-layer state list
        """
        assert not self.config.bidirectional, "step() requires a causal model"
        if input_embed is not None:
            x = input_embed
        else:
            if token_id.dim() == 2:
                token_id = token_id.squeeze(1)
            x = self.embedding(token_id)               # (B, d_model)

        if cache is None:
            cache = [None] * len(self.layers)

        new_cache = []
        for i, layer in enumerate(self.layers):
            residual = x
            normed = layer["norm"](x)
            delta, new_state = layer["mixer"].step(normed, cache[i])
            x = residual + delta
            new_cache.append(new_state)

        if apply_final_norm:
            x = self.norm_f(x)
        return x, new_cache

    def prime_cache(self, input_ids):
        """
        Run the full parallel forward once to warm an SSM cache, then return
        (final_hidden, cache) so subsequent moves extend via O(1) step().

        This is the MCTS / line-evaluation entry point: encode the board ONCE
        (parallel scan over ~76 tokens), then step through candidate moves.
        """
        assert not self.config.bidirectional, "prime_cache() requires a causal model"
        cur = self.embedding(input_ids)                # (B, L, d_model)
        cache = []
        for layer in self.layers:
            normed = layer["norm"](cur)
            mixed, state = layer["mixer"].forward_with_state(normed)
            cur = cur + mixed
            cache.append(state)
        final = self.norm_f(cur[:, -1, :])
        return final, cache
