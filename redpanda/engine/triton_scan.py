"""
Fused Triton kernel for the Mamba-3 exponential-trapezoidal selective scan.

Computes EXACTLY the same recurrence as mamba.Mamba3Block._scan_chunk /
parallel_scan:

    gates[t] = exp(delta[t]*A_real) * cos(delta[t]*A_imag)   (rot channels)
             = tanh(track_theta)                              (track channels)
    dBu[t]   = delta[t] * B[t] * u[t]
    tok[t]   = 0.5*dBu[t] + 0.5*gates[t]*dBu[t-1]
    h[t]     = gates[t]*h[t-1] + tok[t]
    y[t]     = sum_n h[t] * C[t]

but as ONE fused kernel per direction instead of dozens of elementwise ops,
so the (B, L, d_inner, d_state) intermediates never touch VRAM. The backward
kernel computes the ANALYTIC gradient of the same recurrence (a reverse-time
linear scan), recomputing per-chunk forward states from checkpoints.

Same mathematics; floating-point reassociation only (and the kernel
accumulates in fp32, which is MORE precise than the AMP fp16 path).

Gated OFF by default (MambaConfig.use_triton_scan=False). Enable only after
verify_triton.py passes on the training GPU; train.py --triton runs a quick
verification automatically before training starts.
"""

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = torch.cuda.is_available()
except ImportError:  # CPU boxes / Windows without triton: module stays importable
    triton = None
    tl = None
    HAS_TRITON = False

CHUNK = 32  # backward recompute granularity (independent of mamba.SCAN_CHUNK)


if HAS_TRITON:

    @triton.jit
    def _fwd_kernel(delta_ptr, u_ptr, Bc_ptr, Cc_ptr, Ar_ptr, Ai_ptr, Tk_ptr,
                    y_ptr, hck_ptr, dck_ptr,
                    L, D, N, NR,
                    s_db, s_dl, s_dd,
                    s_ub, s_ul, s_ud,
                    s_bb, s_bl, s_bn,
                    s_cb, s_cl, s_cn,
                    s_ad, s_an,
                    s_yb, s_yl, s_yd,
                    s_hb, s_hc, s_hd, s_hn,
                    CH: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr):
        b = tl.program_id(0)
        g = tl.program_id(1)
        offs_d = g * BD + tl.arange(0, BD)
        offs_n = tl.arange(0, BN)
        md = offs_d < D
        mn = offs_n < N
        mdn = md[:, None] & mn[None, :]
        is_rot = (offs_n < NR)[None, :] & mdn

        ap = offs_d[:, None] * s_ad + offs_n[None, :] * s_an
        Ar = tl.load(Ar_ptr + ap, mask=mdn, other=0.0).to(tl.float32)
        Ai = tl.load(Ai_ptr + ap, mask=mdn, other=0.0).to(tl.float32)
        Tk = tl.load(Tk_ptr + ap, mask=mdn, other=0.0).to(tl.float32)

        h = tl.zeros((BD, BN), dtype=tl.float32)
        dbu_prev = tl.zeros((BD, BN), dtype=tl.float32)

        nch = tl.cdiv(L, CH)
        for ci in range(0, nch):
            hp = b * s_hb + ci * s_hc + offs_d[:, None] * s_hd + offs_n[None, :] * s_hn
            tl.store(hck_ptr + hp, h, mask=mdn)
            tl.store(dck_ptr + hp, dbu_prev, mask=mdn)
            for tt in range(0, CH):
                t = ci * CH + tt
                mt = t < L
                mdt = md & mt
                dlt = tl.load(delta_ptr + b * s_db + t * s_dl + offs_d * s_dd,
                              mask=mdt, other=0.0).to(tl.float32)
                ut = tl.load(u_ptr + b * s_ub + t * s_ul + offs_d * s_ud,
                             mask=mdt, other=0.0).to(tl.float32)
                Bv = tl.load(Bc_ptr + b * s_bb + t * s_bl + offs_n * s_bn,
                             mask=mn & mt, other=0.0).to(tl.float32)
                Cv = tl.load(Cc_ptr + b * s_cb + t * s_cl + offs_n * s_cn,
                             mask=mn & mt, other=0.0).to(tl.float32)
                da = dlt[:, None]
                gate = tl.where(is_rot, tl.exp(da * Ar) * tl.cos(da * Ai), Tk)
                dbu = da * Bv[None, :] * ut[:, None]
                tok = 0.5 * dbu + 0.5 * gate * dbu_prev
                h_new = gate * h + tok
                h = tl.where(mt, h_new, h)
                dbu_prev = tl.where(mt, dbu, dbu_prev)
                yt = tl.sum(h * Cv[None, :], axis=1)
                tl.store(y_ptr + b * s_yb + t * s_yl + offs_d * s_yd, yt, mask=mdt)

    @triton.jit
    def _bwd_kernel(delta_ptr, u_ptr, Bc_ptr, Cc_ptr, Ar_ptr, Ai_ptr, Tk_ptr,
                    dy_ptr, hck_ptr, dck_ptr, hscr_ptr,
                    dd_ptr, du_ptr, dBp_ptr, dCp_ptr, dAr_ptr, dAi_ptr, dTk_ptr,
                    L, D, N, NR,
                    s_db, s_dl, s_dd,
                    s_ub, s_ul, s_ud,
                    s_bb, s_bl, s_bn,
                    s_cb, s_cl, s_cn,
                    s_ad, s_an,
                    s_yb, s_yl, s_yd,
                    s_hb, s_hc, s_hd, s_hn,
                    s_sb, s_sd, s_st, s_sn,
                    s_pg, s_pb, s_pl, s_pn,
                    CH: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr):
        b = tl.program_id(0)
        g = tl.program_id(1)
        offs_d = g * BD + tl.arange(0, BD)
        offs_n = tl.arange(0, BN)
        md = offs_d < D
        mn = offs_n < N
        mdn = md[:, None] & mn[None, :]
        is_rot = (offs_n < NR)[None, :] & mdn

        ap = offs_d[:, None] * s_ad + offs_n[None, :] * s_an
        Ar = tl.load(Ar_ptr + ap, mask=mdn, other=0.0).to(tl.float32)
        Ai = tl.load(Ai_ptr + ap, mask=mdn, other=0.0).to(tl.float32)
        Tk = tl.load(Tk_ptr + ap, mask=mdn, other=0.0).to(tl.float32)

        prev_q = tl.zeros((BD, BN), dtype=tl.float32)
        prev_gate = tl.zeros((BD, BN), dtype=tl.float32)
        dAr_acc = tl.zeros((BD, BN), dtype=tl.float32)
        dAi_acc = tl.zeros((BD, BN), dtype=tl.float32)
        dTk_acc = tl.zeros((BD, BN), dtype=tl.float32)

        nch = tl.cdiv(L, CH)
        for cj in range(0, nch):
            ci = nch - 1 - cj
            hp = b * s_hb + ci * s_hc + offs_d[:, None] * s_hd + offs_n[None, :] * s_hn
            h0 = tl.load(hck_ptr + hp, mask=mdn, other=0.0)
            d0 = tl.load(dck_ptr + hp, mask=mdn, other=0.0)

            # forward recompute within the chunk -> scratch h[tt]
            hr = h0
            dr = d0
            for tt in range(0, CH):
                t = ci * CH + tt
                mt = t < L
                mdt = md & mt
                dlt = tl.load(delta_ptr + b * s_db + t * s_dl + offs_d * s_dd,
                              mask=mdt, other=0.0).to(tl.float32)
                ut = tl.load(u_ptr + b * s_ub + t * s_ul + offs_d * s_ud,
                             mask=mdt, other=0.0).to(tl.float32)
                Bv = tl.load(Bc_ptr + b * s_bb + t * s_bl + offs_n * s_bn,
                             mask=mn & mt, other=0.0).to(tl.float32)
                da = dlt[:, None]
                gate = tl.where(is_rot, tl.exp(da * Ar) * tl.cos(da * Ai), Tk)
                dbu = da * Bv[None, :] * ut[:, None]
                tok = 0.5 * dbu + 0.5 * gate * dr
                hr_new = gate * hr + tok
                hr = tl.where(mt, hr_new, hr)
                dr = tl.where(mt, dbu, dr)
                sp = b * s_sb + offs_d[:, None] * s_sd + tt * s_st + offs_n[None, :] * s_sn
                tl.store(hscr_ptr + sp, hr, mask=mdn)

            # reverse pass within the chunk
            for tt2 in range(0, CH):
                tt = CH - 1 - tt2
                t = ci * CH + tt
                mt = t < L
                mdt = md & mt
                dlt = tl.load(delta_ptr + b * s_db + t * s_dl + offs_d * s_dd,
                              mask=mdt, other=0.0).to(tl.float32)
                ut = tl.load(u_ptr + b * s_ub + t * s_ul + offs_d * s_ud,
                             mask=mdt, other=0.0).to(tl.float32)
                Bv = tl.load(Bc_ptr + b * s_bb + t * s_bl + offs_n * s_bn,
                             mask=mn & mt, other=0.0).to(tl.float32)
                Cv = tl.load(Cc_ptr + b * s_cb + t * s_cl + offs_n * s_cn,
                             mask=mn & mt, other=0.0).to(tl.float32)
                dyt = tl.load(dy_ptr + b * s_yb + t * s_yl + offs_d * s_yd,
                              mask=mdt, other=0.0).to(tl.float32)
                da = dlt[:, None]
                e = tl.exp(da * Ar)
                cc = tl.cos(da * Ai)
                ss = tl.sin(da * Ai)
                gate = tl.where(is_rot, e * cc, Tk)
                dbu = da * Bv[None, :] * ut[:, None]

                # h[t] and h[t-1]
                sp = b * s_sb + offs_d[:, None] * s_sd + tt * s_st + offs_n[None, :] * s_sn
                h_t = tl.load(hscr_ptr + sp, mask=mdn, other=0.0)
                spm = b * s_sb + offs_d[:, None] * s_sd + (tt - 1) * s_st + offs_n[None, :] * s_sn
                hm1 = tl.load(hscr_ptr + spm, mask=mdn & (tt > 0), other=0.0)
                h_prev = tl.where(tt > 0, hm1, h0)

                # dBu[t-1] (recomputed pointwise; chunk-entry carry at tt == 0).
                # NOTE the `mt` in the masks: padding steps of the last chunk
                # (t >= L) must not read at t-1 — out-of-bounds = illegal access.
                tm = t - 1
                dl_m = tl.load(delta_ptr + b * s_db + tm * s_dl + offs_d * s_dd,
                               mask=md & (tt > 0) & mt, other=0.0).to(tl.float32)
                u_m = tl.load(u_ptr + b * s_ub + tm * s_ul + offs_d * s_ud,
                              mask=md & (tt > 0) & mt, other=0.0).to(tl.float32)
                B_m = tl.load(Bc_ptr + b * s_bb + tm * s_bl + offs_n * s_bn,
                              mask=mn & (tt > 0) & mt, other=0.0).to(tl.float32)
                dbu_m1 = tl.where(tt > 0,
                                  dl_m[:, None] * B_m[None, :] * u_m[:, None], d0)

                # backward linear scan:  q[t] = dy[t]*C[t] + gates[t+1]*q[t+1]
                q = dyt[:, None] * Cv[None, :] + prev_gate * prev_q

                # dC partial (summed over this program's d-slice)
                pcv = tl.sum(dyt[:, None] * h_t, axis=0)
                pp = g * s_pg + b * s_pb + t * s_pl + offs_n * s_pn
                tl.store(dCp_ptr + pp, pcv, mask=mn & mt)

                # dgate / ddBu
                dgate = q * h_prev + 0.5 * q * dbu_m1
                ddbu = 0.5 * q + 0.5 * prev_gate * prev_q

                dAr_acc += tl.where(is_rot & mt, dgate * da * e * cc, 0.0)
                dAi_acc += tl.where(is_rot & mt, -dgate * da * e * ss, 0.0)
                dTk_acc += tl.where((~(offs_n < NR)[None, :]) & mdn & mt, dgate, 0.0)

                # ddelta: gate chain (rot lanes) + dBu chain
                ddl = tl.sum(tl.where(is_rot, dgate * (Ar * e * cc - Ai * e * ss), 0.0), axis=1)
                sB = tl.sum(ddbu * Bv[None, :], axis=1)
                ddl += sB * ut
                dut = sB * dlt
                tl.store(dd_ptr + b * s_db + t * s_dl + offs_d * s_dd, ddl, mask=mdt)
                tl.store(du_ptr + b * s_ub + t * s_ul + offs_d * s_ud, dut, mask=mdt)

                # dB partial
                pbv = tl.sum(ddbu * (dlt * ut)[:, None], axis=0)
                tl.store(dBp_ptr + pp, pbv, mask=mn & mt)

                prev_q = q
                prev_gate = gate

        tl.atomic_add(dAr_ptr + ap, dAr_acc, mask=mdn)
        tl.atomic_add(dAi_ptr + ap, dAi_acc, mask=mdn)
        tl.atomic_add(dTk_ptr + ap, dTk_acc, mask=mdn)


class _TrapezoidalScanFn(torch.autograd.Function):
    """y = trapezoidal selective scan (see module docstring). Analytic backward."""

    @staticmethod
    def forward(ctx, delta, u, Bc, Cc, Ar_p, Ai_p, Tk_p, n_rot):
        B, L, D = u.shape
        N = Bc.shape[-1]
        delta = delta.contiguous()
        u = u.contiguous()
        Bc = Bc.contiguous()
        Cc = Cc.contiguous()
        Ar_c = Ar_p.detach().to(torch.float32).contiguous()
        Ai_c = Ai_p.detach().to(torch.float32).contiguous()
        Tk_c = Tk_p.detach().to(torch.float32).contiguous()

        nch = (L + CHUNK - 1) // CHUNK
        y = torch.empty(B, L, D, device=u.device, dtype=torch.float32)
        hck = torch.empty(B, nch, D, N, device=u.device, dtype=torch.float32)
        dck = torch.empty(B, nch, D, N, device=u.device, dtype=torch.float32)

        BD = 32
        BN = triton.next_power_of_2(max(N, 2))
        grid = (B, triton.cdiv(D, BD))
        _fwd_kernel[grid](
            delta, u, Bc, Cc, Ar_c, Ai_c, Tk_c, y, hck, dck,
            L, D, N, n_rot,
            *delta.stride(), *u.stride(), *Bc.stride(), *Cc.stride(),
            *Ar_c.stride(), *y.stride(), *hck.stride(),
            CH=CHUNK, BD=BD, BN=BN, num_warps=4,
        )
        ctx.save_for_backward(delta, u, Bc, Cc, Ar_c, Ai_c, Tk_c, hck, dck)
        ctx.n_rot = n_rot
        ctx.in_dtypes = (delta.dtype, u.dtype, Bc.dtype, Cc.dtype,
                         Ar_p.dtype, Ai_p.dtype, Tk_p.dtype)
        # Return fp32: casting to fp16 here would manufacture inf whenever
        # |y| > 65504 even though the value is finite. Downstream autocast
        # layers handle the mixed dtype; strictly more precise, same math.
        return y

    @staticmethod
    def backward(ctx, dy):
        delta, u, Bc, Cc, Ar_c, Ai_c, Tk_c, hck, dck = ctx.saved_tensors
        B, L, D = u.shape
        N = Bc.shape[-1]
        dy = dy.contiguous()

        BD = 32
        BN = triton.next_power_of_2(max(N, 2))
        G = triton.cdiv(D, BD)
        dd = torch.empty(B, L, D, device=u.device, dtype=torch.float32)
        du = torch.empty(B, L, D, device=u.device, dtype=torch.float32)
        dBp = torch.zeros(G, B, L, N, device=u.device, dtype=torch.float32)
        dCp = torch.zeros(G, B, L, N, device=u.device, dtype=torch.float32)
        dAr = torch.zeros_like(Ar_c)
        dAi = torch.zeros_like(Ai_c)
        dTk = torch.zeros_like(Tk_c)
        hscr = torch.empty(B, D, CHUNK, N, device=u.device, dtype=torch.float32)

        grid = (B, G)
        _bwd_kernel[grid](
            delta, u, Bc, Cc, Ar_c, Ai_c, Tk_c,
            dy, hck, dck, hscr,
            dd, du, dBp, dCp, dAr, dAi, dTk,
            L, D, N, ctx.n_rot,
            *delta.stride(), *u.stride(), *Bc.stride(), *Cc.stride(),
            *Ar_c.stride(), *dy.stride(), *hck.stride(),
            *hscr.stride(), *dBp.stride(),
            CH=CHUNK, BD=BD, BN=BN, num_warps=4,
        )
        td, tu, tb, tc, tar, tai, ttk = ctx.in_dtypes
        return (dd.to(td), du.to(tu), dBp.sum(0).to(tb), dCp.sum(0).to(tc),
                dAr.to(tar), dAi.to(tai), dTk.to(ttk), None)


def triton_trapezoidal_scan(delta, u, B_contracted, C_contracted,
                            A_real, A_imag, track, n_rot, n_state):
    """
    Drop-in for the chunk loop in _selective_scan_trapezoidal (y WITHOUT the
    u*D skip term; the caller adds it). Pads the per-group eigenvalue tensors
    to the full state dim so one kernel covers rot+track channels.
    """
    d_inner = u.shape[-1]
    dev = u.device
    if A_real is not None:
        Ar_p = F.pad(A_real, (0, n_state - A_real.shape[-1]))
        Ai_p = (F.pad(A_imag, (0, n_state - A_imag.shape[-1]))
                if A_imag is not None else torch.zeros(d_inner, n_state, device=dev))
    else:
        Ar_p = torch.zeros(d_inner, n_state, device=dev)
        Ai_p = torch.zeros(d_inner, n_state, device=dev)
    Tk_p = (F.pad(track, (n_rot, 0))
            if track is not None else torch.zeros(d_inner, n_state, device=dev))
    return _TrapezoidalScanFn.apply(delta, u, B_contracted, C_contracted,
                                    Ar_p, Ai_p, Tk_p, n_rot)


# ─── Verification (run on the training GPU before enabling --triton) ─────────

def verify(device="cuda", thorough=True, verbose=True):
    """
    Gate: compares the Triton path against the reference PyTorch path (forward
    AND all gradients) across shapes. Raises AssertionError on any mismatch.
    """
    assert HAS_TRITON, "triton/CUDA not available"
    import mamba as M
    from mamba import Mamba, MambaConfig

    torch.manual_seed(0)
    vocab = 500
    cfgs = [dict(d_model=32, n_layer=2, d_state=12, n_track_state=4, mimo_p=2),
            dict(d_model=48, n_layer=2, d_state=16, n_track_state=0, mimo_p=2)]
    if thorough:
        cfgs.append(dict(d_model=512, n_layer=2, d_state=64, n_track_state=16, mimo_p=4))
    lens = (1, 31, 32, 33, 70, 107, 160) if thorough else (33, 70)

    for kw in cfgs:
        for bidir in (False, True):
            # Reference uses the scan-checkpointed path: proven bit-identical to
            # the raw chunk path (grad diff 0.0) and memory-bounded, so the
            # verifier's big config fits alongside desktop use on an 8 GB card.
            ref = Mamba(MambaConfig(vocab_size=vocab, bidirectional=bidir,
                                    scan_checkpoint=True, use_triton_scan=False,
                                    **kw)).to(device)
            tri = Mamba(MambaConfig(vocab_size=vocab, bidirectional=bidir,
                                    scan_checkpoint=False, use_triton_scan=True,
                                    **kw)).to(device)
            tri.load_state_dict(ref.state_dict())
            for L in lens:
                ids = torch.randint(0, vocab, (3, L), device=device)
                ref.train(); tri.train()
                yr = ref(ids); yt = tri(ids)
                fd = (yr - yt).abs().max().item()
                assert fd < 2e-3, f"forward mismatch {fd:.2e} cfg={kw} L={L} bidir={bidir}"
                yr.pow(2).mean().backward()
                yt.pow(2).mean().backward()
                gd = 0.0
                for pr, pt in zip(ref.parameters(), tri.parameters()):
                    if pr.grad is not None and pt.grad is not None:
                        denom = pr.grad.abs().max().clamp(min=1.0)
                        gd = max(gd, ((pr.grad - pt.grad).abs().max() / denom).item())
                assert gd < 2e-3, f"grad mismatch {gd:.2e} cfg={kw} L={L} bidir={bidir}"
                if verbose:
                    print(f"  ok cfg(d={kw['d_model']},N={kw['d_state']},"
                          f"trk={kw['n_track_state']}) L={L} bidir={int(bidir)} "
                          f"| fwd {fd:.1e} grad {gd:.1e}")
                # Free this case's graphs/grads before the next one piles on.
                del yr, yt
                ref.zero_grad(set_to_none=True)
                tri.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            del ref, tri
            torch.cuda.empty_cache()
    if verbose:
        print("TRITON SCAN VERIFIED: forward + gradients match the reference path.")
    return True


if __name__ == "__main__":
    verify()
