"""
Pure-PyTorch MPS implementation of the two natten ops NAF uses:
na2d_qk and na2d_av (2D neighborhood attention, dilated, clamped windows).

natten has no Apple Silicon support (its kernels hard-exit on MPS tensors)
and its CPU kernels are slow and effectively single-threaded (~29s for a
512x512 call on an M5 Pro). This shim runs the same math on the GPU.

Approach: natten's dilated neighborhood attention is exactly standard
(dilation-1) clamped-window attention applied independently to each of the
dh*dw dilation subgrids. We fold subgrids into the batch dimension, gather
k/v neighborhoods per query with advanced indexing, and contract with
einsum — chunked over the folded batch to bound peak memory.

install() monkey-patches natten.functional so code that did
`from natten.functional import na2d_qk, na2d_av` before or after install
gets MPS dispatch for MPS tensors and the original kernels for CPU tensors.
"""

import torch

_orig_qk = None
_orig_av = None

# Per-chunk element budget for the gathered neighborhood tensor
# (B'*h*Hs*kh*Ws*kw*d). 2^28 elements ~= 1 GiB in fp32.
_CHUNK_ELEMENT_BUDGET = 2 ** 28


def _norm_pair(x):
    return (x, x) if isinstance(x, int) else tuple(x)


def _window_indices(n, k, device):
    """Clamped window start per position (natten border behavior), expanded
    to per-position absolute indices [n, k]."""
    starts = (torch.arange(n, device=device) - k // 2).clamp(0, n - k)
    return starts[:, None] + torch.arange(k, device=device)[None, :]


def _to_subgrids(x, dh, dw):
    """[B, h, H, W, d] -> [B*dh*dw, h, H/dh, W/dw, d], grouping pixels by
    (row mod dh, col mod dw) — natten's dilation semantics."""
    B, h, H, W, d = x.shape
    x = x.view(B, h, H // dh, dh, W // dw, dw, d)
    x = x.permute(0, 3, 5, 1, 2, 4, 6)
    return x.reshape(B * dh * dw, h, H // dh, W // dw, d)


def _from_subgrids(x, B, dh, dw):
    Bp, h, Hs, Ws, L = x.shape
    x = x.view(B, dh, dw, h, Hs, Ws, L).permute(0, 3, 4, 1, 5, 2, 6)
    return x.reshape(B, h, Hs * dh, Ws * dw, L)


def _gather_neighborhoods(t, ri, cj):
    """t: [B', h, Hs, Ws, d] -> [B', h, Hs, Ws, kh, kw, d]"""
    # Index dims 2 and 3 with broadcast [Hs,kh,1,1] x [1,1,Ws,kw] -> result
    # [B', h, Hs, kh, Ws, kw, d]; then move kh next to kw.
    g = t[:, :, ri[:, :, None, None], cj[None, None, :, :], :]
    return g.permute(0, 1, 2, 4, 3, 5, 6)


def _chunk_size(per_item_elements):
    return max(1, _CHUNK_ELEMENT_BUDGET // max(per_item_elements, 1))


def _qk_dense(q, k, kh, kw):
    """Dilation-1 clamped-window QK on [B', h, Hs, Ws, d]."""
    Bp, h, Hs, Ws, d = q.shape
    ri = _window_indices(Hs, kh, q.device)
    cj = _window_indices(Ws, kw, q.device)
    out = torch.empty(Bp, h, Hs, Ws, kh * kw, dtype=q.dtype, device=q.device)
    step = _chunk_size(h * Hs * Ws * kh * kw * d)
    for s in range(0, Bp, step):
        kn = _gather_neighborhoods(k[s:s + step], ri, cj)  # [b,h,Hs,Ws,kh,kw,d]
        out[s:s + step] = torch.einsum("bhxyd,bhxyijd->bhxyij", q[s:s + step], kn).reshape(
            kn.shape[0], h, Hs, Ws, kh * kw
        )
    return out


def _av_dense(attn, v, kh, kw):
    """Dilation-1 clamped-window AV on attn [B', h, Hs, Ws, kh*kw], v [B', h, Hs, Ws, d]."""
    Bp, h, Hs, Ws, _ = attn.shape
    d = v.shape[-1]
    ri = _window_indices(Hs, kh, v.device)
    cj = _window_indices(Ws, kw, v.device)
    out = torch.empty(Bp, h, Hs, Ws, d, dtype=v.dtype, device=v.device)
    step = _chunk_size(h * Hs * Ws * kh * kw * d)
    a = attn.view(Bp, h, Hs, Ws, kh, kw)
    for s in range(0, Bp, step):
        vn = _gather_neighborhoods(v[s:s + step], ri, cj)
        out[s:s + step] = torch.einsum("bhxyij,bhxyijd->bhxyd", a[s:s + step], vn)
    return out


def _supported(t, kernel_size, dilation, extra=None):
    if t.device.type != "mps":
        return False
    kh, kw = _norm_pair(kernel_size)
    dh, dw = _norm_pair(dilation)
    H, W = t.shape[2], t.shape[3]
    if H % dh or W % dw:
        return False
    return H // dh >= kh and W // dw >= kw


def na2d_qk(q, k, kernel_size, dilation=1, *args, **kwargs):
    if not _supported(q, kernel_size, dilation) or kwargs.get("rpb") is not None:
        return _orig_qk(q, k, kernel_size=kernel_size, dilation=dilation, *args, **kwargs)
    kh, kw = _norm_pair(kernel_size)
    dh, dw = _norm_pair(dilation)
    B = q.shape[0]
    attn = _qk_dense(_to_subgrids(q, dh, dw), _to_subgrids(k, dh, dw), kh, kw)
    return _from_subgrids(attn, B, dh, dw)


def na2d_av(attn, v, kernel_size, dilation=1, *args, **kwargs):
    if not _supported(v, kernel_size, dilation):
        return _orig_av(attn, v, kernel_size=kernel_size, dilation=dilation, *args, **kwargs)
    kh, kw = _norm_pair(kernel_size)
    dh, dw = _norm_pair(dilation)
    B = v.shape[0]
    a_sub = _to_subgrids(attn, dh, dw)
    out = _av_dense(a_sub, _to_subgrids(v, dh, dw), kh, kw)
    return _from_subgrids(out, B, dh, dw)


def install():
    """Patch natten.functional in place. Idempotent."""
    global _orig_qk, _orig_av
    import natten.functional as nf

    if getattr(nf, "_mps_shim_installed", False):
        return
    _orig_qk = nf.na2d_qk
    _orig_av = nf.na2d_av
    nf.na2d_qk = na2d_qk
    nf.na2d_av = na2d_av
    nf._mps_shim_installed = True
