"""
Lightweight in-process timing for the sparse DiT hot path (no root needed,
unlike py-spy on macOS). Wraps key ops with MPS-synchronized timers and
prints a running breakdown every 45s. Enable via PROFILE_HOOKS=1.

Synchronizing each call serializes the GPU pipeline (so absolute times are
inflated), but the *relative* shares are what we want for attribution.
"""

import threading
import time

import torch

TIMINGS = {}
COUNTS = {}
_lock = threading.Lock()


def _sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _timed(name, fn):
    def wrapper(*a, **k):
        _sync(); t0 = time.time()
        out = fn(*a, **k)
        _sync(); dt = time.time() - t0
        with _lock:
            TIMINGS[name] = TIMINGS.get(name, 0.0) + dt
            COUNTS[name] = COUNTS.get(name, 0) + 1
        return out
    return wrapper


def _printer():
    while True:
        time.sleep(45)
        with _lock:
            tot = sum(TIMINGS.values()) or 1e-9
            rows = sorted(TIMINGS.items(), key=lambda x: -x[1])
        print("\n[profile] cumulative GPU-synced time by op:", flush=True)
        for n, t in rows:
            print(f"[profile]   {n:28s} {t:8.1f}s ({100*t/tot:4.1f}%)  x{COUNTS[n]}", flush=True)


def install():
    # Attention: the fused Metal kernel
    try:
        import mtlattn
        mtlattn.varlen_attention = _timed("attention: mtlattn", mtlattn.varlen_attention)
    except Exception as e:
        print("[profile] mtlattn hook failed:", e)

    # Sparse attention dispatch (incl. serialization/window-partition overhead)
    try:
        from pixal3d.modules.sparse.attention import windowed_attn as wa
        wa.sparse_windowed_scaled_dot_product_self_attention = _timed(
            "attn+serialize: windowed", wa.sparse_windowed_scaled_dot_product_self_attention)
    except Exception as e:
        print("[profile] windowed hook failed:", e)

    # Sparse 3D convolution
    try:
        from pixal3d.modules.sparse.conv.conv import SparseConv3d
        SparseConv3d.forward = _timed("sparse_conv3d", SparseConv3d.forward)
    except Exception as e:
        print("[profile] conv hook failed:", e)

    # segment_reduce (the CPU-fallback suspect)
    try:
        torch.segment_reduce = _timed("segment_reduce", torch.segment_reduce)
    except Exception as e:
        print("[profile] segment_reduce hook failed:", e)

    t = threading.Thread(target=_printer, daemon=True)
    t.start()
    print("[profile] hooks installed", flush=True)
