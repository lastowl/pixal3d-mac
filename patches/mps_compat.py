"""
Apply MPS (Apple Silicon) compatibility patches to a fresh Pixal3D clone.

Pixal3D is built on the TRELLIS.2 codebase, so this follows the same recipe
as trellis-mac (https://github.com/shivampkumar/trellis-mac):
  - SDPA attention backend for the sparse windowed attention (upstream already
    has SDPA branches for full attention, but not windowed)
  - Metal/pure-PyTorch backends for the CUDA-only libraries
  - device-agnostic rewrites of hardcoded .cuda() calls on the inference path
  - skip decode-time cumesh ops (Metal cumesh is unstable on decode-sized meshes)
  - pure-Python flexible_dual_grid_to_mesh override (Metal o_voxel.convert
    segfaults on decoder output)

Run once after cloning Pixal3D:
    python patches/mps_compat.py
"""

import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIXAL_ROOT = os.path.join(ROOT, "Pixal3D")
BACKENDS_DIR = os.path.join(ROOT, "backends")
STUBS_DIR = os.path.join(ROOT, "stubs")

_warnings = []


def read_file(path):
    with open(path, "r") as f:
        return f.read()


def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)
    print(f"  Patched: {os.path.relpath(path, PIXAL_ROOT)}")


def sub(src, old, new, path, count=-1):
    """Replace old->new; record a warning if nothing matched."""
    if old not in src:
        _warnings.append(f"{os.path.relpath(path, PIXAL_ROOT)}: pattern not found:\n      {old.splitlines()[0]!r} ...")
        return src
    return src.replace(old, new, count) if count > 0 else src.replace(old, new)


BEST_DEVICE = "torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda')"


def patch_windowed_attention():
    """Add an SDPA backend to sparse windowed attention (self + cross).

    Upstream dispatches on config.ATTN with xformers/flash_attn/flash_attn_4
    branches only. We add vectorized pad -> SDPA -> unpad branches: windows
    number in the thousands, so a Python per-window loop would dominate
    runtime; instead sequences are scattered into a padded batch with a key
    padding mask.
    """
    path = os.path.join(PIXAL_ROOT, "pixal3d/modules/sparse/attention/windowed_attn.py")
    src = read_file(path)

    if "_sdpa_varlen" in src:
        print(f"  Already patched: {os.path.relpath(path, PIXAL_ROOT)}")
        return

    helpers = '''

def _mtlattn():
    """Fused Metal varlen attention (mtlattn), or None if unavailable."""
    global _MTLATTN_MOD
    try:
        return _MTLATTN_MOD
    except NameError:
        pass
    try:
        import mtlattn as _m
        _MTLATTN_MOD = _m
    except ImportError:
        _MTLATTN_MOD = None
    return _MTLATTN_MOD


def _sdpa_pad(feats, seq_lens):
    """Scatter [M, H, C] feats (concatenated variable-length sequences) into a
    padded [B, H, L, C] batch. Returns (padded, batch_idx, pos_idx) where the
    index pair maps padded positions back to rows of feats."""
    device = feats.device
    B = seq_lens.shape[0]
    L = int(seq_lens.max())
    offsets = torch.cumsum(seq_lens, dim=0) - seq_lens
    batch_idx = torch.repeat_interleave(torch.arange(B, device=device), seq_lens)
    pos_idx = torch.arange(feats.shape[0], device=device) - offsets[batch_idx]
    padded = torch.zeros(B, L, feats.shape[1], feats.shape[2], dtype=feats.dtype, device=device)
    padded[batch_idx, pos_idx] = feats
    return padded.permute(0, 2, 1, 3), batch_idx, pos_idx


def _sdpa_varlen(q, k, v, q_seq_lens, kv_seq_lens):
    """Variable-length attention. On MPS, prefers the fused Metal kernel
    (mtlattn): no padding, no materialized scores, and it sidesteps a
    torch MPS bug where masked SDPA silently corrupts outputs for large
    score matrices (empirically heads*L^2 above ~7e9). Falls back to
    padded SDPA.
    q: [Mq, H, C]; k, v: [Mkv, H, C]. Returns [Mq, H, C]."""
    if q.device.type == 'mps' and _mtlattn() is not None:
        zero = torch.zeros(1, dtype=torch.int32, device=q.device)
        cu_q = torch.cat([zero, torch.cumsum(q_seq_lens, 0).int()])
        cu_kv = cu_q if kv_seq_lens is q_seq_lens else torch.cat(
            [zero, torch.cumsum(kv_seq_lens, 0).int()])
        return _mtlattn().varlen_attention(q, k, v, cu_q, cu_kv, int(q_seq_lens.max()))
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    qp, q_bidx, q_pidx = _sdpa_pad(q, q_seq_lens)
    kp, _, _ = _sdpa_pad(k, kv_seq_lens)
    vp, _, _ = _sdpa_pad(v, kv_seq_lens)
    Lk = kp.shape[2]
    kv_mask = torch.arange(Lk, device=q.device).unsqueeze(0) < kv_seq_lens.unsqueeze(1)
    out = sdpa(qp, kp, vp, attn_mask=kv_mask[:, None, None, :])
    out = out.permute(0, 2, 1, 3)
    return out[q_bidx, q_pidx]

'''
    src = sub(
        src,
        "__all__ = [",
        helpers.rstrip("\n") + "\n\n\n__all__ = [",
        path, count=1,
    )

    # calc_window_partition: sdpa needs no precomputed args
    src = sub(
        src,
        "    return fwd_indices, bwd_indices, seq_lens, attn_func_args",
        "    elif config.ATTN == 'sdpa':\n"
        "        attn_func_args = {}\n"
        "\n"
        "    return fwd_indices, bwd_indices, seq_lens, attn_func_args",
        path, count=1,
    )

    # self-attention dispatch
    src = sub(
        src,
        "    out = out[bwd_indices]      # [T, H, C]",
        "    elif config.ATTN == 'sdpa':\n"
        "        q, k, v = qkv_feats.unbind(dim=1)                                       # [M, H, C]\n"
        "        out = _sdpa_varlen(q, k, v, seq_lens, seq_lens)                         # [M, H, C]\n"
        "\n"
        "    out = out[bwd_indices]      # [T, H, C]",
        path, count=1,
    )

    # cross-attention dispatch
    src = sub(
        src,
        "    out = out[q_bwd_indices]      # [T, H, C]",
        "    elif config.ATTN == 'sdpa':\n"
        "        k, v = kv_feats.unbind(dim=1)                                           # [M, H, C]\n"
        "        out = _sdpa_varlen(q_feats, k, v, q_seq_lens, kv_seq_lens)              # [M, H, C]\n"
        "\n"
        "    out = out[q_bwd_indices]      # [T, H, C]",
        path, count=1,
    )

    write_file(path, src)


def patch_full_attn_mtlattn():
    """Hybrid full-attention dispatch inside the upstream sdpa branch:
    above MTLATTN_MIN_SEQLEN tokens, use the fused Metal varlen kernel —
    torch's MPS sdpa materializes the [H, L, L] score matrix there (the
    1536-pipeline OOM: 54 GiB at 49K tokens) while mtlattn streams in
    constant memory. Below the threshold torch sdpa is faster; keep it."""
    path = os.path.join(PIXAL_ROOT, "pixal3d/modules/sparse/attention/full_attn.py")
    src = read_file(path)

    if "mtlattn" in src:
        print(f"  Already patched: {os.path.relpath(path, PIXAL_ROOT)}")
        return

    src = sub(
        src,
        "    elif config.ATTN == 'sdpa':\n"
        "        from torch.nn.functional import scaled_dot_product_attention as _sdpa\n"
        "        if num_all_args == 1:\n"
        "            q, k, v = qkv.unbind(dim=1)   # [T, H, C] each\n"
        "        elif num_all_args == 2:\n"
        "            k, v = kv.unbind(dim=1)        # [T_KV, H, C] each\n"
        "        # process each batch element independently (no varlen kernel needed)\n",
        "    elif config.ATTN == 'sdpa':\n"
        "        from torch.nn.functional import scaled_dot_product_attention as _sdpa\n"
        "        if num_all_args == 1:\n"
        "            q, k, v = qkv.unbind(dim=1)   # [T, H, C] each\n"
        "        elif num_all_args == 2:\n"
        "            k, v = kv.unbind(dim=1)        # [T_KV, H, C] each\n"
        "        _kv_seqlen = kv_seqlen if kv_seqlen is not None else q_seqlen\n"
        "        if (q.device.type == 'mps' and max(_kv_seqlen) > _MTLATTN_MIN_SEQLEN):\n"
        "            try:\n"
        "                import mtlattn\n"
        "            except ImportError:\n"
        "                mtlattn = None\n"
        "            if mtlattn is not None:\n"
        "                cu_q = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seqlen), dim=0)]).int().to(device)\n"
        "                cu_kv = cu_q if kv_seqlen is None else torch.cat(\n"
        "                    [torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), dim=0)]).int().to(device)\n"
        "                out = mtlattn.varlen_attention(q, k, v, cu_q, cu_kv, max(q_seqlen))\n"
        "                if s is not None:\n"
        "                    return s.replace(out)\n"
        "                else:\n"
        "                    return out.reshape(N, L, H, -1)\n"
        "        # process each batch element independently (no varlen kernel needed)\n",
        path,
    )
    src = sub(
        src,
        "from .. import config",
        "from .. import config\n"
        "import os as _os\n"
        "_MTLATTN_MIN_SEQLEN = int(_os.environ.get('MTLATTN_MIN_SEQLEN', 20000))",
        path,
    )
    write_file(path, src)


def patch_pipeline_base():
    """Pipeline.cuda() -> MPS when available."""
    path = os.path.join(PIXAL_ROOT, "pixal3d/pipelines/base.py")
    src = read_file(path)

    if "mps.is_available()" in src:
        print(f"  Already patched: {os.path.relpath(path, PIXAL_ROOT)}")
        return

    src = sub(
        src,
        '        self.to(torch.device("cuda"))',
        f"        self.to({BEST_DEVICE})",
        path,
    )
    write_file(path, src)


def patch_birefnet():
    """Device property + fix hardcoded .to(\"cuda\") in BiRefNet rembg."""
    path = os.path.join(PIXAL_ROOT, "pixal3d/pipelines/rembg/BiRefNet.py")
    src = read_file(path)

    if "def device(self)" in src:
        print(f"  Already patched: {os.path.relpath(path, PIXAL_ROOT)}")
        return

    src = sub(
        src,
        "    def to(self, device: str):\n"
        "        self.model.to(device)\n"
        "\n"
        "    def cuda(self):\n"
        "        self.model.cuda()",
        "    @property\n"
        "    def device(self):\n"
        "        return next(self.model.parameters()).device\n"
        "\n"
        "    def to(self, device):\n"
        "        self.model.to(device)\n"
        "        return self\n"
        "\n"
        "    def cuda(self):\n"
        f"        self.model.to({BEST_DEVICE})",
        path,
    )
    src = sub(
        src,
        '.unsqueeze(0).to("cuda")',
        ".unsqueeze(0).to(self.device)",
        path,
    )
    write_file(path, src)


def patch_birefnet_fallback():
    """The TencentARC/Pixal3D pipeline.json configures rembg with the gated
    briaai/RMBG-2.0. RMBG-2.0 is BiRefNet-based; the canonical ungated
    ZhengPeng7/BiRefNet loads through the identical AutoModel path, so fall
    back to it when the configured repo isn't accessible. Override with
    PIXAL3D_REMBG_MODEL to force a specific repo (e.g. after HF auth)."""
    path = os.path.join(PIXAL_ROOT, "pixal3d/pipelines/rembg/BiRefNet.py")
    src = read_file(path)

    if "PIXAL3D_REMBG_MODEL" in src:
        print(f"  Already patched: {os.path.relpath(path, PIXAL_ROOT)}")
        return

    src = sub(
        src,
        "from typing import *",
        "from typing import *\nimport os",
        path, count=1,
    )
    src = sub(
        src,
        '    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):\n'
        "        self.model = AutoModelForImageSegmentation.from_pretrained(\n"
        "            model_name, trust_remote_code=True\n"
        "        )",
        '    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):\n'
        '        model_name = os.environ.get("PIXAL3D_REMBG_MODEL", model_name)\n'
        "        try:\n"
        "            self.model = AutoModelForImageSegmentation.from_pretrained(\n"
        "                model_name, trust_remote_code=True\n"
        "            )\n"
        "        except Exception as e:\n"
        '            fallback = "ZhengPeng7/BiRefNet"\n'
        "            if model_name == fallback:\n"
        "                raise\n"
        '            print(f"[rembg] {model_name} unavailable ({type(e).__name__}); falling back to {fallback}")\n'
        "            self.model = AutoModelForImageSegmentation.from_pretrained(\n"
        "                fallback, trust_remote_code=True\n"
        "            )",
        path,
    )
    write_file(path, src)


def patch_image_feature_extractor():
    """Device-aware .cuda() in DinoV2/DinoV3 feature extractors."""
    path = os.path.join(PIXAL_ROOT, "pixal3d/modules/image_feature_extractor.py")
    src = read_file(path)

    if "mps.is_available()" in src:
        print(f"  Already patched: {os.path.relpath(path, PIXAL_ROOT)}")
        return

    # Both classes share the same to/cuda/cpu block.
    src = sub(
        src,
        "    def cuda(self):\n"
        "        self.model.cuda()",
        "    @property\n"
        "    def device(self):\n"
        "        return next(self.model.parameters()).device\n"
        "\n"
        "    def cuda(self):\n"
        f"        self.model.to({BEST_DEVICE})",
        path,
    )
    src = sub(
        src,
        "            image = torch.stack(image).cuda()",
        "            image = torch.stack(image).to(self.device)",
        path,
    )
    src = sub(
        src,
        "        image = self.transform(image).cuda()",
        "        image = self.transform(image).to(self.device)",
        path,
    )
    write_file(path, src)


def patch_image_conditioned_proj():
    """Device-aware .cuda() in DinoV3ProjFeatureExtractor and the latent
    variant used by inference.py's image conditioning."""
    path = os.path.join(PIXAL_ROOT, "pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py")
    src = read_file(path)

    if "mps.is_available()" in src:
        print(f"  Already patched: {os.path.relpath(path, PIXAL_ROOT)}")
        return

    # DinoV3ProjFeatureExtractor.cuda()
    src = sub(
        src,
        "    def cuda(self):\n"
        "        super().cuda()\n"
        "        self.model.cuda()\n"
        "        self.proj_grid.cuda()\n"
        "        if self.naf_model is not None:\n"
        "            self.naf_model.cuda()\n"
        "        return self",
        "    def cuda(self):\n"
        f"        return self.to({BEST_DEVICE})",
        path,
    )

    # Latent-variant extractor .cuda()
    src = sub(
        src,
        "    def cuda(self):\n"
        "        super().cuda()\n"
        "        self.dino_model.cuda()\n"
        "        self.proj_grid.cuda()\n"
        "        if self._vae is not None:\n"
        "            self._vae.cuda()\n"
        "        return self",
        "    def cuda(self):\n"
        f"        return self.to({BEST_DEVICE})",
        path,
    )

    # Tensor .cuda() call sites (both classes are nn.Modules)
    src = sub(
        src,
        "            image = torch.stack(image).cuda()",
        "            image = torch.stack(image).to(next(self.parameters()).device)",
        path,
    )

    # Trainer mixin setup (not on the inference.py path, but harmless)
    src = sub(
        src,
        "            self.image_cond_model.cuda()",
        f"            self.image_cond_model.to({BEST_DEVICE})",
        path,
    )

    write_file(path, src)


def patch_naf():
    """Run the NAF upsampler (torch.hub valeoai/NAF) on MPS via the
    pure-PyTorch natten shim (backends/natten_mps_shim.py) — natten itself
    hard-exits on MPS tensors and its CPU kernels cost minutes per call at
    1536-pipeline sizes (~25x slower than the shim). Falls back to CPU if
    the shim can't install. NAF stays fp32 either way (weights are fp32;
    natten CPU kernels don't support fp16)."""
    path = os.path.join(PIXAL_ROOT, "pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py")
    src = read_file(path)

    if "_ensure_naf_mps_ok" in src:
        print(f"  Already patched: {os.path.relpath(path, PIXAL_ROOT)}")
        return

    # Helper injected after the imports block (before the first def).
    src = sub(
        src,
        "def project_points_to_image_batch(",
        "def _ensure_naf_mps_ok():\n"
        "    \"\"\"Install the pure-torch natten MPS shim; True if NAF can run on MPS.\"\"\"\n"
        "    try:\n"
        "        import sys as _sys, os as _os\n"
        "        _root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..', '..'))\n"
        "        if _root not in _sys.path:\n"
        "            _sys.path.append(_root)\n"
        "        from backends.natten_mps_shim import install as _install\n"
        "        _install()\n"
        "        return True\n"
        "    except Exception as e:\n"
        "        print(f'[NAF] natten MPS shim unavailable ({e}); NAF runs on CPU')\n"
        "        return False\n"
        "\n"
        "\n"
        "def project_points_to_image_batch(",
        path, count=1,
    )

    # Load NAF on the model device; drop to CPU only if the shim is broken.
    src = sub(
        src,
        "            device = next(self.model.parameters()).device\n"
        "            self.naf_model = torch.hub.load(",
        "            device = next(self.model.parameters()).device\n"
        "            if device.type == 'mps' and not _ensure_naf_mps_ok():\n"
        "                device = torch.device('cpu')\n"
        "            self.naf_model = torch.hub.load(",
        path,
    )

    # Keep .to() consistent with shim availability.
    src = sub(
        src,
        "        if self.naf_model is not None:\n"
        "            self.naf_model.to(device)",
        "        if self.naf_model is not None:\n"
        "            _dev = torch.device(device) if not isinstance(device, torch.device) else device\n"
        "            if _dev.type != 'mps' or _ensure_naf_mps_ok():\n"
        "                self.naf_model.to(_dev)",
        path,
    )

    # fp32 in/out at the call site, device-agnostic.
    src = sub(
        src,
        "                hr_features = self.naf_model(\n"
        "                    image_for_naf, lr_features_bchw, self.naf_target_size\n"
        "                )  # [B, D, H', W']",
        "                _naf_dev = next(self.naf_model.parameters()).device\n"
        "                hr_features = self.naf_model(\n"
        "                    image_for_naf.to(_naf_dev, dtype=torch.float32),\n"
        "                    lr_features_bchw.to(_naf_dev, dtype=torch.float32),\n"
        "                    self.naf_target_size,\n"
        "                ).to(lr_features_bchw.device, dtype=lr_features_bchw.dtype)  # [B, D, H', W']",
        path,
    )
    write_file(path, src)


def patch_mesh_base():
    """Guard cumesh/flex_gemm imports; make fill_holes device-agnostic
    (mtlmesh's 2026-04-21 bounds-check fix made it safe on Metal — verified
    up to 900K faces / 133K boundary loops); skip remove_faces/simplify,
    which are unused on the inference path."""
    path = os.path.join(PIXAL_ROOT, "pixal3d/representations/mesh/base.py")
    src = read_file(path)
    orig = src

    if "except (ImportError, RuntimeError)" not in src:
        src = sub(
            src,
            "import cumesh\n"
            "from flex_gemm.ops.grid_sample import grid_sample_3d",
            "try:\n"
            "    import cumesh\n"
            "except (ImportError, RuntimeError):\n"
            "    cumesh = None\n"
            "try:\n"
            "    from flex_gemm.ops.grid_sample import grid_sample_3d\n"
            "except (ImportError, RuntimeError):\n"
            "    def grid_sample_3d(*args, **kwargs):\n"
            '        raise RuntimeError("flex_gemm grid_sample unavailable")',
            path,
        )

    # fill_holes: enabled and device-agnostic (MtlMesh.init moves tensors to
    # MPS itself; mesh.read() output is restored via the existing .to(self.device)).
    if "safe on Metal since mtlmesh bounds-check fix" not in src:
        enabled = (
            "    def fill_holes(self, max_hole_perimeter=3e-2):\n"
            "        # safe on Metal since mtlmesh bounds-check fix (2026-04-21)\n"
            "        vertices = self.vertices.clone().contiguous()\n"
            "        faces = self.faces.clone().contiguous()"
        )
        for old in (
            # previously-applied skip version
            "    def fill_holes(self, max_hole_perimeter=3e-2):\n"
            "        return  # Skip on Apple Silicon — Metal cumesh unstable on decode-sized meshes\n"
            "        vertices = self.vertices.clone().cuda().contiguous()\n"
            "        faces = self.faces.clone().cuda().contiguous()",
            # fresh upstream
            "    def fill_holes(self, max_hole_perimeter=3e-2):\n"
            "        vertices = self.vertices.clone().cuda().contiguous()\n"
            "        faces = self.faces.clone().cuda().contiguous()",
        ):
            if old in src:
                src = src.replace(old, enabled, 1)
                break
        else:
            _warnings.append(f"{os.path.relpath(path, PIXAL_ROOT)}: fill_holes pattern not found")

    for method in (
        "    def remove_faces(self, face_mask: torch.Tensor):",
        "    def simplify(self, target=1000000, verbose: bool=False, options: dict={}):",
    ):
        old = method + "\n        vertices = self.vertices.clone().cuda().contiguous()"
        if old in src:
            src = src.replace(
                old,
                method + "\n        return  # unused at inference; not validated on Metal\n"
                "        vertices = self.vertices.clone().cuda().contiguous()",
                1,
            )

    if src != orig:
        write_file(path, src)
    else:
        print(f"  Already patched: {os.path.relpath(path, PIXAL_ROOT)}")


def patch_fdg_vae():
    """Force the pure-Python flexible_dual_grid_to_mesh: the Metal o_voxel
    convert segfaults on decoder output (same failure trellis-mac hit)."""
    path = os.path.join(PIXAL_ROOT, "pixal3d/models/sc_vaes/fdg_vae.py")
    src = read_file(path)

    if "o_voxel_override_convert" in src:
        print(f"  Already patched: {os.path.relpath(path, PIXAL_ROOT)}")
        return

    src = sub(
        src,
        "from o_voxel.convert import flexible_dual_grid_to_mesh\n",
        "# Pure-Python mesh extraction — Metal/CUDA o_voxel.convert segfaults on\n"
        "# decoder output. stubs/ is appended (not prepended) so the pip-installed\n"
        "# o_voxel still wins for other submodules like o_voxel.postprocess.\n"
        "import sys, os\n"
        "_stubs = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'stubs')\n"
        "if _stubs not in sys.path:\n"
        "    sys.path.append(_stubs)\n"
        "try:\n"
        "    from o_voxel_override_convert import flexible_dual_grid_to_mesh\n"
        "except ImportError:\n"
        "    from o_voxel.convert import flexible_dual_grid_to_mesh\n",
        path,
    )
    write_file(path, src)


def patch_pipeline():
    """Guard torch.cuda.synchronize()/empty_cache() in the main pipeline."""
    path = os.path.join(PIXAL_ROOT, "pixal3d/pipelines/pixal3d_image_to_3d.py")
    src = read_file(path)

    if "mps.synchronize" in src:
        print(f"  Already patched: {os.path.relpath(path, PIXAL_ROOT)}")
        return

    src = sub(
        src,
        "        torch.cuda.synchronize()",
        "        (torch.cuda.synchronize() if torch.cuda.is_available()\n"
        "         else torch.mps.synchronize() if torch.backends.mps.is_available() else None)",
        path,
    )
    src = sub(
        src,
        "torch.cuda.empty_cache()",
        "(torch.cuda.empty_cache() if torch.cuda.is_available() else None)",
        path,
    )
    write_file(path, src)


def patch_inference():
    """Device portability for the official CLI entrypoint."""
    path = os.path.join(PIXAL_ROOT, "inference.py")
    src = read_file(path)

    if "DEFAULT_DEVICE" in src:
        print(f"  Already patched: {os.path.relpath(path, PIXAL_ROOT)}")
        return

    src = sub(
        src,
        'os.environ.setdefault("ATTN_BACKEND", "flash_attn")',
        'os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")\n'
        'os.environ.setdefault("ATTN_BACKEND", "sdpa" if not torch.cuda.is_available() else "flash_attn")\n'
        'os.environ.setdefault("SPARSE_ATTN_BACKEND", os.environ["ATTN_BACKEND"])',
        path,
    )
    src = sub(
        src,
        "from pixal3d.pipelines import Pixal3DImageTo3DPipeline",
        'DEFAULT_DEVICE = "mps" if torch.backends.mps.is_available() else "cuda"\n'
        "\n"
        "from pixal3d.pipelines import Pixal3DImageTo3DPipeline",
        path,
    )
    src = sub(src, 'device="cuda"', "device=DEFAULT_DEVICE", path)
    src = sub(
        src,
        "torch.cuda.empty_cache()",
        "(torch.cuda.empty_cache() if torch.cuda.is_available() else None)",
        path,
    )
    write_file(path, src)


def patch_inference_tmpdir():
    """Upstream bug: the preprocessed tmp image is saved into the output
    directory before that directory is created (makedirs only runs at
    export time)."""
    path = os.path.join(PIXAL_ROOT, "inference.py")
    src = read_file(path)

    if "# ensure output dir exists for tmp file" in src:
        print(f"  Already patched: {os.path.relpath(path, PIXAL_ROOT)}")
        return

    src = sub(
        src,
        "    tmp_path = os.path.join(os.path.dirname(os.path.abspath(output_path)),",
        "    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)  # ensure output dir exists for tmp file\n"
        "    tmp_path = os.path.join(os.path.dirname(os.path.abspath(output_path)),",
        path,
    )
    write_file(path, src)


def patch_inference_to_glb_cpu():
    """Hand mesh tensors to o_voxel.postprocess.to_glb on CPU. The Metal
    o_voxel/cumesh stack creates internal tensors on CPU and mixes them with
    the inputs (e.g. remesh_narrow_band_dc), so MPS inputs raise a device
    mismatch — same constraint trellis-mac documents for its bake path."""
    path = os.path.join(PIXAL_ROOT, "inference.py")
    src = read_file(path)

    if "mesh.vertices.cpu()" in src:
        print(f"  Already patched: {os.path.relpath(path, PIXAL_ROOT)}")
        return

    src = sub(
        src,
        "    glb = o_voxel.postprocess.to_glb(\n"
        "        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,\n"
        "        coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,",
        "    glb = o_voxel.postprocess.to_glb(\n"
        "        vertices=mesh.vertices.cpu(), faces=mesh.faces.cpu(), attr_volume=mesh.attrs.cpu(),\n"
        "        coords=mesh.coords.cpu(), attr_layout=pipeline.pbr_attr_layout,",
        path,
    )
    write_file(path, src)


def install_conv_backend():
    """Pure-PyTorch sparse conv fallback (SPARSE_CONV_BACKEND=none)."""
    dst = os.path.join(PIXAL_ROOT, "pixal3d/modules/sparse/conv/conv_none.py")
    if os.path.exists(dst):
        print("  Already installed: pixal3d/modules/sparse/conv/conv_none.py")
        return
    shutil.copy2(os.path.join(BACKENDS_DIR, "conv_none.py"), dst)
    print("  Installed: pixal3d/modules/sparse/conv/conv_none.py")


def install_mesh_extract():
    """Pure-Python flexible_dual_grid_to_mesh override module."""
    os.makedirs(STUBS_DIR, exist_ok=True)
    dst = os.path.join(STUBS_DIR, "o_voxel_override_convert.py")
    shutil.copy2(os.path.join(BACKENDS_DIR, "mesh_extract.py"), dst)
    print("  Installed: stubs/o_voxel_override_convert.py")


def main():
    print("Applying MPS compatibility patches to Pixal3D...")
    print(f"  Pixal3D root: {PIXAL_ROOT}")
    print()

    if not os.path.isdir(PIXAL_ROOT):
        print(f"Error: Pixal3D not found at {PIXAL_ROOT}")
        print("Run setup.sh first to clone the repository.")
        return False

    patch_windowed_attention()
    patch_full_attn_mtlattn()
    patch_pipeline_base()
    patch_birefnet()
    patch_birefnet_fallback()
    patch_image_feature_extractor()
    patch_image_conditioned_proj()
    patch_naf()
    patch_mesh_base()
    patch_fdg_vae()
    patch_pipeline()
    patch_inference()
    patch_inference_tmpdir()
    patch_inference_to_glb_cpu()
    install_conv_backend()
    install_mesh_extract()

    print()
    if _warnings:
        print("WARNINGS — some patterns did not match (upstream may have changed):")
        for w in _warnings:
            print(f"  - {w}")
        return False
    print("All patches applied.")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
