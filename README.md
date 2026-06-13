# Pixal3D for Apple Silicon

Run [Pixal3D](https://github.com/TencentARC/Pixal3D) — TencentARC's pixel-aligned
image-to-3D model — natively on Mac. No NVIDIA GPU required.

Pixal3D is built on the [TRELLIS.2](https://github.com/microsoft/TRELLIS.2)
codebase, so this port follows the recipe pioneered by
[trellis-mac](https://github.com/shivampkumar/trellis-mac): the CUDA-only
libraries are replaced with [Pedro Naugusto's Metal stack](https://github.com/pedronaugusto/trellis2-apple)
(`mtlgemm`/`flex_gemm`, `mtlmesh`/`cumesh`, `mtlbvh`, `mtldiffrast`, CPU `o_voxel`)
plus pure-PyTorch fallbacks, attention runs through SDPA, and the remaining
hardcoded `.cuda()` call sites are patched to be device-agnostic.

New in this port (vs. what trellis-mac already solved):

- **SDPA backend for sparse *windowed* attention** — Pixal3D's windowed
  self/cross attention only had xformers/flash-attn backends upstream.
  Implemented as a vectorized pad → masked-SDPA → unpad (no per-window
  Python loop).
- **Ungated background removal** — the official pipeline config requests the
  gated `briaai/RMBG-2.0`; this port falls back to the ungated, MIT-licensed
  `ZhengPeng7/BiRefNet` (same architecture). Set `PIXAL3D_REMBG_MODEL` to
  override after authenticating with HuggingFace.

All model weights used by default (TencentARC/Pixal3D, MoGe-2, the DINOv3
mirror, BiRefNet) are ungated — **no HuggingFace login required**.

## Requirements

- macOS on Apple Silicon, 24 GB+ unified memory recommended
- Python 3.11+
- Xcode with the Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`)
  for the Metal backends; without it, setup falls back to slower pure-PyTorch paths
- ~20 GB disk for model weights (downloaded on first run)

## Quick start

```bash
bash setup.sh
source .venv/bin/activate

# Single image (forwards to the official inference.py CLI)
python generate.py --image photo.png --output out/model.glb

# Batch: pipeline + MoGe + rembg load once and stay resident in unified
# memory — each extra image costs only its generation time
python generate.py --batch images/ --output-dir out/
```

`generate.py` supports the official flags (`--seed`, `--fov`,
`--resolution 1024|1536`, `--low_vram`) and **auto-configures by unified
memory** when flags are omitted: ≥24 GB → 1024 cascade, all models
resident; <24 GB → 1024 + low-VRAM mode; ≥96 GB → 1536 cascade.

The 1536 cascade currently needs a ~90 GB working set on MPS (measured
OOM on a 64 GB machine — the padded-SDPA attention intermediates dominate;
CUDA fits it in far less via fused attention). Shrinking that footprint is
the same future work as the fused Metal attention kernel.

## How it works

`setup.sh` clones Pixal3D, installs dependencies and the Metal backends, then
runs `patches/mps_compat.py`, which patches the checkout in place:

| Patch | Why |
|---|---|
| SDPA branches in `windowed_attn.py` | only CUDA attention backends upstream |
| `Pipeline.cuda()` → MPS | hardcoded cuda device |
| BiRefNet device handling + ungated fallback | hardcoded `.to("cuda")`, gated RMBG-2.0 |
| feature extractors device handling | hardcoded `.cuda()` tensor moves |
| skip decode-time `fill_holes`/`remove_faces`/`simplify` | Metal cumesh unstable on decode-sized meshes |
| pure-Python `flexible_dual_grid_to_mesh` override | Metal/CUDA `o_voxel.convert` segfaults on decoder output |
| guard `torch.cuda.synchronize`/`empty_cache` | no-ops without CUDA |
| `inference.py` device portability | `device="cuda"` defaults |
| decode-time `fill_holes` enabled, device-agnostic | safe on Metal since mtlmesh's 2026-04-21 bounds-check fix (trellis-mac predates it and still skips this) |
| NAF upsampler pinned to CPU (fp32) | its `natten` dependency hard-exits on MPS tensors |
| `natten` 0.17.5 compat (`patches/natten_compat.py`) | last release with NAF's `na2d_qk/av` API; needs import fixes under torch ≥ 2.6 |
| mesh tensors handed to `to_glb` on CPU | Metal cumesh remesher mixes internal CPU tensors with inputs |
| create output dir before tmp-image save | upstream ordering bug |

## Status

Verified end-to-end on an M5 Pro (64 GB), macOS 26: single image →
1024_cascade pipeline → 1.1M-vertex mesh with 4096² PBR textures in
~22 min wall-clock (~5 min of that is pipeline load + camera estimation).

The full inference pipeline now matches upstream behavior, including
decode-time hole filling. Note that exported GLBs are not strictly
watertight on any platform — the final surface comes from `to_glb`'s
narrow-band remesher and has UV seams; run a standard mesh repair
(e.g. `trimesh.repair.fill_holes`, Blender's Make Manifold) if you need
closed volumes for 3D printing.

**Fused Metal attention (`deps/mtlattn`)**: this port ships a custom
flash-attention-style variable-length kernel for MPS (simdgroup-matrix
tiling, fp32 online softmax, no padding, no materialized score matrix).
It handles all windowed sparse attention (~20× over padded SDPA) and any
full attention above ~20K tokens — where torch's own MPS SDPA either
needs a 54 GiB score tensor or, worse, **silently returns garbage for
large score matrices** (empirically ≥ ~24K tokens at 12 heads; an upstream
PyTorch MPS bug this kernel both exposed and sidesteps — see
`deps/mtlattn/tests/test_mps_sdpa_bug.py`). End-to-end: 12.2 min per
1024-cascade generation on an M5 Pro (was 22 min at first light),
GPU-bound at ~74% mean utilization. Remaining gap vs a 4090-class CUDA
setup: ~3-4×.

## Credits

- [Pixal3D](https://github.com/TencentARC/Pixal3D) by TencentARC — the model
- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) by Microsoft Research — the base codebase
- [trellis-mac](https://github.com/shivampkumar/trellis-mac) by Shivam Kumar — the porting recipe this follows
- [Pedro Naugusto](https://github.com/pedronaugusto) — the Metal backend stack
