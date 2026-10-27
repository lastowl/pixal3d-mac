# Pixal3D for Apple Silicon

Run [Pixal3D](https://github.com/TencentARC/Pixal3D) — TencentARC's pixel-aligned
image-to-3D model — natively on Mac. No NVIDIA GPU required.

Pixal3D is built on the [TRELLIS.2](https://github.com/microsoft/TRELLIS.2)
codebase, so this port follows the recipe pioneered by
[trellis-mac](https://github.com/shivampkumar/trellis-mac): the CUDA-only
libraries are replaced with [Pedro Naugusto's Metal stack](https://github.com/pedronaugusto/trellis2-apple)
(`mtlgemm`/`flex_gemm`, `mtlmesh`/`cumesh`, `mtlbvh`, `mtldiffrast`, `o_voxel`)
plus pure-PyTorch fallbacks, attention runs through PyTorch's MPS SDPA, and the
remaining hardcoded `.cuda()` call sites are patched to be device-agnostic.

New in this port (vs. what trellis-mac already solved):

- **NAF upsampler on the GPU** — Pixal3D's feature upsampler depends on
  `natten`, which hard-exits on MPS tensors and no longer builds against
  torch ≥ 2.14. A pure-PyTorch implementation of the two ops it uses
  (`backends/natten_mps_shim.py`) stands in for the package and runs them on
  MPS.
- **Ungated background removal** — the official pipeline config requests the
  gated `briaai/RMBG-2.0`; this port falls back to the ungated, MIT-licensed
  `ZhengPeng7/BiRefNet` (same architecture). Set `PIXAL3D_REMBG_MODEL` to
  override after authenticating with HuggingFace.
- **Decode-time hole filling on Metal**, which trellis-mac still skips.

All model weights used by default (TencentARC/Pixal3D, MoGe-2, the DINOv3
mirror, BiRefNet) are ungated — **no HuggingFace login required**.

## Requirements

- macOS on Apple Silicon, 24 GB+ unified memory recommended
- **macOS 26.2+** for fast attention (PyTorch's MPS SDPA uses Metal 4
  `matmul2d` there; older macOS works, slower)
- Python 3.11+
- Xcode with the Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`)
  for the Metal backends; without it, setup falls back to slower pure-PyTorch paths
- ~20 GB disk for model weights (downloaded on first run), plus 16.5 GB for
  the multi-view weights if you use `--views_dir`

## Quick start

```bash
bash setup.sh
source .venv/bin/activate

# Single image (forwards to the official inference.py CLI)
python generate.py --image photo.png --output out/model.glb

# Batch: pipeline + MoGe + rembg load once and stay resident in unified
# memory — each extra image costs only its generation time
python generate.py --batch images/ --output-dir out/

# Multi-view: several posed views of one object (forwards to inference_mv.py)
python generate.py --views_dir views/ --output out/model.glb
```

### Multiple views

Upstream Pixal3D can condition on several views of the same object, using a
separate set of multi-view weights (16.5 GB more, downloaded on first use).
The views directory holds the images plus a `transforms.json` with each
view's camera; `Pixal3D/assets/mv_images/example` is a working input. For the
usual four shots at 0°/90°/180°/270° around the object at eye level, copy that
example's `transforms.json` and point its `file_path`s at your images. The
first view must be the front. Views without an alpha channel are
background-matted automatically; they are not cropped, so frame them the way
the cameras describe. `--num_views N` uses only the first N views. See the
[upstream README](https://github.com/TencentARC/Pixal3D#multi-view-inference)
for the camera convention.

`generate.py` supports the official flags (`--seed`, `--fov`,
`--resolution 1024|1536`, `--low_vram`) and **auto-configures by unified
memory** when flags are omitted: ≥64 GB → 1536 cascade (batch mode, the web
UI and multi-view need ≥96 GB for 1536); ≥24 GB → 1024 cascade, all models
resident; <24 GB → 1024 + low-VRAM mode.

`setup.sh` pins every source it fetches — Pixal3D, the Metal stack, MoGe and
`torch==2.14.1` — to tested versions, so an upstream change can't silently
break the patches. Re-running it moves an existing checkout to the pins.

## How it works

`setup.sh` fetches Pixal3D and the Metal backends at their pinned commits,
installs dependencies, then runs `patches/mps_compat.py`, which patches the
checkout in place:

| Patch | Why |
|---|---|
| `Pipeline.cuda()` → MPS | hardcoded cuda device |
| BiRefNet device handling + ungated fallback | hardcoded `.to("cuda")`, gated RMBG-2.0 |
| feature extractors device handling | hardcoded `.cuda()` tensor moves |
| NAF upsampler on MPS via the natten stand-in | `natten` hard-exits on MPS and doesn't build on torch ≥ 2.14 |
| decode-time `fill_holes` enabled, device-agnostic; `remove_faces`/`simplify` skipped | Metal cumesh runs hole filling (M1/M2 via mtlmesh's float-atomic fallback); the other two are unused at inference |
| pure-Python `flexible_dual_grid_to_mesh` override | Metal/CUDA `o_voxel.convert` segfaults on decoder output |
| `torch.cuda.synchronize` → MPS | no CUDA |
| `inference.py` device portability | `device="cuda"` defaults |
| mesh tensors handed to `to_glb` on CPU | Metal cumesh remesher mixes internal CPU tensors with inputs |
| `inference_mv.py`: same device / attention / export fixes as `inference.py` | multi-view CLI has the same CUDA assumptions |
| `torch.cuda.empty_cache()` → `torch.mps.empty_cache()`; models freed before export | without it the MPS allocator keeps every stage's buffers and the 1536 export runs a 64 GB Mac out of memory |
| create output dir before tmp-image save | upstream ordering bug |

Two dependency fixes happen in `setup.sh` rather than as patches: the Metal
backends are built with `-std=c++20` (torch 2.14's headers require it; the
upstream `setup.py` files still say C++17), and MoGe is pinned to its last 2.x
commit (Pixal3D installs MoGe unpinned, and MoGe 3.0 conflicts with Pixal3D's
`trimesh` pin).

Attention needs no patch: Pixal3D's flow models use full attention throughout,
and upstream's `sdpa` backend (one SDPA call per sequence) runs on torch
2.14's MPS SDPA, which has a fused `matmul2d` prefill kernel — no padding and
no materialized score matrix. (Earlier versions of this port shipped a custom
Metal kernel, [mtlattn](https://github.com/lastowl/mtlattn), for this; on torch
2.14 native SDPA is ~2.4× faster than it.)

## Status

Validated on an M5 Pro (64 GB, macOS 26.5, torch 2.14.1) with the shoe test
image, and upstream's four-view example for multi-view:

| | time | notes |
|---|---|---|
| 1024 cascade | ~7.5 min | ~1M-face GLB, 4096² PBR textures, opaque |
| 1536 cascade | ~13.5 min | export is the memory peak (10–13 GB of swap on 64 GB) |
| multi-view, 4 views, 1024 | ~11.5 min | 1536 multi-view swap-thrashes on 64 GB, so auto-config only picks it at ≥96 GB (untested there) |

That's 1.2× (1024) and 1.8× (1536) faster than this port's previous release
(torch 2.12 + mtlattn), mostly from torch 2.14's attention kernel.
Generations are not bit-identical to that release or to CUDA (different
kernels → different samples), but mesh quality is on par: across four test
seeds the main shell held 99–99.9% of the faces with 0–77 open edges, and
one seed produced a thin detached inner sheet just under the surface — the
same kind of sample-dependent artifact the previous release showed.

Exported GLBs are not strictly watertight on any platform — the final surface
comes from `to_glb`'s narrow-band remesher and has UV seams; run a standard
mesh repair (e.g. `trimesh.repair.fill_holes`, Blender's Make Manifold) if you
need closed volumes for 3D printing.

An upstream macOS port is in review as
[TencentARC/Pixal3D#35](https://github.com/TencentARC/Pixal3D/pull/35); once
something like it merges, this repo becomes a thin convenience layer at most.

## Credits

- [Pixal3D](https://github.com/TencentARC/Pixal3D) by TencentARC — the model
- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) by Microsoft Research — the base codebase
- [trellis-mac](https://github.com/shivampkumar/trellis-mac) by Shivam Kumar — the porting recipe this follows
- [Pedro Naugusto](https://github.com/pedronaugusto) — the Metal backend stack
