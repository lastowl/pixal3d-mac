"""
Generate 3D meshes from images using Pixal3D on Apple Silicon.

Single image (forwards to the official Pixal3D/inference.py CLI):

    python generate.py --image photo.png --output out/model.glb --resolution 1024

Batch mode (unified-memory optimization: pipeline, MoGe and rembg are loaded
once and stay resident — saves ~6 min of model loading per image vs the
upstream flow, which reloads everything per run to protect a small VRAM pool):

    python generate.py --batch images_dir/ --output-dir out/ --resolution 1024
"""

import os
import sys

# MPS fallback must be set before torch is imported (including transitively
# via flex_gemm) or unsupported ops crash instead of falling back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")

ROOT = os.path.dirname(os.path.abspath(__file__))
PIXAL_ROOT = os.path.join(ROOT, "Pixal3D")

try:
    import flex_gemm  # noqa: F401
    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
except (ImportError, RuntimeError):
    # No Metal stack (SKIP_METAL=1) or metallib needs newer macOS.
    os.environ.setdefault("SPARSE_CONV_BACKEND", "none")

sys.path.insert(0, PIXAL_ROOT)
sys.path.append(os.path.join(ROOT, "stubs"))

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def unified_memory_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError):
        return 0.0


def auto_defaults():
    """Pick pipeline defaults from the machine's unified memory.

    Pixal3D standard mode keeps ~18 GB of models resident. Tiers
    (overridable via explicit CLI flags):
      >= 96 GB: full-quality 1536 cascade, everything resident
      >= 24 GB: 1024 cascade, everything resident
      <  24 GB: 1024 cascade, low-VRAM mode (models loaded per stage),
                reduced sparse-token budget

    The 1536 cascade's working set is ~90 GB on MPS (measured OOM on a
    64 GB M5 Pro: 77 GiB allocated + a 13 GiB request against an 88 GiB
    cap) — the padded-SDPA attention intermediates dominate. Until that's
    optimized, 1536 is only defaulted on 96 GB+ machines.
    """
    gb = unified_memory_gb()
    if gb >= 96:
        return {"resolution": 1536, "low_vram": False, "max_num_tokens": 49152, "ram_gb": gb}
    if gb >= 24:
        return {"resolution": 1024, "low_vram": False, "max_num_tokens": 49152, "ram_gb": gb}
    return {"resolution": 1024, "low_vram": True, "max_num_tokens": 32768, "ram_gb": gb}


def run_batch():
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Pixal3D batch generation (resident models)")
    parser.add_argument("--batch", type=str, required=True, help="Directory of input images")
    parser.add_argument("--output-dir", type=str, default="out", help="Output directory for GLBs")
    auto = auto_defaults()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=auto["resolution"], choices=[1024, 1536],
                        help=f"default auto-selected for {auto['ram_gb']:.0f} GB unified memory: {auto['resolution']}")
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--max-num-tokens", type=int, default=auto["max_num_tokens"],
                        help="Sparse token budget; raise for more detail (unified memory permits)")
    parser.add_argument("--fov", type=float, default=-1.0, help="Manual FOV in radians; default auto via MoGe")
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()

    import math
    import numpy as np
    import torch
    from PIL import Image

    import inference as inf
    import o_voxel

    images = sorted(
        os.path.join(args.batch, f)
        for f in os.listdir(args.batch)
        if f.lower().endswith(IMAGE_EXTS)
    )
    if not images:
        print(f"No images ({'/'.join(IMAGE_EXTS)}) found in {args.batch}")
        sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Batch: {len(images)} image(s) -> {args.output_dir}/")

    # --- Load everything once; on >=24 GB machines it all stays resident ---
    t0 = time.time()
    pipeline = inf.init_pipeline(args.model_path or inf.MODEL_PATH, low_vram=auto["low_vram"])
    moge_model = inf.load_moge_model(device="cpu" if auto["low_vram"] else inf.DEFAULT_DEVICE)
    print(f"Models loaded in {time.time() - t0:.0f}s (resident for the whole batch)")

    rot = np.array(
        [[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
    )

    failures = []
    for i, img_path in enumerate(images, 1):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.output_dir, f"{stem}.glb")
        print(f"\n[{i}/{len(images)}] {img_path} -> {out_path}")
        t1 = time.time()
        try:
            img = Image.open(img_path)
            pre = pipeline.preprocess_image(img)

            tmp_path = os.path.join(args.output_dir, f"_tmp_pre_{stem}.png")
            pre.save(tmp_path)
            try:
                if args.fov > 0:
                    camera_angle_x = float(args.fov)
                    distance = inf.distance_from_fov(
                        camera_angle_x, torch.tensor([-1.0, 0.0, 0.0]),
                        torch.tensor([0, 511]), 1.0, 512,
                    )["distance_from_x"]
                    cam = {"camera_angle_x": camera_angle_x, "distance": distance, "mesh_scale": 1.0}
                else:
                    cam = inf.get_camera_params_wild_moge(
                        tmp_path, moge_model, device=inf.DEFAULT_DEVICE,
                        mesh_scale=1.0, extend_pixel=0, image_resolution=512,
                    )
            finally:
                os.remove(tmp_path)
            print(f"  camera_angle_x={cam['camera_angle_x']:.4f}, distance={cam['distance']:.4f}")

            torch.manual_seed(args.seed)
            # Sampler defaults mirror inference.run_inference
            mesh_list, (_, _, res) = pipeline.run(
                pre,
                camera_params=cam,
                seed=args.seed,
                sparse_structure_sampler_params={
                    "steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0,
                },
                shape_slat_sampler_params={
                    "steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5, "rescale_t": 3.0,
                },
                tex_slat_sampler_params={
                    "steps": 12, "guidance_strength": 1.0, "guidance_rescale": 0.0, "rescale_t": 3.0,
                },
                preprocess_image=False,
                return_latent=True,
                pipeline_type=f"{args.resolution}_cascade",
                max_num_tokens=args.max_num_tokens,
            )
            mesh = mesh_list[0]

            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices.cpu(), faces=mesh.faces.cpu(),
                attr_volume=mesh.attrs.cpu(), coords=mesh.coords.cpu(),
                attr_layout=pipeline.pbr_attr_layout,
                grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=1000000, texture_size=args.texture_size,
                remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
            )
            glb.apply_transform(rot)
            try:
                from backends.glb_postprocess import fix_alpha_mode
                fix_alpha_mode(glb)
            except Exception:
                pass
            glb.export(out_path, extension_webp=True)
            print(f"  Saved {out_path} in {time.time() - t1:.0f}s")
        except Exception as e:
            print(f"  FAILED ({type(e).__name__}): {e}")
            failures.append(img_path)
        finally:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    print(f"\nBatch done: {len(images) - len(failures)}/{len(images)} succeeded")
    for f in failures:
        print(f"  failed: {f}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    if "--batch" in sys.argv:
        run_batch()
    else:
        # Inject RAM-based defaults unless the user chose explicitly.
        auto = auto_defaults()
        if "--resolution" not in sys.argv:
            sys.argv += ["--resolution", str(auto["resolution"])]
        if auto["low_vram"] and "--low_vram" not in sys.argv:
            sys.argv.append("--low_vram")
        print(f"[auto-config] {auto['ram_gb']:.0f} GB unified memory -> "
              f"resolution {auto['resolution']}, low_vram={auto['low_vram']}")
        import runpy
        runpy.run_path(os.path.join(PIXAL_ROOT, "inference.py"), run_name="__main__")
