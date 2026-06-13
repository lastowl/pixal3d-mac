"""
Local Gradio web UI for Pixal3D on Apple Silicon.

Loads the pipeline, MoGe, and rembg once at startup and keeps them resident
in unified memory, so each generation is just inference time — no per-request
model loading. Drag-drop an image, get a textured GLB.

    python app_mac.py [--share] [--port 7860] [--resolution 1024|1536]

Unlike upstream app.py (HF Spaces / ZeroGPU), this runs entirely locally:
no `spaces` decorators, no remote GPU allocation, models stay warm.
"""

import os
import sys

# Environment must be set before torch is imported transitively.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

ROOT = os.path.dirname(os.path.abspath(__file__))
PIXAL_ROOT = os.path.join(ROOT, "Pixal3D")

try:
    import flex_gemm  # noqa: F401
    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
except (ImportError, RuntimeError):
    os.environ.setdefault("SPARSE_CONV_BACKEND", "none")

sys.path.insert(0, PIXAL_ROOT)
sys.path.append(os.path.join(ROOT, "stubs"))

import argparse
import time

import numpy as np
import torch
import gradio as gr
from PIL import Image

import inference as inf
import o_voxel

from generate import auto_defaults

# Loaded once in main(), referenced by generate_fn.
PIPELINE = None
MOGE = None
DEFAULTS = None

ROT = np.array(
    [[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
)


def load_models(resolution_override=None):
    global PIPELINE, MOGE, DEFAULTS
    DEFAULTS = auto_defaults()
    if resolution_override:
        DEFAULTS["resolution"] = resolution_override
    print(f"[app] {DEFAULTS['ram_gb']:.0f} GB unified memory -> "
          f"resolution {DEFAULTS['resolution']}, low_vram={DEFAULTS['low_vram']}")
    t0 = time.time()
    PIPELINE = inf.init_pipeline(inf.MODEL_PATH, low_vram=DEFAULTS["low_vram"])
    MOGE = inf.load_moge_model(device="cpu" if DEFAULTS["low_vram"] else inf.DEFAULT_DEVICE)
    print(f"[app] models resident in {time.time() - t0:.0f}s")


def generate_fn(image, seed, fov, resolution, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Upload an image first.")
    progress(0.05, desc="Preprocessing")
    out_dir = os.path.join(ROOT, "out", "ui")
    os.makedirs(out_dir, exist_ok=True)
    stamp = int(seed) if seed is not None else 42

    pre = PIPELINE.preprocess_image(image)
    tmp = os.path.join(out_dir, "_tmp_ui_pre.png")
    pre.save(tmp)
    try:
        progress(0.15, desc="Estimating camera (MoGe)")
        if fov and fov > 0:
            dist = inf.distance_from_fov(
                float(fov), torch.tensor([-1.0, 0.0, 0.0]),
                torch.tensor([0, 511]), 1.0, 512,
            )["distance_from_x"]
            cam = {"camera_angle_x": float(fov), "distance": dist, "mesh_scale": 1.0}
        else:
            cam = inf.get_camera_params_wild_moge(
                tmp, MOGE, device=("cpu" if DEFAULTS["low_vram"] else inf.DEFAULT_DEVICE),
                mesh_scale=1.0, extend_pixel=0, image_resolution=512,
            )
    finally:
        os.remove(tmp)

    progress(0.3, desc=f"Generating ({resolution}_cascade)")
    torch.manual_seed(int(seed))
    t0 = time.time()
    mesh_list, (_, _, res) = PIPELINE.run(
        pre, camera_params=cam, seed=int(seed),
        sparse_structure_sampler_params={"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0},
        shape_slat_sampler_params={"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5, "rescale_t": 3.0},
        tex_slat_sampler_params={"steps": 12, "guidance_strength": 1.0, "guidance_rescale": 0.0, "rescale_t": 3.0},
        preprocess_image=False, return_latent=True,
        pipeline_type=f"{resolution}_cascade", max_num_tokens=DEFAULTS["max_num_tokens"],
    )
    mesh = mesh_list[0]

    progress(0.85, desc="Baking textures + GLB")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices.cpu(), faces=mesh.faces.cpu(),
        attr_volume=mesh.attrs.cpu(), coords=mesh.coords.cpu(),
        attr_layout=PIPELINE.pbr_attr_layout, grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000, texture_size=4096,
        remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
    )
    glb.apply_transform(ROT)
    try:
        from backends.glb_postprocess import fix_alpha_mode
        fix_alpha_mode(glb)
    except Exception:
        pass
    out_path = os.path.join(out_dir, f"pixal3d_{stamp}_{int(time.time())}.glb")
    glb.export(out_path, extension_webp=True)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    elapsed = time.time() - t0
    return out_path, out_path, f"Done in {elapsed:.0f}s · {resolution}_cascade · seed {stamp}"


def build_ui():
    with gr.Blocks(title="Pixal3D · Apple Silicon", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Pixal3D on Apple Silicon\n"
            "Single image → textured 3D mesh, running locally on Metal. "
            "Models stay warm between generations."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="Input image", height=320)
                with gr.Accordion("Settings", open=False):
                    resolution = gr.Radio(
                        [1024, 1536], value=DEFAULTS["resolution"],
                        label="Pipeline resolution",
                        info="1536 is higher quality but much heavier (needs 96 GB+ to be comfortable)",
                    )
                    seed = gr.Number(value=42, label="Seed", precision=0)
                    fov = gr.Number(
                        value=-1, label="Manual FOV (radians)",
                        info="-1 = auto-estimate via MoGe. Try 0.2 if a result looks distorted.",
                    )
                run = gr.Button("Generate 3D", variant="primary")
                status = gr.Markdown("")
            with gr.Column(scale=1):
                model = gr.Model3D(label="Result", height=400)
                download = gr.File(label="Download GLB")
        run.click(
            generate_fn,
            inputs=[image, seed, fov, resolution],
            outputs=[model, download, status],
        )
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="Create a public Gradio link")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--resolution", type=int, default=None, choices=[1024, 1536])
    args = ap.parse_args()

    load_models(resolution_override=args.resolution)
    demo = build_ui()
    demo.queue().launch(server_port=args.port, share=args.share, show_error=True)


if __name__ == "__main__":
    main()
