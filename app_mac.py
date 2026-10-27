"""
Local Gradio web UI for Pixal3D on Apple Silicon.

Loads the pipeline, MoGe, and rembg once at startup and keeps them resident
in unified memory, so each generation is just inference time — no per-request
model loading. Drag-drop an image, get a textured GLB.

The "Multiple views" tab runs upstream's multi-view pipeline (inference_mv.py)
on several posed views of one object. Its weights are a separate 16.5 GB set,
so only one pipeline is resident at a time: generating from the other tab
unloads the current one and costs a model load first.

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
import contextlib
import gc
import io
import json
import shutil
import tempfile
import time

import numpy as np
import torch
import gradio as gr
from PIL import Image

import inference as inf
import inference_mv as inf_mv
import o_voxel

from generate import auto_defaults

# The resident pipeline ("single" or "multiview"), loaded by use_pipeline().
PIPELINE = None
MODE = None
MOGE = None
DEFAULTS = None

ROT = np.array(
    [[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
)

OUT_DIR = os.path.join(ROOT, "out", "ui")
MV_EXAMPLE = os.path.join(PIXAL_ROOT, "assets", "mv_images", "example")
# Slot labels for the example rig's four frames, in transforms.json order.
# The 90° camera sits on the object's left (its face points left in the image).
MV_SLOTS = ["Front (0°)", "Left side (90°)", "Back (180°)", "Right side (270°)"]

SAMPLER_PARAMS = dict(
    sparse_structure_sampler_params={"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0},
    shape_slat_sampler_params={"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5, "rescale_t": 3.0},
    tex_slat_sampler_params={"steps": 12, "guidance_strength": 1.0, "guidance_rescale": 0.0, "rescale_t": 3.0},
)


def use_pipeline(mode, progress=None):
    """Make the single-image or multi-view pipeline resident.

    The two are separate weight sets (~18 GB + 16.5 GB); holding both would
    push a 64 GB Mac into swap, so the other one is unloaded first.
    """
    global PIPELINE, MODE, MOGE
    if MODE == mode:
        return
    if PIPELINE is not None:
        print(f"[app] unloading {MODE} pipeline")
        PIPELINE = MOGE = MODE = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    if progress is not None:
        progress(0.02, desc=f"Loading {mode} models")
    t0 = time.time()
    if mode == "single":
        PIPELINE = inf.init_pipeline(inf.MODEL_PATH, low_vram=DEFAULTS["low_vram"])
        MOGE = inf.load_moge_model(device="cpu" if DEFAULTS["low_vram"] else inf.DEFAULT_DEVICE)
    else:
        PIPELINE = inf_mv.init_pipeline(inf_mv.MODEL_PATH, inf_mv.CONFIG_FILE, low_vram=DEFAULTS["low_vram"])
    MODE = mode
    print(f"[app] {mode} models resident in {time.time() - t0:.0f}s")


def export_glb(mesh, res, stamp):
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices.cpu(), faces=mesh.faces.cpu(),
        attr_volume=mesh.attrs.cpu(), coords=mesh.coords.cpu(),
        attr_layout=PIPELINE.pbr_attr_layout, grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000, texture_size=4096,
        remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
    )
    glb.apply_transform(ROT)
    out_path = os.path.join(OUT_DIR, f"pixal3d_{stamp}_{int(time.time())}.glb")
    glb.export(out_path, extension_webp=True)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return out_path


def generate_fn(image, seed, fov, resolution, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Upload an image first.")
    use_pipeline("single", progress)
    progress(0.05, desc="Preprocessing")
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = int(seed) if seed is not None else 42

    pre = PIPELINE.preprocess_image(image)
    tmp = os.path.join(OUT_DIR, "_tmp_ui_pre.png")
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
        pre, camera_params=cam, seed=int(seed), **SAMPLER_PARAMS,
        preprocess_image=False, return_latent=True,
        pipeline_type=f"{resolution}_cascade", max_num_tokens=DEFAULTS["max_num_tokens"],
    )
    mesh = mesh_list[0]
    del mesh_list
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()  # cached sampling/decode buffers

    progress(0.85, desc="Baking textures + GLB")
    out_path = export_glb(mesh, res, stamp)
    elapsed = time.time() - t0
    return out_path, out_path, f"Done in {elapsed:.0f}s · {resolution}_cascade · seed {stamp}"


def stage_views(work, slot_images, cams_file, view_files):
    """Write a views directory (transforms.json + images) that inference_mv reads.

    Either the four preset slots on the example rig's cameras, or an uploaded
    transforms.json whose frames are matched to uploaded images by file name.
    """
    if cams_file:
        with open(cams_file) as f:
            meta = json.load(f)
        uploads = {os.path.basename(p): p for p in view_files or []}
        missing = [fr["file_path"] for fr in meta["frames"]
                   if os.path.basename(fr["file_path"]) not in uploads]
        if missing:
            raise gr.Error(f"transforms.json names views that weren't uploaded: {', '.join(missing)}")
        for fr in meta["frames"]:
            name = os.path.basename(fr["file_path"])
            shutil.copy(uploads[name], os.path.join(work, name))
            fr["file_path"] = name
    else:
        if slot_images[0] is None:
            raise gr.Error("The front view is required.")
        with open(os.path.join(MV_EXAMPLE, "transforms.json")) as f:
            meta = json.load(f)
        frames = []
        for image, frame in zip(slot_images, meta["frames"]):
            if image is None:
                continue
            name = f"{frame['name']}.png"
            image.save(os.path.join(work, name))
            frames.append({**frame, "file_path": name})
        meta["frames"] = frames
    if len(meta["frames"]) < 2:
        raise gr.Error("Give at least two views (or use the single-image tab).")
    with open(os.path.join(work, "transforms.json"), "w") as f:
        json.dump(meta, f)


def generate_mv_fn(front, left, back, right, cams_file, view_files, seed, resolution,
                   progress=gr.Progress()):
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = int(seed) if seed is not None else 42
    work = tempfile.mkdtemp(prefix="_views_", dir=OUT_DIR)
    try:
        stage_views(work, [front, left, back, right], cams_file, view_files)
        use_pipeline("multiview", progress)
        progress(0.1, desc="Matting views")
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            views = inf_mv.load_views(work, rembg=inf_mv.make_rembg(PIPELINE))
            inf_mv.check_main_view(views)
        print(log.getvalue(), end="")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    warning = next((line.removeprefix("[Warning] ") for line in log.getvalue().splitlines()
                    if line.startswith("[Warning]")), None)

    n = len(views["view_names"])
    progress(0.2, desc=f"Generating from {n} views ({resolution}_cascade)")
    torch.manual_seed(int(seed))
    t0 = time.time()
    mesh_list, (_, _, res) = PIPELINE.run_mv(
        views, seed=int(seed), **SAMPLER_PARAMS, return_latent=True,
        pipeline_type=f"{resolution}_cascade", max_num_tokens=DEFAULTS["max_num_tokens"],
    )
    mesh = mesh_list[0]
    del mesh_list, views
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    progress(0.85, desc="Baking textures + GLB")
    out_path = export_glb(mesh, res, stamp)
    elapsed = time.time() - t0
    status = f"Done in {elapsed:.0f}s · {n} views · {resolution}_cascade · seed {stamp}"
    if warning:
        status += f"\n\n⚠️ {warning}"
    return out_path, out_path, status


def build_ui():
    with gr.Blocks(title="Pixal3D · Apple Silicon") as demo:
        gr.Markdown(
            "# Pixal3D on Apple Silicon\n"
            "Image → textured 3D mesh, running locally on Metal. "
            "Models stay warm between generations."
        )
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("Single image"):
                        image = gr.Image(type="pil", label="Input image", height=320)
                        fov = gr.Number(
                            value=-1, label="Manual FOV (radians)",
                            info="-1 = auto-estimate via MoGe. Try 0.2 if a result looks distorted.",
                        )
                        run = gr.Button("Generate 3D", variant="primary")
                    with gr.Tab("Multiple views"):
                        gr.Markdown(
                            "Shots of one object from the front and, optionally, its left, back "
                            "and right, at eye level and the same distance. Views are not "
                            "cropped: centre the object and let it fill most of the frame, as "
                            "in the example below. The first run downloads the 16.5 GB "
                            "multi-view weights; switching between tabs reloads models."
                        )
                        # image_mode=None keeps an RGBA PNG's alpha, which is used as the mask.
                        slots = []
                        for row in (MV_SLOTS[:2], MV_SLOTS[2:]):
                            with gr.Row():
                                for label in row:
                                    slots.append(gr.Image(type="pil", image_mode=None,
                                                          label=label, height=200))
                        with gr.Accordion("Custom cameras", open=False):
                            gr.Markdown(
                                "Upload a `transforms.json` (Blender/NeRF camera-to-world, Z-up; "
                                "frame 0 must be the front view) and the images it names. "
                                "This replaces the four slots above."
                            )
                            cams_file = gr.File(label="transforms.json", file_types=[".json"],
                                                type="filepath")
                            view_files = gr.File(label="View images", file_count="multiple",
                                                 file_types=["image"], type="filepath")
                        run_mv = gr.Button("Generate 3D from views", variant="primary")
                        gr.Examples(
                            examples=[[os.path.join(MV_EXAMPLE, f"view0{i}_azim{a:03d}.png")
                                       for i, a in enumerate((0, 90, 180, 270))]],
                            inputs=slots, label="Example views",
                        )
                with gr.Accordion("Settings", open=False):
                    resolution = gr.Radio(
                        [1024, 1536], value=DEFAULTS["resolution"],
                        label="Pipeline resolution",
                        info="1536 is higher quality but much heavier (needs 96 GB+ to be comfortable)",
                    )
                    seed = gr.Number(value=42, label="Seed", precision=0)
                status = gr.Markdown("")
            with gr.Column(scale=1):
                model = gr.Model3D(label="Result", height=400)
                download = gr.File(label="Download GLB")
        # One job at a time across both tabs: they share the GPU and the resident pipeline.
        run.click(
            generate_fn,
            inputs=[image, seed, fov, resolution],
            outputs=[model, download, status],
            concurrency_id="gpu",
        )
        run_mv.click(
            generate_mv_fn,
            inputs=[*slots, cams_file, view_files, seed, resolution],
            outputs=[model, download, status],
            concurrency_id="gpu",
        )
    return demo


def main():
    global DEFAULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="Create a public Gradio link")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--resolution", type=int, default=None, choices=[1024, 1536])
    args = ap.parse_args()

    DEFAULTS = auto_defaults(batch=True)  # models stay resident between requests
    if args.resolution:
        DEFAULTS["resolution"] = args.resolution
    print(f"[app] {DEFAULTS['ram_gb']:.0f} GB unified memory -> "
          f"resolution {DEFAULTS['resolution']}, low_vram={DEFAULTS['low_vram']}")
    use_pipeline("single")
    demo = build_ui()
    demo.queue().launch(server_port=args.port, share=args.share, show_error=True,
                        theme=gr.themes.Soft(), allowed_paths=[MV_EXAMPLE])


if __name__ == "__main__":
    main()
