#!/usr/bin/env bash
#
# Set up Pixal3D for Apple Silicon.
# Creates a venv, installs dependencies, and applies MPS patches.
# Mirrors the approach of https://github.com/shivampkumar/trellis-mac
#

set -euo pipefail
cd "$(dirname "$0")"

echo "=== Pixal3D for Apple Silicon — Setup ==="
echo

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "Warning: This project requires Apple Silicon (M1 or later)."
fi

# Every source is pinned to a tested commit, so a later upstream change can't
# silently break the patches. An existing checkout is moved to the pin too
# (local patches applied to it are discarded).
pin_repo() {
    local url="$1" dir="$2" sha="$3"
    if [ ! -d "$dir/.git" ]; then
        echo "Cloning $dir ..."
        git init -q "$dir"
    fi
    git -C "$dir" remote remove origin 2>/dev/null || true
    git -C "$dir" remote add origin "$url"
    if [ "$(git -C "$dir" rev-parse -q --verify HEAD 2>/dev/null)" != "$sha" ]; then
        git -C "$dir" fetch -q --depth 1 origin "$sha"
        git -C "$dir" checkout -q --force "$sha"
    fi
    git -C "$dir" submodule update -q --init --recursive --depth 1
}

pin_repo https://github.com/TencentARC/Pixal3D.git Pixal3D f7cf38429b0bd264f1995f0f8743a88b1c728b94

# Metal stack sources (same as trellis-mac). The fixes this port needed
# (mtlbvh#1, mtldiffrast#3, mtlmesh#1/#2) are merged upstream.
DEPS_DIR="deps"
mkdir -p "$DEPS_DIR"
pin_repo https://github.com/pedronaugusto/mtlbvh.git         "$DEPS_DIR/mtlbvh"         6b2a0f63b476ad8225695d092902104beafd7f58
pin_repo https://github.com/pedronaugusto/mtldiffrast.git    "$DEPS_DIR/mtldiffrast"    c9499ba2e4ed6849e95ce5da7aae5bf61addb528
pin_repo https://github.com/pedronaugusto/mtlmesh.git        "$DEPS_DIR/mtlmesh"        7de3864f783407201486bf6c900a4bd76a4018a3
pin_repo https://github.com/pedronaugusto/mtlgemm.git        "$DEPS_DIR/mtlgemm"        566c133781ce4992de3f77c3a97dd6feaae9d013
pin_repo https://github.com/pedronaugusto/trellis2-apple.git "$DEPS_DIR/trellis2-apple" 17347247c91c36c8cdc1896234983e878a457bba

# torch >= 2.14 headers require C++20; the Metal deps still build with
# -std=c++17. Bump the flag in their setup.py files (the forced checkout
# above resets it, so this stays idempotent).
for f in "$DEPS_DIR"/{mtlbvh,mtldiffrast,mtlmesh,mtlgemm}/setup.py "$DEPS_DIR/trellis2-apple/o-voxel/setup.py"; do
    sed -i '' 's/c++17/c++20/g' "$f"
done

# Create venv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    if command -v uv &>/dev/null; then
        uv venv .venv --python python3.11
    else
        python3 -m venv .venv
    fi
fi
source .venv/bin/activate

if command -v uv &>/dev/null; then
    PIP="uv pip install"
else
    PIP="pip install"
fi

echo "Installing dependencies..."
# torch 2.14's MPS SDPA (Metal Performance Primitives prefill kernel) is what
# makes Pixal3D's attention fast and correct on Apple Silicon — keep >= 2.14.
$PIP torch==2.14.1 torchvision
# Pixal3D's requirements pull MoGe unpinned; MoGe 3.0 (2026-08-19) needs
# trimesh>=4.11, conflicting with Pixal3D's trimesh==4.10.1 pin. Install the
# last MoGe 2.x commit instead.
grep -v "MoGe.git" Pixal3D/requirements.txt > .venv/pixal3d-requirements.txt
$PIP -r .venv/pixal3d-requirements.txt \
    "moge @ git+https://github.com/microsoft/MoGe.git@07444410f1e33f402353b99d6ccd26bd31e469e8"
$PIP huggingface_hub safetensors trimesh scipy tqdm xatlas fast-simplification einops
# MoGe-2 accesses utils3d.pt — needs a newer utils3d than MoGe's own pin
# (upstream inconsistency); 1.7 is tested with this port.
$PIP "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@5066f998930a019695d2b2f372577109f46bcdc7"

# Metal backends. Need torch at build time -> --no-build-isolation.
if [ "${SKIP_METAL:-0}" != "1" ]; then
    export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-12.0}
    echo
    echo "Installing Metal backends (set SKIP_METAL=1 to skip)..."
    PIP_NB="$PIP --no-build-isolation"
    $PIP setuptools wheel pybind11
    $PIP_NB "$DEPS_DIR/mtlbvh"      || echo "  mtlbvh install failed — continuing without Metal BVH"
    $PIP_NB "$DEPS_DIR/mtldiffrast" || echo "  mtldiffrast install failed — continuing without Metal rasterizer"
    $PIP_NB "$DEPS_DIR/mtlmesh"     || echo "  mtlmesh (cumesh) install failed — continuing without Metal mesh ops"
    $PIP_NB "$DEPS_DIR/mtlgemm"     || echo "  mtlgemm (flex_gemm) install failed — sparse conv falls back to pure PyTorch"
    $PIP_NB "$DEPS_DIR/trellis2-apple/o-voxel" \
        || echo "  o_voxel (Apple fork) install failed — GLB texture bake unavailable"
fi

echo
echo "Applying MPS compatibility patches..."
python3 patches/mps_compat.py

echo
echo "=== Setup complete ==="
echo "Activate the environment:  source .venv/bin/activate"
echo "Generate a 3D model:       python generate.py --image photo.png --output out/model.glb"
echo "Batch (models stay warm):  python generate.py --batch images/ --output-dir out/"
echo

# Report the RAM-based auto-configuration generate.py will use
RAM_GB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
echo "Detected ${RAM_GB} GB unified memory. generate.py auto-config:"
if [ "$RAM_GB" -ge 96 ]; then
    echo "  -> full-quality 1536 cascade, all models resident"
elif [ "$RAM_GB" -ge 64 ]; then
    echo "  -> 1536 cascade for single images (models are freed before export;"
    echo "     expect some swap); batch, multi-view and the web UI default to 1024"
elif [ "$RAM_GB" -ge 24 ]; then
    echo "  -> 1024 cascade, all models resident"
else
    echo "  -> 1024 cascade, low-VRAM mode (models loaded per stage)"
    echo "     Note: 16 GB machines are below Pixal3D's comfortable minimum;"
    echo "     expect heavy swapping during generation."
fi
echo "  (override anytime with --resolution / --low_vram)"
