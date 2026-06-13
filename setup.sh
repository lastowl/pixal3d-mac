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

if [ ! -d "Pixal3D" ]; then
    echo "Cloning Pixal3D ..."
    git clone --depth 1 https://github.com/TencentARC/Pixal3D.git Pixal3D
fi

# Metal stack sources (same as trellis-mac)
DEPS_DIR="deps"
mkdir -p "$DEPS_DIR"
clone_dep() {
    local url="$1" dir="$2"
    if [ ! -d "$DEPS_DIR/$dir" ]; then
        echo "Cloning $dir ..."
        git clone --depth 1 "$url" "$DEPS_DIR/$dir"
    else
        echo "  $dir already present — skipping"
    fi
}
clone_dep https://github.com/pedronaugusto/mtlbvh.git        mtlbvh
clone_dep https://github.com/pedronaugusto/mtldiffrast.git   mtldiffrast
clone_dep https://github.com/pedronaugusto/mtlmesh.git       mtlmesh
clone_dep https://github.com/pedronaugusto/mtlgemm.git       mtlgemm
clone_dep https://github.com/pedronaugusto/trellis2-apple.git trellis2-apple
clone_dep https://github.com/lastowl/mtlattn.git              mtlattn

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
$PIP torch torchvision torchaudio
$PIP -r Pixal3D/requirements.txt
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
    $PIP_NB "$DEPS_DIR/mtlattn" \
        || echo "  mtlattn install failed — large/windowed attention falls back to padded SDPA"
fi

# natten: CPU-only neighborhood attention for the NAF upsampler.
# 0.17.5 is the last release with the na2d_qk/na2d_av API NAF uses;
# it needs small compat patches to import under torch >= 2.6.
$PIP --no-build-isolation "natten==0.17.5" || echo "  natten install failed — NAF upsampler will not work"
python3 patches/natten_compat.py

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
elif [ "$RAM_GB" -ge 24 ]; then
    echo "  -> 1024 cascade, all models resident"
else
    echo "  -> 1024 cascade, low-VRAM mode (models loaded per stage)"
    echo "     Note: 16 GB machines are below Pixal3D's comfortable minimum;"
    echo "     expect heavy swapping during generation."
fi
echo "  (override anytime with --resolution / --low_vram)"
