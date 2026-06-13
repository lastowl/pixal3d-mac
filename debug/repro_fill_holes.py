"""
Reproduce the cumesh (mtlmesh) instability on decode-sized meshes.

Runs the exact fill_holes call chain from pixal3d/representations/mesh/base.py
step by step with flushed prints, so a native crash identifies the failing
step. Run via the driver (debug/find_threshold.py) or directly:

    python debug/repro_fill_holes.py <n_faces> [glb_path]
"""

import faulthandler
import sys

faulthandler.enable()

import numpy as np
import torch
import trimesh

N_FACES = int(sys.argv[1])
GLB = sys.argv[2] if len(sys.argv) > 2 else "out/shoe.glb"


def log(msg):
    print(msg, flush=True)


# --- Build a test mesh of the requested size with holes ---
scene = trimesh.load(GLB)
mesh = (
    trimesh.util.concatenate(list(scene.geometry.values()))
    if isinstance(scene, trimesh.Scene)
    else scene
)
if len(mesh.faces) > N_FACES:
    import fast_simplification

    verts, faces = fast_simplification.simplify(
        mesh.vertices.astype(np.float32),
        mesh.faces.astype(np.int64),
        1.0 - N_FACES / len(mesh.faces),
    )
else:
    verts, faces = mesh.vertices.astype(np.float32), mesh.faces

# Punch ~100 holes by deleting random faces (deterministic)
rng = np.random.RandomState(0)
keep = np.ones(len(faces), dtype=bool)
keep[rng.choice(len(faces), size=min(100, len(faces) // 100), replace=False)] = False
faces = faces[keep]

log(f"mesh: {len(verts)} verts, {len(faces)} faces")

vertices = torch.from_numpy(np.ascontiguousarray(verts)).float().contiguous()
faces_t = torch.from_numpy(np.ascontiguousarray(faces)).int().contiguous()

# --- The fill_holes chain from pixal3d mesh/base.py ---
import cumesh

m = cumesh.CuMesh()
log("step: init")
m.init(vertices, faces_t)
log("step: get_edges")
m.get_edges()
log("step: get_boundary_info")
m.get_boundary_info()
log(f"  num_boundaries={m.num_boundaries}")
if m.num_boundaries == 0:
    log("OK (no boundaries)")
    sys.exit(0)
log("step: get_vertex_edge_adjacency")
m.get_vertex_edge_adjacency()
log("step: get_vertex_boundary_adjacency")
m.get_vertex_boundary_adjacency()
log("step: get_manifold_boundary_adjacency")
m.get_manifold_boundary_adjacency()
log("step: read_manifold_boundary_adjacency")
m.read_manifold_boundary_adjacency()
log("step: get_boundary_connected_components")
m.get_boundary_connected_components()
log("step: get_boundary_loops")
m.get_boundary_loops()
log(f"  num_boundary_loops={m.num_boundary_loops}")
log("step: fill_holes")
m.fill_holes(max_hole_perimeter=3e-2)
log("step: read")
nv, nf = m.read()
log(f"OK: result {nv.shape[0]} verts, {nf.shape[0]} faces")
