"""
Force opaque GLB materials back to alphaMode='OPAQUE' when they're only
"transparent" due to numerical drift.

o_voxel.postprocess.to_glb sets alphaMode='BLEND' if *any* baked atlas texel
has alpha < 250. bf16 flow models on Apple Silicon leave a handful of stray
sub-1.0 texels (drift / empty atlas cells), so fully opaque models ship as
BLEND and render see-through in glTF viewers (donmccurdy, three.js, Blender).

This applies the same percentile-based detection as the upstream fix
(pedronaugusto/trellis2-apple#1) at our export layer, so it works regardless
of which o_voxel build is installed: if the 1st-percentile alpha is solidly
opaque, the few low texels are drift, not real transparency -> OPAQUE.
"""

import numpy as np


def fix_alpha_mode(glb, percentile=1.0, threshold=250):
    """Mutate a trimesh Scene/Trimesh in place; return it. Sets each PBR
    material to OPAQUE when its baseColorTexture alpha is opaque at the given
    percentile (genuinely transparent models are left untouched)."""
    try:
        import trimesh
    except ImportError:
        return glb

    geoms = glb.geometry.values() if isinstance(glb, trimesh.Scene) else [glb]
    for geom in geoms:
        mat = getattr(getattr(geom, "visual", None), "material", None)
        if mat is None or getattr(mat, "alphaMode", None) != "BLEND":
            continue
        tex = getattr(mat, "baseColorTexture", None)
        if tex is None or tex.mode not in ("RGBA", "LA"):
            # No usable alpha channel -> the BLEND flag is spurious.
            mat.alphaMode = "OPAQUE"
            continue
        alpha = np.asarray(tex)[..., -1]
        if float(np.percentile(alpha, percentile)) >= threshold:
            mat.alphaMode = "OPAQUE"
    return glb
