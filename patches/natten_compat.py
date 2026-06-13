"""
Make natten 0.17.5 (needed by the NAF upsampler's na2d_qk/na2d_av API)
import under torch >= 2.6 on a CUDA-less machine. Patches the installed
package in site-packages; idempotent. natten still runs CPU-only — MPS
tensors are handled by keeping NAF on CPU (see mps_compat.patch_naf_cpu).

Run after `pip install natten==0.17.5`:
    python patches/natten_compat.py
"""

import importlib.util
import os

# find_spec locates the package without importing it — a fresh (unpatched)
# natten install fails at import, which is exactly what this script fixes.
_spec = importlib.util.find_spec("natten")
NATTEN_DIR = os.path.dirname(os.path.abspath(_spec.origin))


def main():
    import glob

    patched = 0

    # torch.cuda._device_t typing alias was removed in newer torch
    for path in glob.glob(os.path.join(NATTEN_DIR, "**", "*.py"), recursive=True):
        src = open(path).read()
        old = "from torch.cuda import _device_t"
        if old in src:
            src = src.replace(
                old,
                'from typing import Union as _U; _device_t = _U[int, str, "torch.device", None]',
            )
            open(path, "w").write(src)
            print(f"  Patched _device_t: {os.path.relpath(path, NATTEN_DIR)}")
            patched += 1

    # get_device_cc assumes a CUDA device exists
    misc = os.path.join(NATTEN_DIR, "utils", "misc.py")
    src = open(misc).read()
    old = "    major, minor = torch.cuda.get_device_capability(device_index)"
    if old in src and "is_available" not in src.split("def get_device_cc")[1].split("\ndef ")[0]:
        src = src.replace(old, "    if not torch.cuda.is_available():\n        return 0\n" + old, 1)
        open(misc, "w").write(src)
        print("  Patched get_device_cc: utils/misc.py")
        patched += 1

    print("natten compat: done" if patched else "natten compat: already patched")


if __name__ == "__main__":
    main()
