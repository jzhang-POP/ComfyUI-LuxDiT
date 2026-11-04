"""Fetches NVIDIA's LuxDiT source and imports the pieces the nodes need.

The upstream code is NOT vendored in this repo (it is under the NVIDIA OneWay
Noncommercial License). It is cloned at a pinned commit into ``third_party/``
on first use, and imported under an isolated module namespace so its top-level
``src`` package cannot collide with any other custom node's ``src``.
"""

import importlib
import os
import subprocess
import sys
import threading

LUXDIT_REPO = "https://github.com/nv-tlabs/LuxDiT.git"
LUXDIT_COMMIT = "a191884afbd757bab546bb01f2eaa967fb80b375"

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
UPSTREAM_DIR = os.path.join(NODE_DIR, "third_party", "LuxDiT")
_SENTINEL = os.path.join(UPSTREAM_DIR, "src", "pipelines", "pipeline_cogvideox_rgbxenv.py")

_lock = threading.Lock()
_modules = None


def ensure_upstream():
    """Clone LuxDiT at the pinned commit if it is not already present."""
    if os.path.isfile(_SENTINEL):
        return UPSTREAM_DIR
    os.makedirs(os.path.dirname(UPSTREAM_DIR), exist_ok=True)
    print(f"[LuxDiT] cloning {LUXDIT_REPO} @ {LUXDIT_COMMIT[:10]} -> {UPSTREAM_DIR}")
    subprocess.check_call(["git", "clone", "--quiet", LUXDIT_REPO, UPSTREAM_DIR])
    subprocess.check_call(["git", "-C", UPSTREAM_DIR, "checkout", "--quiet", LUXDIT_COMMIT])
    return UPSTREAM_DIR


def _is_src(name):
    return name == "src" or name.startswith("src.")


def load():
    """Import the upstream modules once and return them as a dict."""
    global _modules
    with _lock:
        if _modules is not None:
            return _modules
        ensure_upstream()

        # Park any foreign `src` package, import LuxDiT's, then move LuxDiT's
        # modules to a private alias and put the foreign ones back.
        parked = {k: sys.modules.pop(k) for k in list(sys.modules) if _is_src(k)}
        sys.path.insert(0, UPSTREAM_DIR)
        try:
            mods = {
                "transformer": importlib.import_module("src.models.custom_cogvideox_transformer_3d"),
                "vae": importlib.import_module("src.models.custom_autoencoder_kl_cogvideox"),
                "pipeline": importlib.import_module("src.pipelines.pipeline_cogvideox_rgbxenv"),
                "hdr": importlib.import_module("src.models.hdr_model"),
                "render": importlib.import_module("src.data.rendering_utils"),
            }
        finally:
            try:
                sys.path.remove(UPSTREAM_DIR)
            except ValueError:
                pass
            for k in [k for k in sys.modules if _is_src(k)]:
                sys.modules["_luxdit_upstream." + k] = sys.modules.pop(k)
            sys.modules.update(parked)

        _modules = mods
        return _modules
