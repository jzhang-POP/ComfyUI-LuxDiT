"""Run by ComfyUI-Manager after cloning, or by hand: python install.py

Installs Python deps into the current interpreter and fetches the upstream
LuxDiT source at the pinned commit. Model weights download on first use.
"""

import importlib.util
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])


pip("-r", os.path.join(HERE, "requirements.txt"))
if importlib.util.find_spec("cv2") is None:
    pip("opencv-python-headless")

sys.path.insert(0, HERE)
import luxdit_bridge  # noqa: E402

luxdit_bridge.ensure_upstream()
print("[LuxDiT] install complete")
