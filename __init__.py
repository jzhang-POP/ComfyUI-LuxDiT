import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

from . import hdr_tools, photo_check, photo_paste, sun_prior
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **hdr_tools.NODE_CLASS_MAPPINGS, **sun_prior.NODE_CLASS_MAPPINGS,
                       **photo_check.NODE_CLASS_MAPPINGS, **photo_paste.NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **hdr_tools.NODE_DISPLAY_NAME_MAPPINGS,
                              **sun_prior.NODE_DISPLAY_NAME_MAPPINGS, **photo_check.NODE_DISPLAY_NAME_MAPPINGS,
                              **photo_paste.NODE_DISPLAY_NAME_MAPPINGS}
WEB_DIRECTORY = "./js"  # sun_picker.js: click the sun on the LuxDiT Sun (prior) preview

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
