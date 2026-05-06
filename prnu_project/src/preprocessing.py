"""
Image loading and basic preprocessing (no EXIF usage).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

PathLike = Union[str, Path]


def sanitize_device_id(device_id: str) -> str:
    """
    Normalize a device string for safe filenames and fingerprint keys.

    Parameters
    ----------
    device_id : str
        Raw device label from folder names.

    Returns
    -------
    str
        Sanitized identifier.
    """
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in device_id)


def load_image_rgb_float(path: PathLike) -> np.ndarray:
    """
    Load an image as HxWx3 float32 RGB in [0, 255] range.

    Strips metadata by re-encoding through Pillow pixel buffer only
    (no EXIF is exposed to models).

    Parameters
    ----------
    path : PathLike
        Filesystem path to an image file.

    Returns
    -------
    np.ndarray
        Float32 array of shape (H, W, 3).
    """
    p = Path(path)
    with Image.open(p) as im:
        rgb = im.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)
    return arr.astype(np.float32)
