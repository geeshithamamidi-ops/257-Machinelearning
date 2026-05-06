"""
Image compression and social-media simulation utilities.

Provides JPEG sweeps, resize stress tests, and empirically parameterized
WhatsApp/Flickr simulators (resize + JPEG round-trip).
"""

from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
from PIL import Image


def _ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """
    Ensure image is HxWx3 uint8 RGB.

    Parameters
    ----------
    img : np.ndarray
        Input array in RGB order, uint8 or float.

    Returns
    -------
    np.ndarray
        uint8 RGB array of shape (H, W, 3).
    """
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.dtype != np.uint8:
        img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    return img


def _resize_max_dim(img: np.ndarray, max_dim: int) -> np.ndarray:
    """
    Resize so max(height, width) <= max_dim using bicubic interpolation.

    Parameters
    ----------
    img : np.ndarray
        HxWx3 uint8 RGB.
    max_dim : int
        Maximum allowed edge length.

    Returns
    -------
    np.ndarray
        Resized RGB image (unchanged if already within bounds).
    """
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img
    scale = max_dim / float(m)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)


def _jpeg_roundtrip(img: np.ndarray, quality: int) -> np.ndarray:
    """
    Encode RGB image as JPEG and decode back to numpy (uint8 RGB).

    Parameters
    ----------
    img : np.ndarray
        HxWx3 uint8 RGB.
    quality : int
        JPEG quality 1-95 (Pillow scale).

    Returns
    -------
    np.ndarray
        Decoded RGB array.
    """
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality), optimize=True)
    buf.seek(0)
    out = np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)
    return out


class WhatsAppSimulator:
    """
    Empirically measured WhatsApp compression pipeline:

    1. Resize so max dimension <= 1600px (bicubic)
    2. JPEG encode at Q=77
    3. JPEG decode back to numpy array
    """

    MAX_DIM = 1600
    JPEG_QUALITY = 77

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply simulated WhatsApp compression.

        Parameters
        ----------
        img : np.ndarray
            Input image (HxW grayscale or HxWx3 RGB).

        Returns
        -------
        np.ndarray
            Compressed RGB uint8 image (HxWx3).
        """
        rgb = _ensure_uint8_rgb(img)
        resized = _resize_max_dim(rgb, self.MAX_DIM)
        return _jpeg_roundtrip(resized, self.JPEG_QUALITY)


class FlickrSimulator:
    """
    Empirically measured Flickr compression pipeline:

    1. Resize so max dimension <= 2048px (bicubic)
    2. JPEG encode at Q=92
    3. JPEG decode back to numpy array
    """

    MAX_DIM = 2048
    JPEG_QUALITY = 92

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply simulated Flickr compression.

        Parameters
        ----------
        img : np.ndarray
            Input image (HxW or HxWx3).

        Returns
        -------
        np.ndarray
            Compressed RGB uint8 image (HxWx3).
        """
        rgb = _ensure_uint8_rgb(img)
        resized = _resize_max_dim(rgb, self.MAX_DIM)
        return _jpeg_roundtrip(resized, self.JPEG_QUALITY)


class JPEGSweep:
    """Apply JPEG compression at a fixed quality Q."""

    QUALITY_LEVELS = [95, 90, 80, 70, 60, 50, 40, 30]

    def __init__(self, quality: int) -> None:
        """
        Parameters
        ----------
        quality : int
            JPEG quality level (Pillow 1-95).
        """
        self.quality = int(quality)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        JPEG-compress the image at fixed Q without resizing.

        Parameters
        ----------
        img : np.ndarray
            HxW or HxWx3 uint8 image.

        Returns
        -------
        np.ndarray
            RGB uint8 array after JPEG round-trip.
        """
        rgb = _ensure_uint8_rgb(img)
        return _jpeg_roundtrip(rgb, self.quality)


class ResizeSweep:
    """
    Downsample by scale factor then upsample back to original size (bicubic).

    Stress-tests resampling + interpolation artifacts.
    """

    SCALE_FACTORS = [0.9, 0.8, 0.7, 0.6, 0.5]

    def __init__(self, scale: float) -> None:
        """
        Parameters
        ----------
        scale : float
            Downsample factor in (0, 1]; output is restored to input resolution.
        """
        if not 0 < scale <= 1.0:
            raise ValueError("scale must be in (0, 1]")
        self.scale = float(scale)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Resize down then back to original height/width.

        Parameters
        ----------
        img : np.ndarray
            HxW or HxWx3 image.

        Returns
        -------
        np.ndarray
            RGB uint8 image matching original spatial size.
        """
        rgb = _ensure_uint8_rgb(img)
        h, w = rgb.shape[:2]
        nh, nw = max(1, int(round(h * self.scale))), max(1, int(round(w * self.scale)))
        small = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
        restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        return restored.astype(np.uint8)
