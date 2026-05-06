"""
PRNU-related denoising and fingerprint estimation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from scipy.signal import wiener

from src.preprocessing import load_image_rgb_float


def _resize_long_edge(rgb: np.ndarray, max_dim: int) -> np.ndarray:
    """
    Resize RGB float image so max(height, width) <= max_dim (area interpolation).

    Parameters
    ----------
    rgb : np.ndarray
        HxWx3 float32/float64.
    max_dim : int
        Maximum edge length in pixels.

    Returns
    -------
    np.ndarray
        Resized float32 RGB.
    """
    h, w = rgb.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return rgb.astype(np.float32)
    scale = max_dim / float(m)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    return cv2.resize(
        rgb.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA
    )


class WienerDenoiser:
    """
    Wiener filter denoising in the spatial domain (scipy.signal.wiener).

    Residual: W = I - F(I), where F is Wiener filtering on luminance.
    """

    def __init__(self, window_size: int = 3) -> None:
        """
        Parameters
        ----------
        window_size : int
            Odd-ish local window for ``scipy.signal.wiener`` (mysize).
        """
        self.window_size = int(window_size)

    def denoise(self, img: np.ndarray) -> np.ndarray:
        """
        Denoise a float RGB image and return float RGB denoised copy.

        Parameters
        ----------
        img : np.ndarray
            HxWx3 float32/float64 RGB in [0, 255] or [0,1].

        Returns
        -------
        np.ndarray
            Denoised RGB float32 array, same shape as input.
        """
        x = img.astype(np.float64)
        if x.max() <= 1.0:
            x = x * 255.0
        out = np.zeros_like(x, dtype=np.float64)
        ws = (self.window_size, self.window_size)
        for c in range(3):
            # Tiny offset avoids scipy wiener divide-by-zero on flat regions (uniform patches).
            ch = x[..., c] + 1e-4
            filt = wiener(ch, mysize=ws)
            out[..., c] = np.nan_to_num(filt, nan=ch, posinf=ch, neginf=ch)
        return out.astype(np.float32)

    def residual(self, img: np.ndarray) -> np.ndarray:
        """
        Compute noise residual W = I - F(I).

        Parameters
        ----------
        img : np.ndarray
            HxWx3 float RGB.

        Returns
        -------
        np.ndarray
            Residual W, float32 HxWx3.
        """
        fimg = self.denoise(img)
        return (img.astype(np.float32) - fimg.astype(np.float32)).astype(np.float32)


class PRNUFingerprintEstimator:
    """
    Estimate per-device reference PRNU fingerprints from training images.

    Uses K_d = sum_m (W_m ⊙ I_m) / sum_m (I_m^2 + eps) with per-channel accumulation.
    """

    def __init__(
        self,
        denoiser: WienerDenoiser,
        epsilon: float = 1e-6,
        resize_max_dim: Optional[int] = None,
        max_images_per_device: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        denoiser : WienerDenoiser
            Denoiser for residuals W_m.
        epsilon : float
            Stabilizer for division.
        resize_max_dim : int, optional
            If set, downscale each image so max side <= this before Wiener (much faster).
        max_images_per_device : int, optional
            If set, use at most this many images per device (sorted paths, first N).
        """
        self.denoiser = denoiser
        self.epsilon = float(epsilon)
        self.resize_max_dim = int(resize_max_dim) if resize_max_dim is not None else None
        self.max_images_per_device = (
            int(max_images_per_device) if max_images_per_device is not None else None
        )

    def estimate(self, image_paths: Iterable[str], device_id: str) -> np.ndarray:
        """
        Estimate fingerprint K_d for one device from a list of image paths.

        Parameters
        ----------
        image_paths : Iterable[str]
            Paths to training images for this device.
        device_id : str
            Device identifier (for logging only here).

        Returns
        -------
        np.ndarray
            Fingerprint array, float32 HxWx3 (same size as first image; subsequent
            images are resized to match the first).
        """
        paths = sorted(image_paths)
        if self.max_images_per_device is not None:
            paths = paths[: self.max_images_per_device]
        if not paths:
            raise ValueError("image_paths must be non-empty")

        k_num: np.ndarray | None = None
        k_den: np.ndarray | None = None
        shape_ref: tuple[int, int] | None = None

        for p in paths:
            rgb = load_image_rgb_float(p)
            if self.resize_max_dim is not None:
                rgb = _resize_long_edge(rgb, self.resize_max_dim)
            if shape_ref is None:
                shape_ref = (rgb.shape[0], rgb.shape[1])
            elif (rgb.shape[0], rgb.shape[1]) != shape_ref:
                rgb = cv2.resize(
                    rgb,
                    (shape_ref[1], shape_ref[0]),
                    interpolation=cv2.INTER_AREA,
                ).astype(np.float32)

            w = self.denoiser.residual(rgb)
            i = rgb.astype(np.float32)
            if k_num is None:
                k_num = np.zeros_like(w, dtype=np.float64)
                k_den = np.zeros_like(w, dtype=np.float64)
            k_num += (w * i).astype(np.float64)
            k_den += (i * i + self.epsilon).astype(np.float64)

        assert k_num is not None and k_den is not None
        k = (k_num / np.maximum(k_den, self.epsilon)).astype(np.float32)
        return k

    def save(self, fingerprint: np.ndarray, device_id: str, out_dir: str | Path) -> Path:
        """
        Save fingerprint tensor to ``.npy`` under ``out_dir``.

        Parameters
        ----------
        fingerprint : np.ndarray
            Fingerprint array to persist.
        device_id : str
            Device key used in filename (sanitized).
        out_dir : str | Path
            Output directory.

        Returns
        -------
        Path
            Path to written ``.npy`` file.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in device_id)
        path = out / f"fingerprint_{safe}.npy"
        np.save(path, fingerprint)
        return path

    def load(self, device_id: str, out_dir: str | Path) -> np.ndarray:
        """
        Load cached fingerprint if present.

        Parameters
        ----------
        device_id : str
            Device identifier.
        out_dir : str | Path
            Directory containing ``fingerprint_*.npy``.

        Returns
        -------
        np.ndarray
            Loaded fingerprint array.

        Raises
        ------
        FileNotFoundError
            If no cached file exists for the device.
        """
        out = Path(out_dir)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in device_id)
        path = out / f"fingerprint_{safe}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        return np.load(path)

    def has_cache(self, device_id: str, out_dir: str | Path) -> bool:
        """
        Return True if a cached ``.npy`` exists for ``device_id``.

        Parameters
        ----------
        device_id : str
            Device identifier.
        out_dir : str | Path
            Cache directory.

        Returns
        -------
        bool
            Whether the cache file exists.
        """
        out = Path(out_dir)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in device_id)
        return (out / f"fingerprint_{safe}.npy").is_file()


def fingerprint_estimator_from_config(
    denoiser: WienerDenoiser, cfg: dict
) -> PRNUFingerprintEstimator:
    """
    Build a ``PRNUFingerprintEstimator`` from a config dict.

    Recognized keys: ``fingerprint_epsilon``, ``fingerprint_resize_max_dim``,
    ``fingerprint_max_images_per_device``.

    Parameters
    ----------
    denoiser : WienerDenoiser
        Denoiser instance.
    cfg : dict
        Configuration mapping (e.g. loaded YAML).

    Returns
    -------
    PRNUFingerprintEstimator
        Configured estimator.
    """
    return PRNUFingerprintEstimator(
        denoiser,
        epsilon=float(cfg.get("fingerprint_epsilon", 1e-6)),
        resize_max_dim=cfg.get("fingerprint_resize_max_dim"),
        max_images_per_device=cfg.get("fingerprint_max_images_per_device"),
    )
