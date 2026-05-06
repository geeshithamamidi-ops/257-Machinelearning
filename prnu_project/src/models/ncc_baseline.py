"""
Normalized cross-correlation baseline for PRNU fingerprint matching.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np

from src.prnu_extraction import WienerDenoiser


def _to_gray(residual: np.ndarray) -> np.ndarray:
    """
    Convert HxWx3 residual to grayscale (mean channel).

    Parameters
    ----------
    residual : np.ndarray
        Residual tensor.

    Returns
    -------
    np.ndarray
        2D float array.
    """
    if residual.ndim == 2:
        return residual.astype(np.float64)
    return residual.mean(axis=2).astype(np.float64)


def _crop_center(a: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Center-crop array to (h, w).

    Parameters
    ----------
    a : np.ndarray
        2D array.
    h : int
        Target height.
    w : int
        Target width.

    Returns
    -------
    np.ndarray
        Cropped array.
    """
    H, W = a.shape[:2]
    y0 = max(0, (H - h) // 2)
    x0 = max(0, (W - w) // 2)
    return a[y0 : y0 + h, x0 : x0 + w]


def _ncc_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    Normalized cross-correlation between same-shaped 2D patches.

    Parameters
    ----------
    a, b : np.ndarray
        Same shape 2D arrays.

    Returns
    -------
    float
        Pearson correlation in [-1, 1].
    """
    aa = a.astype(np.float64).ravel()
    bb = b.astype(np.float64).ravel()
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-12
    return float(np.dot(aa, bb) / denom)


def _pce_like_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    Simple PCE-style emphasis: NCC scaled by energy ratio (diagnostic).

    Parameters
    ----------
    a, b : np.ndarray
        Same-shaped 2D arrays.

    Returns
    -------
    float
        Heuristic score (higher is better match).
    """
    ncc = _ncc_score(a, b)
    e = float(np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()) + 1e-12)
    return ncc * np.sqrt(e)


class NCCBaseline:
    """
    Normalized cross-correlation device attribution using pre-computed fingerprints.

    For a test image, compute residual ``W_t``, then score each device fingerprint
    ``K_d`` with NCC (optionally PCE-like scaling).
    """

    def __init__(
        self,
        denoiser: Optional[WienerDenoiser] = None,
        use_pce: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        denoiser : WienerDenoiser, optional
            Denoiser for test residual; default window 3.
        use_pce : bool
            If True, use PCE-like score instead of raw NCC.
        """
        self.denoiser = denoiser or WienerDenoiser(window_size=3)
        self.use_pce = bool(use_pce)
        self._fingerprints: dict[str, np.ndarray] = {}
        self._device_order: list[str] = []

    def fit(self, fingerprint_dir: str | Path) -> None:
        """
        Load ``fingerprint_*.npy`` tensors from disk.

        Parameters
        ----------
        fingerprint_dir : str | Path
            Directory containing ``fingerprint_<device>.npy`` files.

        Returns
        -------
        None
        """
        d = Path(fingerprint_dir)
        self._fingerprints.clear()
        for p in sorted(d.glob("fingerprint_*.npy")):
            m = re.match(r"fingerprint_(.+)\.npy", p.name)
            key = m.group(1) if m else p.stem.replace("fingerprint_", "")
            self._fingerprints[key] = np.load(p)
        self._device_order = sorted(self._fingerprints.keys())

    def _align(self, w: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Spatially align two 2D maps by center cropping to common shape.

        Parameters
        ----------
        w, k : np.ndarray
            2D maps.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Cropped ``w``, cropped ``k``.
        """
        h = min(w.shape[0], k.shape[0])
        ww = min(w.shape[1], k.shape[1])
        wc = _crop_center(w, h, ww)
        kc = _crop_center(k, h, ww)
        return wc, kc

    def predict(self, image: np.ndarray) -> list[tuple[str, float]]:
        """
        Rank devices by correlation score for a single RGB uint8/float image.

        Parameters
        ----------
        image : np.ndarray
            HxWx3 RGB image.

        Returns
        -------
        list[tuple[str, float]]
            Sorted list ``(device_id, score)`` descending.
        """
        rgb = image.astype(np.float32)
        if rgb.max() <= 1.0:
            rgb = rgb * 255.0
        w_t = _to_gray(self.denoiser.residual(rgb))
        scores: list[tuple[str, float]] = []
        for dev in self._device_order:
            k = _to_gray(self._fingerprints[dev])
            wa, ka = self._align(w_t, k)
            if self.use_pce:
                s = _pce_like_score(wa, ka)
            else:
                s = _ncc_score(wa, ka)
            scores.append((dev, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def predict_batch(
        self, images: list[np.ndarray]
    ) -> list[list[tuple[str, float]]]:
        """
        Predict ranked lists for a batch of images.

        Parameters
        ----------
        images : list[np.ndarray]
            List of HxWx3 images.

        Returns
        -------
        list[list[tuple[str, float]]]
            Per-image ranked device lists.
        """
        return [self.predict(im) for im in images]
