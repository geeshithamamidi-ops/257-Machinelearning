"""
Dataset loaders for Dresden, IEEE SP, and PRNU patch PyTorch datasets.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import bisect
from pathlib import Path
from typing import Callable, Iterator, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing import load_image_rgb_float
from src.prnu_extraction import WienerDenoiser
from src.sampler import StratifiedDeviceSampler

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def apply_train_sampling(
    splits: dict[str, list[tuple[str, object]]],
    cfg: dict[str, object],
    verbose: bool = True,
) -> dict[str, list[tuple[str, object]]]:
    """
    Reduce the ``train`` split via :class:`StratifiedDeviceSampler` if enabled.

    Only the ``train`` split is affected; ``val`` and ``test`` are returned
    unchanged so evaluation remains valid on the full held-out data.

    Parameters
    ----------
    splits : dict[str, list[tuple[str, object]]]
        Split dictionary with keys ``train``/``val``/``test``.
    cfg : dict[str, object]
        Loaded YAML configuration. Relevant keys:
        ``use_sampling`` (bool, default False),
        ``sample_fraction`` (float, default 0.20),
        ``sample_min_per_class`` (int, default 30),
        ``seed`` (int, default 42).
    verbose : bool
        If True, print a one-line summary after sampling.

    Returns
    -------
    dict[str, list[tuple[str, object]]]
        New split dict; ``val`` and ``test`` are the same objects as in the
        input, ``train`` is the sampled list.
    """
    if not bool(cfg.get("use_sampling", False)):
        return splits
    sampler = StratifiedDeviceSampler(
        fraction=float(cfg.get("sample_fraction", 0.20)),
        min_per_class=int(cfg.get("sample_min_per_class", 30)),
        seed=int(cfg.get("seed", 42)),
    )
    sampled_train = sampler.sample(list(splits.get("train", [])))
    summary = sampler.summary()
    if verbose:
        print(
            "[sampler] train reduced: "
            f"{summary['total_original']} -> {summary['total_sampled']} "
            f"({summary['fraction_achieved']:.1%}), "
            f"classes={summary['classes_total']}, "
            f"per-class=[{summary['min_per_class']}"
            f"..{summary['max_per_class']}], "
            f"below-floor classes={summary['classes_below_floor']}",
            flush=True,
        )
    out = dict(splits)
    out["train"] = sampled_train
    return out


def apply_device_image_subset(
    splits: dict[str, list[tuple[str, str]]],
    num_devices: int = 5,
    images_per_device: int = 100,
    seed: int = 42,
) -> dict[str, list[tuple[str, str]]]:
    """
    Build a reproducible small subset across all splits for fast experiments.

    Sampling policy:
    - Select up to ``num_devices`` devices uniformly at random.
    - For each selected device, sample up to ``images_per_device`` images total
      from the union of train/val/test.
    - Preserve each sampled image's original split assignment.

    Parameters
    ----------
    splits : dict[str, list[tuple[str, str]]]
        Split dict with keys like ``train``, ``val``, ``test``.
    num_devices : int
        Maximum number of devices to keep.
    images_per_device : int
        Maximum total images to keep per selected device.
    seed : int
        RNG seed for reproducible sampling.

    Returns
    -------
    dict[str, list[tuple[str, str]]]
        New split dictionary containing only the sampled subset.
    """
    if num_devices <= 0 or images_per_device <= 0:
        raise ValueError("num_devices and images_per_device must be positive")
    rng = random.Random(int(seed))
    all_items = list(splits.get("train", [])) + list(splits.get("val", [])) + list(
        splits.get("test", [])
    )
    if not all_items:
        return {k: [] for k in splits.keys()}

    devices = sorted({d for _, d in all_items})
    if len(devices) > num_devices:
        selected_devices = set(rng.sample(devices, k=num_devices))
    else:
        selected_devices = set(devices)

    split_by_path: dict[str, str] = {}
    for split_name, items in splits.items():
        for p, _ in items:
            split_by_path[p] = split_name

    by_device: dict[str, list[tuple[str, str]]] = {}
    for p, d in all_items:
        if d in selected_devices:
            by_device.setdefault(d, []).append((p, d))

    out: dict[str, list[tuple[str, str]]] = {k: [] for k in splits.keys()}
    for dev in sorted(selected_devices):
        items = by_device.get(dev, [])
        if not items:
            continue
        if len(items) > images_per_device:
            picked = rng.sample(items, k=images_per_device)
        else:
            picked = list(items)
        for p, d in picked:
            split_name = split_by_path.get(p, "train")
            out.setdefault(split_name, []).append((p, d))
    return out


def _list_images(root: Path) -> list[Path]:
    """
    Recursively list image files under ``root``.

    Parameters
    ----------
    root : Path
        Root directory.

    Returns
    -------
    list[Path]
        Sorted list of image paths.
    """
    out: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                out.append(p)
    return sorted(out)


def count_images_under(root: str | Path) -> int:
    """
    Count supported image files under ``root`` (recursive).

    Parameters
    ----------
    root : str | Path
        Directory to scan (must exist as a directory to count).

    Returns
    -------
    int
        Number of image paths found; 0 if missing or not a directory.
    """
    p = Path(root)
    if not p.is_dir():
        return 0
    return len(_list_images(p))


def residual_cache_path(residual_root: str | Path, image_path: str) -> Path:
    """
    Deterministically map an image path to a cached residual file path.

    Parameters
    ----------
    residual_root : str | Path
        Root directory for precomputed residuals.
    image_path : str
        Absolute or relative source image path.

    Returns
    -------
    Path
        Path under ``residual_root/<device>/<hash>.npy``.
    """
    p = Path(image_path)
    device = (p.parent.name or "unknown").replace(os.sep, "_")
    h = hashlib.md5(str(p.resolve()).encode("utf-8")).hexdigest()
    return Path(residual_root) / device / f"{h}.npy"


def residual_cache_pt_path(residual_root: str | Path, image_path: str) -> Path:
    """
    Deterministically map an image path to a cached residual tensor (.pt).

    Parameters
    ----------
    residual_root : str | Path
        Root directory for precomputed residuals.
    image_path : str
        Absolute or relative source image path.

    Returns
    -------
    Path
        Path under ``residual_root/<device>/<hash>.pt``.
    """
    p = Path(image_path)
    device = (p.parent.name or "unknown").replace(os.sep, "_")
    h = hashlib.md5(str(p.resolve()).encode("utf-8")).hexdigest()
    return Path(residual_root) / device / f"{h}.pt"


def _parse_dresden_sample(
    root: Path, path: Path
) -> tuple[str, str, str]:
    """
    Parse (device_id, scene_id, image_key) from a path under Dresden root.

    Parameters
    ----------
    root : Path
        Dataset root.
    path : Path
        Absolute path to an image.

    Returns
    -------
    tuple[str, str, str]
        device_id, scene_id, unique image key for bookkeeping.
    """
    rel = path.relative_to(root)
    parts = rel.parts
    if len(parts) < 2:
        device_id = "unknown"
        scene_id = path.stem
    else:
        device_id = parts[0]
        if len(parts) >= 3:
            scene_id = parts[1]
        else:
            stem = path.stem
            scene_id = re.sub(r"_\d+$", "", stem)
            if scene_id == stem:
                scene_id = stem
    image_key = str(rel)
    return device_id, scene_id, image_key


class DresdenLoader:
    """
    Load Dresden dataset images with scene-aware stratified splits.

    - Filters devices with at least ``min_images_per_device`` images.
    - Scene-aware: images sharing the same (device, scene) stay in one split.
    - Split ratios: train/val/test from config defaults 60/20/20.
    """

    def __init__(
        self,
        root_dir: str | Path,
        min_images_per_device: int = 50,
        seed: int = 42,
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        test_ratio: float = 0.20,
    ) -> None:
        """
        Parameters
        ----------
        root_dir : str | Path
            Path to unpacked Dresden root (contains device folders).
        min_images_per_device : int
            Minimum total images per device to retain the device.
        seed : int
            RNG seed for splitting.
        train_ratio, val_ratio, test_ratio : float
            Must sum to 1.0.
        """
        self.root = Path(root_dir)
        self.min_images = int(min_images_per_device)
        self.seed = int(seed)
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)

    def _group_by_scene(self) -> dict[tuple[str, str], list[Path]]:
        """
        Group image paths by (device, scene).

        Returns
        -------
        dict[tuple[str, str], list[Path]]
            Mapping scene key to image paths.
        """
        groups: dict[tuple[str, str], list[Path]] = {}
        for p in _list_images(self.root):
            dev, sc, _ = _parse_dresden_sample(self.root, p)
            groups.setdefault((dev, sc), []).append(p)
        return groups

    def get_splits(self) -> dict[str, list[tuple[str, str]]]:
        """
        Build stratified scene-aware splits.

        Returns
        -------
        dict[str, list[tuple[str, str]]]
            Keys ``train``, ``val``, ``test`` mapping to lists of
            ``(absolute_path_str, device_id)``.
        """
        groups = self._group_by_scene()
        by_device: dict[str, list[tuple[str, str]]] = {}
        for (dev, sc), paths in groups.items():
            for p in paths:
                by_device.setdefault(dev, []).append((str(p.resolve()), dev))

        device_counts = {d: len(v) for d, v in by_device.items()}
        kept = [d for d, c in device_counts.items() if c >= self.min_images]
        kept.sort()

        rng = random.Random(self.seed)
        train: list[tuple[str, str]] = []
        val: list[tuple[str, str]] = []
        test: list[tuple[str, str]] = []

        scene_groups: dict[str, dict[tuple[str, str], list[str]]] = {}
        for (dev, sc), plist in groups.items():
            if dev not in kept:
                continue
            scene_groups.setdefault(dev, {}).setdefault((dev, sc), []).extend(
                [str(x.resolve()) for x in plist]
            )

        for dev in kept:
            scenes = list(scene_groups[dev].keys())
            rng.shuffle(scenes)
            n = len(scenes)
            if n == 0:
                continue
            nt = int(round(n * self.train_ratio))
            nv = int(round(n * self.val_ratio))
            if n >= 3 and nt + nv >= n:
                nv = max(1, n - nt - 1)
            if n == 1:
                tr_scenes, va_scenes, te_scenes = set(scenes), set(), set()
            elif n == 2:
                tr_scenes, va_scenes, te_scenes = {scenes[0]}, {scenes[1]}, set()
            else:
                nt = max(1, min(nt, n - 2))
                nv = max(1, min(nv, n - nt - 1))
                tr_scenes = set(scenes[:nt])
                va_scenes = set(scenes[nt : nt + nv])
                te_scenes = set(scenes[nt + nv :])

            for sc_key, img_paths in scene_groups[dev].items():
                if sc_key in tr_scenes:
                    bucket = train
                elif sc_key in va_scenes:
                    bucket = val
                else:
                    bucket = test
                for ip in img_paths:
                    bucket.append((ip, dev))

        return {"train": train, "val": val, "test": test}


class IEEESPLoader:
    """
    Load IEEE SP Camera Model Identification data from ``train/`` (and optional ``test/``).

    Training images are organized by camera model subfolders. Test images may be flat
    without public labels; for supervised metrics this loader can split ``train/`` only.
    """

    def __init__(self, root_dir: str | Path, seed: int = 42) -> None:
        """
        Parameters
        ----------
        root_dir : str | Path
            Unpacked competition root containing ``train/`` (and optionally ``test/``).
        seed : int
            RNG seed for splitting.
        """
        self.root = Path(root_dir)
        self.seed = int(seed)

    def _class_dirs(self) -> list[Path]:
        """
        List class directories under ``train/``.

        Returns
        -------
        list[Path]
            Sorted class directory paths.
        """
        train = self.root / "train"
        if not train.is_dir():
            return []
        return sorted([p for p in train.iterdir() if p.is_dir()])

    def get_label_mapping(self) -> dict[str, int]:
        """
        Map folder name (camera model string) to integer label.

        Returns
        -------
        dict[str, int]
            Bijective mapping for classes found under ``train/``.
        """
        names = [p.name for p in self._class_dirs()]
        names.sort()
        return {n: i for i, n in enumerate(names)}

    def get_splits(
        self,
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        test_ratio: float = 0.20,
    ) -> dict[str, list[tuple[str, int]]]:
        """
        Stratified split over **training** folder images (scene-agnostic per image).

        Returns
        -------
        dict[str, list[tuple[str, int]]]
            ``train``, ``val``, ``test`` with ``(path, label_int)``.
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1")
        mapping = self.get_label_mapping()
        rng = random.Random(self.seed)
        splits: dict[str, list[tuple[str, int]]] = {"train": [], "val": [], "test": []}
        for name, lab in mapping.items():
            paths = _list_images(self.root / "train" / name)
            rng.shuffle(paths)
            n = len(paths)
            nt = int(round(n * train_ratio))
            nv = int(round(n * val_ratio))
            for i, p in enumerate(paths):
                if i < nt:
                    splits["train"].append((str(p.resolve()), lab))
                elif i < nt + nv:
                    splits["val"].append((str(p.resolve()), lab))
                else:
                    splits["test"].append((str(p.resolve()), lab))
        return splits

    def iter_test_unlabeled(self) -> Iterator[Path]:
        """
        Yield paths under ``test/`` if present (filenames only).

        Yields
        ------
        Path
            Test image path.
        """
        test_dir = self.root / "test"
        if not test_dir.is_dir():
            return iter(())
        return iter(_list_images(test_dir))


class PRNUPatchDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """
    PyTorch Dataset of non-overlapping PRNU residual patches.

    Ensures splits are defined at **image** level elsewhere; this class only consumes
    lists built from disjoint image sets so no patch from a test parent appears in train.
    """

    def __init__(
        self,
        samples: list[tuple[str, int]],
        patch_size: int = 64,
        resize_hw: Optional[tuple[int, int]] = None,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        denoiser: Optional[WienerDenoiser] = None,
        max_patches_per_image: Optional[int] = None,
        per_path_transform_factory: Optional[
            Callable[[str], Callable[[np.ndarray], np.ndarray]]
        ] = None,
        cache_dir: Optional[str | Path] = None,
        residual_root: Optional[str | Path] = None,
    ) -> None:
        """
        Parameters
        ----------
        samples : list[tuple[str, int]]
            ``(image_path, device_label_int)`` entries.
        patch_size : int
            Patch edge length (square).
        transform : callable, optional
            Applied to uint8 RGB HxWx3 **before** residual extraction.
        denoiser : WienerDenoiser, optional
            Defaults to a new ``WienerDenoiser(window_size=3)``.
        max_patches_per_image : int, optional
            Cap patches sampled per image (first grid patches in raster order).
        per_path_transform_factory : callable, optional
            If set, ``factory(path) -> transform`` so all patches from one image share
            the same transform instance (e.g., fixed random JPEG quality per image).
        """
        self.samples = list(samples)
        self.patch_size = int(patch_size)
        # Optional fixed resize keeps tensors tiny in fast debug mode (e.g. 128x128),
        # which dramatically reduces denoising and patch extraction cost.
        self.resize_hw = tuple(int(x) for x in resize_hw) if resize_hw is not None else None
        self.transform = transform
        self.per_path_transform_factory = per_path_transform_factory
        self.denoiser = denoiser or WienerDenoiser(window_size=3)
        env_cap = os.environ.get("PRNU_MAX_PATCHES_PER_IMAGE", "").strip()
        if max_patches_per_image is None and env_cap:
            try:
                env_val = int(env_cap)
                if env_val > 0:
                    max_patches_per_image = env_val
            except ValueError:
                pass
        self.max_patches_per_image = max_patches_per_image
        self.residual_root: Optional[Path] = (
            Path(residual_root) if residual_root else None
        )
        env_cache = os.environ.get("PRNU_PATCH_CACHE_DIR", "").strip() or None
        self.cache_dir: Optional[Path] = (
            Path(cache_dir) if cache_dir else (Path(env_cache) if env_cache else None)
        )
        if self.cache_dir is not None and self._cache_eligible():
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._image_paths: list[str] = []
        self._labels: list[int] = []
        self._patch_hw: list[tuple[int, int]] = []
        self._npatches: list[int] = []
        self._offsets: list[int] = []
        self._residual_paths: list[Optional[Path]] = []
        self._build_index()

    def _build_index(self) -> None:
        """
        Pre-compute raster-order patch index for all images.

        Returns
        -------
        None
        """
        from PIL import Image as _PILImage
        self._offsets = [0]
        skipped = 0
        fast_path = self.transform is None and self.per_path_transform_factory is None
        for path, lab in self.samples:
            residual_path: Optional[Path] = None
            if self.residual_root is not None:
                pt_path = residual_cache_pt_path(self.residual_root, path)
                npy_path = residual_cache_path(self.residual_root, path)
                residual_path = pt_path if pt_path.is_file() else npy_path
            if fast_path:
                if residual_path is not None and residual_path.is_file():
                    try:
                        if residual_path.suffix == ".pt":
                            ten = torch.load(residual_path, map_location="cpu")
                            if isinstance(ten, torch.Tensor):
                                arr = ten.numpy()
                            else:
                                raise ValueError(
                                    f"Unexpected .pt residual payload type: {type(ten)}"
                                )
                        else:
                            arr = np.load(residual_path, mmap_mode="r")
                        if self.resize_hw is not None:
                            h, w = self.resize_hw
                        else:
                            h, w = int(arr.shape[0]), int(arr.shape[1])
                    except Exception:
                        skipped += 1
                        continue
                else:
                    try:
                        with _PILImage.open(path) as _im:
                            if self.resize_hw is not None:
                                h, w = self.resize_hw
                            else:
                                w, h = _im.size
                    except Exception:
                        skipped += 1
                        continue
            else:
                try:
                    rgb = load_image_rgb_float(path)
                except Exception:
                    skipped += 1
                    continue
                u8 = np.clip(rgb, 0, 255).astype(np.uint8)
                if self.per_path_transform_factory is not None:
                    try:
                        u8 = self.per_path_transform_factory(path)(u8)
                    except Exception:
                        skipped += 1
                        continue
                elif self.transform is not None:
                    try:
                        u8 = self.transform(u8)
                    except Exception:
                        skipped += 1
                        continue
                if self.resize_hw is not None:
                    h_t, w_t = self.resize_hw
                    u8 = cv2.resize(u8, (w_t, h_t), interpolation=cv2.INTER_AREA)
                h, w = u8.shape[:2]
            ps = self.patch_size
            nh, nw = h // ps, w // ps
            if nh == 0 or nw == 0:
                continue
            n_grid = nh * nw
            if self.max_patches_per_image is not None:
                n_grid = min(n_grid, self.max_patches_per_image)
            self._image_paths.append(path)
            self._labels.append(int(lab))
            self._patch_hw.append((nh, nw))
            self._npatches.append(n_grid)
            self._residual_paths.append(residual_path if residual_path and residual_path.is_file() else None)
            self._offsets.append(self._offsets[-1] + n_grid)

    def __len__(self) -> int:
        """
        Number of patches in the dataset.

        Returns
        -------
        int
            Patch count.
        """
        return self._offsets[-1] if len(self._offsets) > 1 else 0

    def _decode_index(self, idx: int) -> tuple[int, int]:
        """
        Map flat patch index to (image_index, patch_index_within_image).

        Parameters
        ----------
        idx : int
            Global patch index.

        Returns
        -------
        tuple[int, int]
            Image list index and local patch index.
        """
        j = bisect.bisect_right(self._offsets, idx) - 1
        if j < 0 or j >= len(self._image_paths):
            raise IndexError("patch index out of range")
        local = idx - self._offsets[j]
        return j, local

    def _cache_eligible(self) -> bool:
        """Return True when caching extracted patches is safe (no transforms)."""
        return self.transform is None and self.per_path_transform_factory is None

    def _cache_file(self, path: str) -> Optional[Path]:
        """Return per-image cache file path, or None if caching disabled."""
        if self.cache_dir is None or not self._cache_eligible():
            return None
        ws = getattr(self.denoiser, "window_size", 3)
        key = hashlib.md5(
            f"{os.path.abspath(path)}|ps={self.patch_size}|w={ws}".encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{key}.npy"

    def _compute_all_patches(
        self, path: str, nh: int, nw: int, n_keep: int, residual_path: Optional[Path]
    ) -> np.ndarray:
        """Denoise full image once and extract the first ``n_keep`` grayscale patches."""
        can_use_precomputed = (
            residual_path is not None
            and residual_path.is_file()
            and self.transform is None
            and self.per_path_transform_factory is None
        )
        if can_use_precomputed:
            if residual_path.suffix == ".pt":
                ten = torch.load(residual_path, map_location="cpu")
                if isinstance(ten, torch.Tensor):
                    res = ten.numpy()
                else:
                    raise ValueError(f"Unexpected .pt residual payload type: {type(ten)}")
            else:
                res = np.load(residual_path)
            if res.ndim == 2:
                res = res[..., None]
            if self.resize_hw is not None:
                h_t, w_t = self.resize_hw
                res = cv2.resize(res, (w_t, h_t), interpolation=cv2.INTER_AREA)
                if res.ndim == 2:
                    res = res[..., None]
        else:
            rgb = load_image_rgb_float(path)
            u8 = np.clip(rgb, 0, 255).astype(np.uint8)
            if self.per_path_transform_factory is not None:
                u8 = self.per_path_transform_factory(path)(u8)
            elif self.transform is not None:
                u8 = self.transform(u8)
            if self.resize_hw is not None:
                h_t, w_t = self.resize_hw
                u8 = cv2.resize(u8, (w_t, h_t), interpolation=cv2.INTER_AREA)
            rgb = u8.astype(np.float32)
            res = self.denoiser.residual(rgb)
        ps = self.patch_size
        patches = np.empty((n_keep, ps, ps), dtype=np.float32)
        for k in range(n_keep):
            r = k // nw
            c = k % nw
            patch = res[r * ps : (r + 1) * ps, c * ps : (c + 1) * ps, :]
            patches[k] = patch.mean(axis=2)
        return patches

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        """
        Load one patch: residual channel (grayscale mean), label, parent path.

        Parameters
        ----------
        index : int
            Patch index.

        Returns
        -------
        tuple[torch.Tensor, int, str]
            ``(1 x ps x ps`` float tensor, label int, parent image path string).
        """
        ii, pi = self._decode_index(index)
        path = self._image_paths[ii]
        lab = self._labels[ii]
        residual_path = self._residual_paths[ii]
        nh, nw = self._patch_hw[ii]
        total_grid = self._npatches[ii]
        if pi >= total_grid:
            raise IndexError("patch index out of range")

        ps = self.patch_size
        cache_file = self._cache_file(path)

        if cache_file is not None and cache_file.exists():
            try:
                cached = np.load(cache_file, mmap_mode="r")
                if cached.shape[0] >= total_grid and cached.shape[1:] == (ps, ps):
                    gray = np.asarray(cached[pi], dtype=np.float32)
                    ten = torch.from_numpy(gray[None, :, :].copy()).float()
                    return ten, lab, path
            except Exception:
                pass

        patches = self._compute_all_patches(path, nh, nw, total_grid, residual_path)
        if cache_file is not None:
            try:
                tmp_path = cache_file.parent / (cache_file.stem + f".tmp{os.getpid()}.npy")
                np.save(tmp_path, patches.astype(np.float16))
                os.replace(tmp_path, cache_file)
            except Exception:
                pass

        gray = patches[pi]
        ten = torch.from_numpy(gray[None, :, :].copy()).float()
        return ten, lab, path


def build_device_label_map(paths_and_devices: list[tuple[str, str]]) -> dict[str, int]:
    """
    Create stable integer labels for device string ids.

    Parameters
    ----------
    paths_and_devices : list[tuple[str, str]]
        Pairs ``(path, device_id)``.

    Returns
    -------
    dict[str, int]
        Mapping device string -> int.
    """
    devs = sorted({d for _, d in paths_and_devices})
    return {d: i for i, d in enumerate(devs)}


def prnu_collate(
    batch: list[tuple[torch.Tensor, int, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Stack patch tensors and preserve parent paths for aggregation.

    Parameters
    ----------
    batch : list[tuple[torch.Tensor, int, str]]
        Mini-batch from ``PRNUPatchDataset``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, list[str]]
        ``(x, y, paths)``.
    """
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    paths = [b[2] for b in batch]
    return xs, ys, paths


def attach_labels(
    split: list[tuple[str, str]], mapping: dict[str, int]
) -> list[tuple[str, int]]:
    """
    Replace string device ids with ints using ``mapping``.

    Parameters
    ----------
    split : list[tuple[str, str]]
        List of (path, device_str).
    mapping : dict[str, int]
        Device string to int.

    Returns
    -------
    list[tuple[str, int]]
        Labeled samples.
    """
    return [(p, mapping[d]) for p, d in split]
