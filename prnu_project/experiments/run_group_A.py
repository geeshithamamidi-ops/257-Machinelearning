"""
Group A experiments: core performance under clean and social-media simulation.

Usage:
    python experiments/run_group_A.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compression import FlickrSimulator, JPEGSweep, WhatsAppSimulator
from src.data_loader import (
    DresdenLoader,
    PRNUPatchDataset,
    apply_train_sampling,
    attach_labels,
    build_device_label_map,
    count_images_under,
    prnu_collate,
)
from src.evaluate import (
    image_level_predictions_cnn,
    image_level_predictions_ncc,
    image_level_predictions_siamese,
    macro_f1,
    top_k_accuracy,
)
from src.models.cnn_classifier import CNNClassifier
from src.models.ncc_baseline import NCCBaseline
from src.models.siamese_network import SiameseClassifier
from src.preprocessing import sanitize_device_id
from src.prnu_extraction import (
    WienerDenoiser,
    fingerprint_estimator_from_config,
)
from src.train import get_git_hash, load_config, seed_everything, train_cnn, train_siamese


def _resolve_device(requested: str) -> str:
    """
    Resolve requested runtime device with CUDA availability checks.

    Parameters
    ----------
    requested : str
        CLI device argument (e.g., ``"cpu"``, ``"cuda"``, ``"cuda:0"``).

    Returns
    -------
    str
        Concrete device string for torch modules.
    """
    req = str(requested).strip().lower()
    if req in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req.startswith("cuda"):
        if torch.cuda.is_available():
            return requested
        print(
            "WARNING: CUDA requested but not available; falling back to CPU.",
            file=sys.stderr,
        )
        return "cpu"
    return requested


def _project_paths(cfg: dict[str, Any], project_root: Path) -> dict[str, Path]:
    """
    Resolve configured paths against the project root.

    Parameters
    ----------
    cfg : dict[str, Any]
        Loaded YAML configuration.
    project_root : Path
        ``prnu_project`` directory.

    Returns
    -------
    dict[str, Path]
        Absolute paths for ``dresden_root``, ``fingerprint_dir``.
    """
    dr = Path(cfg["dresden_root"])
    if not dr.is_absolute():
        dr = project_root / dr
    fd = Path(cfg["fingerprint_dir"])
    if not fd.is_absolute():
        fd = project_root / fd
    return {"dresden_root": dr, "fingerprint_dir": fd}


def _filter_devices(
    splits: dict[str, list[tuple[str, str]]],
    mapping: dict[str, int],
    max_devices: Optional[int],
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, int]]:
    """
    Restrict splits to the first ``max_devices`` lexicographic device ids.

    Parameters
    ----------
    splits : dict[str, list[tuple[str, str]]]
        Split dictionary.
    mapping : dict[str, int]
        Full device mapping.
    max_devices : int | None
        Optional cap.

    Returns
    -------
    tuple[dict[str, list[tuple[str, str]]], dict[str, int]]
        Filtered splits and remapped integer labels (0..K-1).
    """
    if max_devices is None or max_devices <= 0:
        return splits, mapping
    devs = sorted(mapping.keys())[: max_devices]
    keep = set(devs)
    out: dict[str, list[tuple[str, str]]] = {}
    for k, lst in splits.items():
        out[k] = [(p, d) for p, d in lst if d in keep]
    sub_map = {d: i for i, d in enumerate(sorted(keep))}
    return out, sub_map


def _estimate_fingerprints(
    train_split: list[tuple[str, str]],
    estimator,
    out_dir: Path,
) -> None:
    """
    Estimate and cache per-device fingerprints from training images.

    Parameters
    ----------
    train_split : list[tuple[str, str]]
        Training paths with device ids.
    estimator
        ``PRNUFingerprintEstimator`` instance.
    out_dir : Path
        Output directory for ``.npy`` files.

    Returns
    -------
    None
    """
    by_dev: dict[str, list[str]] = defaultdict(list)
    for p, d in train_split:
        by_dev[d].append(p)
    for dev in tqdm(sorted(by_dev.keys()), desc="Fingerprints"):
        if estimator.has_cache(dev, out_dir):
            continue
        fp = estimator.estimate(by_dev[dev], dev)
        estimator.save(fp, dev, out_dir)


class _JPEGUniformTransform:
    """
    Picklable per-image JPEG transform with a deterministic quality level.

    ``rng_seed`` is derived from ``(md5(path) ^ base_seed)`` so each image gets
    a reproducible quality draw while remaining picklable across DataLoader
    worker processes.
    """

    def __init__(self, rng_seed: int) -> None:
        self._rng_seed = int(rng_seed)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        rng = random.Random(self._rng_seed)
        q = rng.randint(50, 95)
        return JPEGSweep(q)(img)


class _JPEGUniformAugmentFactory:
    """
    Picklable per-image JPEG augmentation factory (Q ~ Uniform[50, 95]).

    Replaces a closure-based factory so PyTorch DataLoader workers can receive
    the dataset (and therefore this object) via ``multiprocessing`` spawn
    pickling without raising ``PicklingError``.
    """

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    def __call__(self, path: str) -> _JPEGUniformTransform:
        h = int(hashlib.md5(path.encode("utf-8")).hexdigest(), 16)
        return _JPEGUniformTransform((h ^ self.seed) & 0xFFFFFFFF)


def _jpeg_uniform_augment_factory(seed: int) -> _JPEGUniformAugmentFactory:
    """
    Build a per-image JPEG augmentation factory (Q ~ Uniform[50, 95]).

    Parameters
    ----------
    seed : int
        Base seed mixed with image path hash for reproducibility.

    Returns
    -------
    _JPEGUniformAugmentFactory
        Picklable factory instance mapping image path to transform.
    """
    return _JPEGUniformAugmentFactory(seed)


def _metrics_block(
    ncc: NCCBaseline,
    cnn: CNNClassifier,
    siam: SiameseClassifier,
    test_samples: list[tuple[str, int]],
    device_san_to_int: dict[str, int],
    cnn_loader: DataLoader,
    siam_loader: DataLoader,
    ncc_transform: Optional[Callable[[np.ndarray], np.ndarray]],
) -> dict[str, Any]:
    """
    Compute top1/top5/macro-F1 for all three models on a shared test condition.

    Parameters
    ----------
    ncc, cnn, siam : models
        Trained models.
    test_samples : list[tuple[str, int]]
        List of (path, label int).
    device_san_to_int : dict[str, int]
        Sanitized device string to class index.
    cnn_loader, siam_loader : DataLoader
        Patch loaders for CNN/Siamese.
    ncc_transform : callable | None
        Optional uint8 transform for NCC path.

    Returns
    -------
    dict[str, Any]
        Metrics block for JSON export.
    """
    yt_ncc, rk_ncc = image_level_predictions_ncc(
        ncc, test_samples, device_san_to_int, transform=ncc_transform
    )
    yt_cnn, rk_cnn = image_level_predictions_cnn(cnn, cnn_loader)
    yt_s, rk_s = image_level_predictions_siamese(siam, siam_loader)
    y_pred_ncc = [r[0] for r in rk_ncc]
    y_pred_cnn = [r[0] for r in rk_cnn]
    y_pred_s = [r[0] for r in rk_s]
    return {
        "NCC": {
            "top1": top_k_accuracy(yt_ncc, rk_ncc, k=1),
            "top5": top_k_accuracy(yt_ncc, rk_ncc, k=5),
            "macro_f1": macro_f1(yt_ncc, y_pred_ncc),
        },
        "CNN": {
            "top1": top_k_accuracy(yt_cnn, rk_cnn, k=1),
            "top5": top_k_accuracy(yt_cnn, rk_cnn, k=5),
            "macro_f1": macro_f1(yt_cnn, y_pred_cnn),
        },
        "Siamese": {
            "top1": top_k_accuracy(yt_s, rk_s, k=1),
            "top5": top_k_accuracy(yt_s, rk_s, k=5),
            "macro_f1": macro_f1(yt_s, y_pred_s),
        },
    }


def _average_blocks(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """
    Elementwise average metrics for two blocks (e.g., WhatsApp + Flickr).

    Parameters
    ----------
    a, b : dict[str, Any]
        Nested metric dicts with identical keys.

    Returns
    -------
    dict[str, Any]
        Averaged metrics.
    """
    out: dict[str, Any] = {}
    for model in ["NCC", "CNN", "Siamese"]:
        out[model] = {
            "top1": 0.5 * (a[model]["top1"] + b[model]["top1"]),
            "top5": 0.5 * (a[model]["top5"] + b[model]["top5"]),
            "macro_f1": 0.5 * (a[model]["macro_f1"] + b[model]["macro_f1"]),
        }
    return out


def main() -> None:
    """CLI entrypoint for Group A."""
    parser = argparse.ArgumentParser(description="Run Group A PRNU experiments")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--max-devices", type=int, default=None)
    parser.add_argument("--max-patches-per-image", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches prefetched per worker (ignored when num_workers=0).",
    )
    parser.add_argument(
        "--no-sampling",
        action="store_true",
        help="Disable the stratified train-split sampler regardless of config.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="Override sample_fraction from config (e.g. 0.10-0.30).",
    )
    parser.add_argument(
        "--residual-root",
        type=str,
        default=None,
        help="Optional precomputed residual root; if set, datasets load cached residuals.",
    )
    args = parser.parse_args()
    run_device = _resolve_device(args.device)
    use_cuda = str(run_device).startswith("cuda")

    project_root = ROOT
    cfg = load_config(project_root / args.config)
    seed = int(cfg["seed"])
    seed_everything(seed)
    paths = _project_paths(cfg, project_root)
    dresden_root = paths["dresden_root"].resolve()
    n_raw = count_images_under(dresden_root)
    if n_raw == 0:
        print(
            "ERROR: No images found under the Dresden root:\n"
            f"  {dresden_root}\n\n"
            "Download and unzip the dataset first, from the prnu_project directory:\n"
            "  kaggle datasets download -d micscodes/dresden-image-database "
            "-p data/raw/dresden --unzip\n"
            "or run:\n"
            "  bash scripts/download_data.sh\n",
            file=sys.stderr,
        )
        raise SystemExit(1)

    loader = DresdenLoader(
        paths["dresden_root"],
        min_images_per_device=int(cfg["min_images_per_device"]),
        seed=seed,
        train_ratio=float(cfg["train_ratio"]),
        val_ratio=float(cfg["val_ratio"]),
        test_ratio=float(cfg["test_ratio"]),
    )
    splits = loader.get_splits()
    mapping = build_device_label_map(
        splits["train"] + splits["val"] + splits["test"]
    )
    max_devices = args.max_devices
    if max_devices is None:
        max_devices = cfg.get("max_devices_subset")
    splits, mapping = _filter_devices(splits, mapping, max_devices)

    if args.no_sampling:
        cfg["use_sampling"] = False
    if args.sample_fraction is not None:
        cfg["use_sampling"] = True
        cfg["sample_fraction"] = float(args.sample_fraction)
    splits = apply_train_sampling(splits, cfg)

    if not splits["train"] or not mapping:
        print(
            "ERROR: No training samples after splitting/filtering.\n"
            f"  Raw images on disk: {n_raw}\n"
            "  Common causes: min_images_per_device is too high for this copy of Dresden, "
            "wrong folder layout (expect device subfolders under the Dresden root), "
            "or --max-devices set incorrectly.\n",
            file=sys.stderr,
        )
        raise SystemExit(1)

    inv_dev = {sanitize_device_id(k): v for k, v in mapping.items()}

    wiener_backend = str(cfg.get("wiener_backend", "scipy"))
    wiener_torch_device = str(cfg.get("wiener_torch_device", "cpu"))
    denoiser = WienerDenoiser(
        window_size=int(cfg["wiener_window"]),
        backend=wiener_backend,
        torch_device=wiener_torch_device,
    )
    est = fingerprint_estimator_from_config(denoiser, cfg)
    paths["fingerprint_dir"].mkdir(parents=True, exist_ok=True)
    _estimate_fingerprints(splits["train"], est, paths["fingerprint_dir"])

    ncc = NCCBaseline(
        denoiser=WienerDenoiser(
            window_size=int(cfg["wiener_window"]),
            backend=wiener_backend,
            torch_device=wiener_torch_device,
        )
    )
    ncc.fit(paths["fingerprint_dir"])

    ps = int(cfg["patch_size"])
    residual_root = args.residual_root or cfg.get("residual_cache_dir")
    train_ds = PRNUPatchDataset(
        attach_labels(splits["train"], mapping),
        patch_size=ps,
        denoiser=denoiser,
        max_patches_per_image=args.max_patches_per_image,
        residual_root=residual_root,
    )
    val_ds = PRNUPatchDataset(
        attach_labels(splits["val"], mapping),
        patch_size=ps,
        denoiser=denoiser,
        max_patches_per_image=args.max_patches_per_image,
        residual_root=residual_root,
    )
    test_ds = PRNUPatchDataset(
        attach_labels(splits["test"], mapping),
        patch_size=ps,
        denoiser=denoiser,
        max_patches_per_image=args.max_patches_per_image,
        residual_root=residual_root,
    )

    if len(train_ds) == 0:
        print(
            "ERROR: Training patch dataset is empty (0 patches).\n"
            f"  patch_size={ps}; training images: {len(splits['train'])}\n"
            "  Images may be smaller than patch_size, or no valid patches after grid split.\n",
            file=sys.stderr,
        )
        raise SystemExit(1)

    loader_workers = max(0, int(args.num_workers))
    if (
        str(wiener_backend).strip().lower() == "torch"
        and str(wiener_torch_device).strip().lower().startswith("cuda")
    ):
        # Colab/PyTorch cannot safely initialize CUDA inside forked DataLoader workers.
        loader_workers = 0
        print(
            "INFO: Using num_workers=0 because wiener_backend=torch on CUDA "
            "is not compatible with forked DataLoader workers.",
            flush=True,
        )
    loader_pin_memory = use_cuda
    loader_kwargs: dict[str, Any] = {
        "num_workers": loader_workers,
        "pin_memory": loader_pin_memory,
    }
    if loader_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["cnn_batch_size"]),
        shuffle=True,
        collate_fn=prnu_collate,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["cnn_batch_size"]),
        shuffle=False,
        collate_fn=prnu_collate,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg["cnn_batch_size"]),
        shuffle=False,
        collate_fn=prnu_collate,
        **loader_kwargs,
    )

    num_classes = len(mapping)
    cnn = CNNClassifier(
        num_classes,
        device=run_device,
        lr=float(cfg["cnn_lr"]),
        weight_decay=float(cfg["cnn_weight_decay"]),
        use_amp=bool(cfg.get("use_amp", True)),
        grad_accum_steps=int(cfg.get("grad_accum_steps", 1)),
    )
    train_cnn(
        cnn,
        train_loader,
        val_loader,
        epochs=int(cfg["cnn_epochs"]),
    )

    siam = SiameseClassifier(
        num_classes=num_classes,
        embedding_dim=int(cfg["siamese_embedding_dim"]),
        device=run_device,
        lr=float(cfg["siamese_lr"]),
        margin=float(cfg["siamese_margin"]),
        mode="triplet",
        use_amp=bool(cfg.get("use_amp", True)),
        grad_accum_steps=int(cfg.get("grad_accum_steps", 1)),
    )
    siam_train = DataLoader(
        train_ds,
        batch_size=int(cfg["siamese_batch_size"]),
        shuffle=True,
        collate_fn=prnu_collate,
        **loader_kwargs,
    )
    train_siamese(siam, siam_train, epochs=int(cfg["siamese_epochs"]))
    siam.fit_centroids(siam_train)

    test_paths = splits["test"]
    test_labeled = attach_labels(test_paths, mapping)

    A1 = _metrics_block(
        ncc,
        cnn,
        siam,
        test_labeled,
        inv_dev,
        test_loader,
        test_loader,
        None,
    )

    wa = WhatsAppSimulator()
    fl = FlickrSimulator()

    test_ds_wa = PRNUPatchDataset(
        test_labeled,
        patch_size=ps,
        transform=wa,
        denoiser=denoiser,
        max_patches_per_image=args.max_patches_per_image,
    )
    test_ds_fl = PRNUPatchDataset(
        test_labeled,
        patch_size=ps,
        transform=fl,
        denoiser=denoiser,
        max_patches_per_image=args.max_patches_per_image,
    )
    tl_wa = DataLoader(
        test_ds_wa,
        batch_size=int(cfg["cnn_batch_size"]),
        shuffle=False,
        collate_fn=prnu_collate,
        **loader_kwargs,
    )
    tl_fl = DataLoader(
        test_ds_fl,
        batch_size=int(cfg["cnn_batch_size"]),
        shuffle=False,
        collate_fn=prnu_collate,
        **loader_kwargs,
    )

    A2_wa = _metrics_block(
        ncc,
        cnn,
        siam,
        test_labeled,
        inv_dev,
        tl_wa,
        tl_wa,
        wa,
    )
    A2_fl = _metrics_block(
        ncc,
        cnn,
        siam,
        test_labeled,
        inv_dev,
        tl_fl,
        tl_fl,
        fl,
    )
    A2 = _average_blocks(A2_wa, A2_fl)

    train_ds_a3 = PRNUPatchDataset(
        attach_labels(splits["train"], mapping),
        patch_size=ps,
        denoiser=denoiser,
        max_patches_per_image=args.max_patches_per_image,
        per_path_transform_factory=_jpeg_uniform_augment_factory(seed),
    )
    val_ds_a3 = PRNUPatchDataset(
        attach_labels(splits["val"], mapping),
        patch_size=ps,
        denoiser=denoiser,
        max_patches_per_image=args.max_patches_per_image,
        residual_root=residual_root,
    )
    train_loader_a3 = DataLoader(
        train_ds_a3,
        batch_size=int(cfg["cnn_batch_size"]),
        shuffle=True,
        collate_fn=prnu_collate,
        **loader_kwargs,
    )
    val_loader_a3 = DataLoader(
        val_ds_a3,
        batch_size=int(cfg["cnn_batch_size"]),
        shuffle=False,
        collate_fn=prnu_collate,
        **loader_kwargs,
    )
    cnn_a3 = CNNClassifier(
        num_classes,
        device=run_device,
        lr=float(cfg["cnn_lr"]),
        weight_decay=float(cfg["cnn_weight_decay"]),
        use_amp=bool(cfg.get("use_amp", True)),
        grad_accum_steps=int(cfg.get("grad_accum_steps", 1)),
    )
    train_cnn(
        cnn_a3,
        train_loader_a3,
        val_loader_a3,
        epochs=int(cfg["cnn_epochs"]),
    )
    siam_a3 = SiameseClassifier(
        num_classes=num_classes,
        embedding_dim=int(cfg["siamese_embedding_dim"]),
        device=run_device,
        lr=float(cfg["siamese_lr"]),
        margin=float(cfg["siamese_margin"]),
        mode="triplet",
        use_amp=bool(cfg.get("use_amp", True)),
        grad_accum_steps=int(cfg.get("grad_accum_steps", 1)),
    )
    siam_train_a3 = DataLoader(
        train_ds_a3,
        batch_size=int(cfg["siamese_batch_size"]),
        shuffle=True,
        collate_fn=prnu_collate,
        **loader_kwargs,
    )
    train_siamese(siam_a3, siam_train_a3, epochs=int(cfg["siamese_epochs"]))
    siam_a3.fit_centroids(siam_train_a3)

    A3_wa = _metrics_block(
        ncc,
        cnn_a3,
        siam_a3,
        test_labeled,
        inv_dev,
        tl_wa,
        tl_wa,
        wa,
    )
    A3_fl = _metrics_block(
        ncc,
        cnn_a3,
        siam_a3,
        test_labeled,
        inv_dev,
        tl_fl,
        tl_fl,
        fl,
    )
    A3 = _average_blocks(A3_wa, A3_fl)

    out = {
        "experiment": "group_A",
        "config": cfg,
        "git_hash": get_git_hash(),
        "A1": A1,
        "A2": A2,
        "A3": A3,
    }
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "group_A.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
