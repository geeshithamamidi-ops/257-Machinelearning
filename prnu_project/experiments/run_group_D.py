"""
Group D experiments: per-device robustness, error taxonomy, runtime profiling.

Usage:
    python experiments/run_group_D.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compression import WhatsAppSimulator
from src.data_loader import (
    DresdenLoader,
    PRNUPatchDataset,
    apply_train_sampling,
    attach_labels,
    build_device_label_map,
    prnu_collate,
)
from src.evaluate import image_level_predictions_cnn, image_level_predictions_ncc
from src.models.cnn_classifier import CNNClassifier
from src.models.ncc_baseline import NCCBaseline
from src.preprocessing import load_image_rgb_float, sanitize_device_id
from src.prnu_extraction import WienerDenoiser, fingerprint_estimator_from_config
from src.train import get_git_hash, load_config, seed_everything


def _filter_devices(
    splits: dict[str, list[tuple[str, str]]],
    mapping: dict[str, int],
    max_devices: Optional[int],
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, int]]:
    """
    Restrict splits to ``max_devices`` devices.

    Parameters
    ----------
    splits : dict
        Splits.
    mapping : dict[str, int]
        Label map.
    max_devices : int | None
        Cap.

    Returns
    -------
    tuple
        Filtered splits and remapped labels.
    """
    if max_devices is None or max_devices <= 0:
        return splits, mapping
    keep = set(sorted(mapping.keys())[: max_devices])
    out = {k: [(p, d) for p, d in v if d in keep] for k, v in splits.items()}
    sub = {d: i for i, d in enumerate(sorted(keep))}
    return out, sub


def _family_key(device_name: str) -> str:
    """
    Heuristic family key from Dresden-style device string.

    Parameters
    ----------
    device_name : str
        Device folder name.

    Returns
    -------
    str
        Family identifier (first two underscore-separated tokens when possible).
    """
    parts = device_name.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


def _center_crop_rgb(img: np.ndarray, size: int) -> np.ndarray:
    """
    Center-crop HxWx3 image to ``size x size``.

    Parameters
    ----------
    img : np.ndarray
        RGB uint8/float.
    size : int
        Square edge.

    Returns
    -------
    np.ndarray
        Cropped image.
    """
    h, w = img.shape[:2]
    if h < size or w < size:
        scale = size / min(h, w)
        nh, nw = int(h * scale), int(w * scale)
        img = cv2.resize(
            img.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_AREA
        )
        h, w = img.shape[:2]
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return img[y0 : y0 + size, x0 : x0 + size, :]


def _profile_nc(
    ncc: NCCBaseline,
    img: np.ndarray,
    crop: Optional[int],
    repeats: int = 5,
) -> tuple[float, float]:
    """
    Measure mean inference time (ms) and peak memory (MB) for NCC on one image.

    Parameters
    ----------
    ncc : NCCBaseline
        Model.
    img : np.ndarray
        RGB uint8 image.
    crop : int | None
        Optional center crop size before ``predict``.
    repeats : int
        Repetitions for timing average.

    Returns
    -------
    tuple[float, float]
        ``(ms_per_image, peak_mb)``.
    """
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    if crop is not None:
        u8 = _center_crop_rgb(u8, int(crop))
    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(repeats):
        ncc.predict(u8)
    ms = (time.perf_counter() - t0) / repeats * 1000.0
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return float(ms), float(peak / (1024 * 1024))


def _profile_cnn(
    clf: CNNClassifier,
    patch: torch.Tensor,
    repeats: int = 10,
) -> tuple[float, float]:
    """
    Profile CNN on a single patch tensor (already batched 1x1xHxW).

    Parameters
    ----------
    clf : CNNClassifier
        Classifier.
    patch : torch.Tensor
        Single patch.
    repeats : int
        Repetitions.

    Returns
    -------
    tuple[float, float]
        ``(ms_per_patch, peak_mb)``.
    """
    dev = clf.device
    x = patch.to(dev)
    tracemalloc.start()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            _ = clf.model(x)
    ms = (time.perf_counter() - t0) / repeats * 1000.0
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return float(ms), float(peak / (1024 * 1024))


def main() -> None:
    """Run Group D experiments."""
    parser = argparse.ArgumentParser(description="Group D forensic analysis")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--max-devices", type=int, default=None)
    parser.add_argument("--cnn-epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
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
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    seed_everything(int(cfg["seed"]))

    dresden_root = Path(cfg["dresden_root"])
    if not dresden_root.is_absolute():
        dresden_root = ROOT / dresden_root
    fp_dir = Path(cfg["fingerprint_dir"])
    if not fp_dir.is_absolute():
        fp_dir = ROOT / fp_dir
    fp_dir.mkdir(parents=True, exist_ok=True)

    dl = DresdenLoader(
        dresden_root,
        min_images_per_device=int(cfg["min_images_per_device"]),
        seed=int(cfg["seed"]),
        train_ratio=float(cfg["train_ratio"]),
        val_ratio=float(cfg["val_ratio"]),
        test_ratio=float(cfg["test_ratio"]),
    )
    splits = dl.get_splits()
    mapping = build_device_label_map(
        splits["train"] + splits["val"] + splits["test"]
    )
    splits, mapping = _filter_devices(splits, mapping, args.max_devices)

    if args.no_sampling:
        cfg["use_sampling"] = False
    if args.sample_fraction is not None:
        cfg["use_sampling"] = True
        cfg["sample_fraction"] = float(args.sample_fraction)
    splits = apply_train_sampling(splits, cfg)

    inv_label = {v: k for k, v in mapping.items()}
    san_map = {sanitize_device_id(k): v for k, v in mapping.items()}

    denoiser = WienerDenoiser(window_size=int(cfg["wiener_window"]))
    est = fingerprint_estimator_from_config(denoiser, cfg)
    by_dev: dict[str, list[str]] = {}
    for p, d in splits["train"]:
        by_dev.setdefault(d, []).append(p)
    for dev in tqdm(sorted(by_dev.keys()), desc="Fingerprints(D)"):
        if est.has_cache(dev, fp_dir):
            continue
        fp = est.estimate(by_dev[dev], dev)
        est.save(fp, dev, fp_dir)

    ncc = NCCBaseline(denoiser=WienerDenoiser(window_size=int(cfg["wiener_window"])))
    ncc.fit(fp_dir)

    ps = int(cfg["patch_size"])
    train_ds = PRNUPatchDataset(
        attach_labels(splits["train"], mapping), patch_size=ps, denoiser=denoiser
    )
    tr_ld = DataLoader(
        train_ds,
        batch_size=int(cfg["cnn_batch_size"]),
        shuffle=True,
        num_workers=2,
        collate_fn=prnu_collate,
    )
    cnn_epochs = int(args.cnn_epochs or cfg["cnn_epochs"])
    cnn = CNNClassifier(
        len(mapping),
        device=args.device,
        lr=float(cfg["cnn_lr"]),
        weight_decay=float(cfg["cnn_weight_decay"]),
    )
    for _ in range(cnn_epochs):
        cnn.train_epoch(tr_ld)

    wa = WhatsAppSimulator()
    test_lab = attach_labels(splits["test"], mapping)

    # D1 per-device robustness (WhatsApp)
    per_dev: dict[str, dict[str, float]] = {}
    fragile: list[str] = []
    for lab in tqdm(sorted(inv_label.keys()), desc="Per-device WA"):
        dev_name = inv_label[lab]
        subset = [(p, y) for p, y in test_lab if y == lab]
        if not subset:
            continue
        yt, rk = image_level_predictions_ncc(
            ncc, subset, san_map, transform=wa
        )
        y_pred = [r[0] for r in rk]
        acc = float(np.mean([int(a == b) for a, b in zip(yt, y_pred)]))
        per_dev[dev_name] = {"top1_whatsapp": acc}
        if acc < 0.5:
            fragile.append(dev_name)

    # D2 confusion + taxonomy on clean test (CNN patch majority via softmax average)
    test_ds = PRNUPatchDataset(
        test_lab, patch_size=ps, denoiser=denoiser
    )
    te_ld = DataLoader(
        test_ds,
        batch_size=int(cfg["cnn_batch_size"]),
        shuffle=False,
        num_workers=2,
        collate_fn=prnu_collate,
    )
    yt, rk_cnn = image_level_predictions_cnn(cnn, te_ld)
    y_pred = [r[0] for r in rk_cnn]
    cm = confusion_matrix(yt, y_pred, labels=sorted(mapping.values())).tolist()
    same_fam = 0
    cross_fam = 0
    total_err = 0
    for t, p in zip(yt, y_pred):
        if t != p:
            total_err += 1
            tf = _family_key(inv_label[t])
            pf = _family_key(inv_label[p])
            if tf == pf:
                same_fam += 1
            else:
                cross_fam += 1
    sf_rate = float(same_fam / total_err) if total_err else 0.0
    cf_rate = float(cross_fam / total_err) if total_err else 0.0

    # D3 runtime profiling on CPU
    sample_path = splits["test"][0][0]
    img = load_image_rgb_float(sample_path)
    u8 = np.clip(img, 0, 255).astype(np.uint8)

    prof: dict[str, Any] = {"NCC": {}, "CNN": {}}
    for pz in (64, 128, 256):
        ms, mb = _profile_nc(ncc, u8, crop=pz)
        prof["NCC"][f"patch_{pz}"] = {"ms_per_image": ms, "peak_mb": mb}

    # CNN synthetic patch size
    cnn.model.eval()
    for pz in (64, 128, 256):
        x = torch.zeros(1, 1, pz, pz, dtype=torch.float32)
        ms, mb = _profile_cnn(cnn, x)
        prof["CNN"][f"patch_{pz}"] = {"ms_per_patch": ms, "peak_mb": mb}

    # Siamese profiling omitted class definition reuse CNN-shaped encoder for timing proxy
    from src.models.siamese_network import SiameseEncoder

    enc = SiameseEncoder(embedding_dim=int(cfg["siamese_embedding_dim"])).to(
        args.device
    )
    enc.eval()
    prof["Siamese"] = {}
    for pz in (64, 128, 256):
        x = torch.zeros(1, 1, pz, pz, dtype=torch.float32).to(args.device)
        tracemalloc.start()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(10):
                _ = enc(x)
        ms = (time.perf_counter() - t0) / 10 * 1000.0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        prof["Siamese"][f"patch_{pz}"] = {
            "ms_per_patch": float(ms),
            "peak_mb": float(peak / (1024 * 1024)),
        }

    out = {
        "experiment": "group_D",
        "config": cfg,
        "git_hash": get_git_hash(),
        "D1": {
            "per_device_whatsapp": per_dev,
            "forensically_fragile": fragile,
        },
        "D2": {
            "confusion_matrix": cm,
            "label_order": [inv_label[i] for i in sorted(inv_label.keys())],
            "same_family_error_rate": sf_rate,
            "cross_family_error_rate": cf_rate,
        },
        "D3": {"profiling_cpu": prof},
    }
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "results" / "group_D.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
