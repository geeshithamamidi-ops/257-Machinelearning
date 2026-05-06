"""
Group B experiments: ablations (Wiener window, training data efficiency).

Usage:
    python experiments/run_group_B.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import (
    DresdenLoader,
    PRNUPatchDataset,
    apply_train_sampling,
    attach_labels,
    build_device_label_map,
    prnu_collate,
)
from src.evaluate import image_level_predictions_ncc, macro_f1, top_k_accuracy
from src.models.cnn_classifier import CNNClassifier
from src.models.ncc_baseline import NCCBaseline
from src.preprocessing import sanitize_device_id
from src.prnu_extraction import WienerDenoiser, fingerprint_estimator_from_config
from src.train import get_git_hash, load_config, seed_everything, train_cnn


def _filter_devices(
    splits: dict[str, list[tuple[str, str]]],
    mapping: dict[str, int],
    max_devices: Optional[int],
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, int]]:
    """
    Restrict splits to at most ``max_devices`` devices and re-index labels.

    Parameters
    ----------
    splits : dict[str, list[tuple[str, str]]]
        Split dictionary.
    mapping : dict[str, int]
        Original mapping.
    max_devices : int | None
        Optional cap.

    Returns
    -------
    tuple[dict[str, list[tuple[str, str]]], dict[str, int]]
        Filtered splits and new mapping.
    """
    if max_devices is None or max_devices <= 0:
        return splits, mapping
    keep = set(sorted(mapping.keys())[: max_devices])
    out = {k: [(p, d) for p, d in v if d in keep] for k, v in splits.items()}
    sub_map = {d: i for i, d in enumerate(sorted(keep))}
    return out, sub_map


def _estimate_fps(
    train: list[tuple[str, str]],
    est,
    out_dir: Path,
) -> None:
    """
    Estimate fingerprints for all training devices.

    Parameters
    ----------
    train : list[tuple[str, str]]
        Training list.
    est
        Fingerprint estimator.
    out_dir : Path
        Output directory.

    Returns
    -------
    None
    """
    by_dev: dict[str, list[str]] = defaultdict(list)
    for p, d in train:
        by_dev[d].append(p)
    for dev in tqdm(sorted(by_dev.keys()), desc="Fingerprints(B)"):
        if est.has_cache(dev, out_dir):
            continue
        fp = est.estimate(by_dev[dev], dev)
        est.save(fp, dev, out_dir)


def main() -> None:
    """Run Group B experiments."""
    parser = argparse.ArgumentParser(description="Group B ablations")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--max-devices", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.5)
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
    seed = int(cfg["seed"])
    seed_everything(seed)

    dresden_root = Path(cfg["dresden_root"])
    if not dresden_root.is_absolute():
        dresden_root = ROOT / dresden_root
    fp_dir = Path(cfg["fingerprint_dir"])
    if not fp_dir.is_absolute():
        fp_dir = ROOT / fp_dir
    fp_dir.mkdir(parents=True, exist_ok=True)

    loader = DresdenLoader(
        dresden_root,
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
    splits, mapping = _filter_devices(splits, mapping, args.max_devices)

    if args.no_sampling:
        cfg["use_sampling"] = False
    if args.sample_fraction is not None:
        cfg["use_sampling"] = True
        cfg["sample_fraction"] = float(args.sample_fraction)
    splits = apply_train_sampling(splits, cfg)

    inv_san = {sanitize_device_id(k): v for k, v in mapping.items()}

    # B1: Wiener window ablation for NCC on validation images
    b1: dict[str, Any] = {}
    for w in (3, 5):
        den = WienerDenoiser(window_size=w)
        est = fingerprint_estimator_from_config(den, cfg)
        subdir = fp_dir / f"ablation_wiener_{w}"
        subdir.mkdir(parents=True, exist_ok=True)
        _estimate_fps(splits["train"], est, subdir)
        ncc = NCCBaseline(denoiser=den)
        ncc.fit(subdir)
        val_lab = attach_labels(splits["val"], mapping)
        yt, rk = image_level_predictions_ncc(ncc, val_lab, inv_san, transform=None)
        y_pred = [r[0] for r in rk]
        b1[f"wiener_{w}"] = {
            "top1": top_k_accuracy(yt, rk, k=1),
            "macro_f1": macro_f1(yt, y_pred),
        }

    # B2: CNN trained on a random subset of training images (data efficiency)
    rng = random.Random(seed)
    train_all = splits["train"][:]
    rng.shuffle(train_all)
    n_keep = max(1, int(len(train_all) * float(args.train_fraction)))
    train_sub = train_all[:n_keep]

    denoiser = WienerDenoiser(window_size=int(cfg["wiener_window"]))
    ps = int(cfg["patch_size"])
    train_ds = PRNUPatchDataset(
        attach_labels(train_sub, mapping), patch_size=ps, denoiser=denoiser
    )
    val_ds = PRNUPatchDataset(
        attach_labels(splits["val"], mapping), patch_size=ps, denoiser=denoiser
    )
    tr_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["cnn_batch_size"]),
        shuffle=True,
        num_workers=2,
        collate_fn=prnu_collate,
    )
    va_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["cnn_batch_size"]),
        shuffle=False,
        num_workers=2,
        collate_fn=prnu_collate,
    )
    cnn_epochs = (
        int(args.cnn_epochs)
        if args.cnn_epochs is not None
        else int(cfg["cnn_epochs"])
    )
    cnn = CNNClassifier(
        len(mapping),
        device=args.device,
        lr=float(cfg["cnn_lr"]),
        weight_decay=float(cfg["cnn_weight_decay"]),
    )
    for ep in range(cnn_epochs):
        cnn.train_epoch(tr_loader)
    ev = cnn.evaluate(va_loader)

    out = {
        "experiment": "group_B",
        "config": cfg,
        "git_hash": get_git_hash(),
        "B1": {"wiener_ablation": b1},
        "B2": {
            "train_fraction": float(args.train_fraction),
            "cnn_val_accuracy": ev["accuracy"],
            "n_train_images_used": n_keep,
        },
    }
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "results" / "group_B.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
