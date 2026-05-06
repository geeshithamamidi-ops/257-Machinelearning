"""
Group C experiments: robustness (JPEG/resize), cross-dataset JPEG sweep, calibration.

Usage:
    python experiments/run_group_C.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_diagram_data,
    selective_risk_curve,
)
from src.compression import JPEGSweep, ResizeSweep
from src.data_loader import (
    DresdenLoader,
    IEEESPLoader,
    PRNUPatchDataset,
    apply_train_sampling,
    attach_labels,
    build_device_label_map,
    prnu_collate,
)
from src.evaluate import (
    aurc,
    image_level_predictions_cnn,
    image_level_predictions_ncc,
    image_level_predictions_siamese,
    relative_accuracy_drop,
    top_k_accuracy,
)
from src.models.cnn_classifier import CNNClassifier
from src.models.ncc_baseline import NCCBaseline
from src.models.siamese_network import SiameseClassifier
from src.preprocessing import sanitize_device_id
from src.prnu_extraction import WienerDenoiser, fingerprint_estimator_from_config
from src.train import get_git_hash, load_config, seed_everything, train_cnn, train_siamese


def _filter_devices(
    splits: dict[str, list[tuple[str, str]]],
    mapping: dict[str, int],
    max_devices: Optional[int],
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, int]]:
    """
    Restrict splits to ``max_devices`` lexicographically first devices.

    Parameters
    ----------
    splits : dict
        Split dict.
    mapping : dict[str, int]
        Label mapping.
    max_devices : int | None
        Optional cap.

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


def _estimate_fps(
    train: list[tuple[str, str]],
    est,
    out_dir: Path,
) -> None:
    """
    Estimate per-device fingerprints.

    Parameters
    ----------
    train : list[tuple[str, str]]
        Training paths.
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
    for dev in tqdm(sorted(by_dev.keys()), desc="Fingerprints(C)"):
        if est.has_cache(dev, out_dir):
            continue
        fp = est.estimate(by_dev[dev], dev)
        est.save(fp, dev, out_dir)


def _train_dresden_stack(
    cfg: dict[str, Any],
    splits: dict[str, list[tuple[str, str]]],
    mapping: dict[str, int],
    fp_dir: Path,
    device: str,
    epochs_cnn: int,
    epochs_siam: int,
) -> tuple[NCCBaseline, CNNClassifier, SiameseClassifier, WienerDenoiser, int]:
    """
    Train NCC + CNN + Siamese on Dresden training split.

    Returns
    -------
    tuple
        Models, denoiser, patch_size.
    """
    denoiser = WienerDenoiser(window_size=int(cfg["wiener_window"]))
    est = fingerprint_estimator_from_config(denoiser, cfg)
    fp_dir.mkdir(parents=True, exist_ok=True)
    _estimate_fps(splits["train"], est, fp_dir)
    ncc = NCCBaseline(denoiser=WienerDenoiser(window_size=int(cfg["wiener_window"])))
    ncc.fit(fp_dir)

    ps = int(cfg["patch_size"])
    train_ds = PRNUPatchDataset(
        attach_labels(splits["train"], mapping), patch_size=ps, denoiser=denoiser
    )
    val_ds = PRNUPatchDataset(
        attach_labels(splits["val"], mapping), patch_size=ps, denoiser=denoiser
    )
    tr_ld = DataLoader(
        train_ds,
        batch_size=int(cfg["cnn_batch_size"]),
        shuffle=True,
        num_workers=2,
        collate_fn=prnu_collate,
    )
    va_ld = DataLoader(
        val_ds,
        batch_size=int(cfg["cnn_batch_size"]),
        shuffle=False,
        num_workers=2,
        collate_fn=prnu_collate,
    )
    nc = len(mapping)
    cnn = CNNClassifier(
        nc,
        device=device,
        lr=float(cfg["cnn_lr"]),
        weight_decay=float(cfg["cnn_weight_decay"]),
    )
    for _ in range(epochs_cnn):
        cnn.train_epoch(tr_ld)
    siam = SiameseClassifier(
        num_classes=nc,
        embedding_dim=int(cfg["siamese_embedding_dim"]),
        device=device,
        lr=float(cfg["siamese_lr"]),
        margin=float(cfg["siamese_margin"]),
        mode="triplet",
    )
    s_tr = DataLoader(
        train_ds,
        batch_size=int(cfg["siamese_batch_size"]),
        shuffle=True,
        num_workers=2,
        collate_fn=prnu_collate,
    )
    for _ in range(epochs_siam):
        siam.train_epoch(s_tr)
    siam.fit_centroids(s_tr)
    return ncc, cnn, siam, denoiser, ps


def _top1_nc(
    ncc: NCCBaseline,
    test: list[tuple[str, int]],
    san_map: dict[str, int],
    tf,
) -> float:
    """
    Top-1 accuracy for NCC under optional uint8 transform ``tf``.

    Parameters
    ----------
    ncc : NCCBaseline
        Model.
    test : list[tuple[str, int]]
        Labeled paths.
    san_map : dict[str, int]
        Sanitized device->int map.
    tf : callable | None
        Transform.

    Returns
    -------
    float
        Top-1 accuracy.
    """
    yt, rk = image_level_predictions_ncc(ncc, test, san_map, transform=tf)
    return top_k_accuracy(yt, rk, k=1)


def _top1_cnn(
    clf: CNNClassifier, loader: DataLoader
) -> float:
    """
    Top-1 image-level accuracy for CNN using averaged softmax.

    Parameters
    ----------
    clf : CNNClassifier
        Classifier.
    loader : DataLoader
        Patch loader.

    Returns
    -------
    float
        Top-1 accuracy.
    """
    yt, rk = image_level_predictions_cnn(clf, loader)
    return top_k_accuracy(yt, rk, k=1)


def _top1_siam(
    siam: SiameseClassifier, loader: DataLoader
) -> float:
    """
    Top-1 image-level accuracy for Siamese.

    Parameters
    ----------
    siam : SiameseClassifier
        Siamese model.
    loader : DataLoader
        Loader.

    Returns
    -------
    float
        Top-1 accuracy.
    """
    yt, rk = image_level_predictions_siamese(siam, loader)
    return top_k_accuracy(yt, rk, k=1)


def _jpeg_curve_models(
    ncc: NCCBaseline,
    cnn: CNNClassifier,
    siam: SiameseClassifier,
    test: list[tuple[str, int]],
    san_map: dict[str, int],
    denoiser: WienerDenoiser,
    ps: int,
    cfg: dict[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    """
    Evaluate all models across JPEG quality levels.

    Returns
    -------
    dict[str, Any]
        Nested metrics including per-Q accuracies, AURC, RAD.
    """
    levels = list(cfg.get("jpeg_quality_levels", JPEGSweep.QUALITY_LEVELS))
    out: dict[str, dict[str, Any]] = {"NCC": {}, "CNN": {}, "Siamese": {}}
    accs = {"NCC": [], "CNN": [], "Siamese": []}
    for Q in tqdm(levels, desc="JPEG sweep"):
        js = JPEGSweep(int(Q))
        ds = PRNUPatchDataset(
            test, patch_size=ps, transform=js, denoiser=denoiser
        )
        ld = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=prnu_collate,
        )
        a_nc = _top1_nc(ncc, test, san_map, js)
        a_cn = _top1_cnn(cnn, ld)
        a_si = _top1_siam(siam, ld)
        out["NCC"][f"Q{Q}"] = a_nc
        out["CNN"][f"Q{Q}"] = a_cn
        out["Siamese"][f"Q{Q}"] = a_si
        accs["NCC"].append(a_nc)
        accs["CNN"].append(a_cn)
        accs["Siamese"].append(a_si)
    for name in out:
        curve = accs[name]
        clean = curve[0]
        worst = curve[-1]
        out[name]["AURC"] = aurc(curve)
        out[name]["RAD"] = relative_accuracy_drop(clean, worst)
    return out


def _resize_curve_models(
    ncc: NCCBaseline,
    cnn: CNNClassifier,
    siam: SiameseClassifier,
    test: list[tuple[str, int]],
    san_map: dict[str, int],
    denoiser: WienerDenoiser,
    ps: int,
    cfg: dict[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    """
    Evaluate models across resize stress scales.

    Returns
    -------
    dict[str, Any]
        Per-scale top-1 accuracies.
    """
    scales = list(cfg.get("resize_scale_factors", ResizeSweep.SCALE_FACTORS))
    out: dict[str, dict[str, float]] = {"NCC": {}, "CNN": {}, "Siamese": {}}
    for sc in tqdm(scales, desc="Resize sweep"):
        rs = ResizeSweep(float(sc))
        ds = PRNUPatchDataset(
            test, patch_size=ps, transform=rs, denoiser=denoiser
        )
        ld = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=prnu_collate,
        )
        out["NCC"][f"S{sc}"] = _top1_nc(ncc, test, san_map, rs)
        out["CNN"][f"S{sc}"] = _top1_cnn(cnn, ld)
        out["Siamese"][f"S{sc}"] = _top1_siam(siam, ld)
    return out


def _train_ieee_models(
    cfg: dict[str, Any],
    ieee_root: Path,
    device: str,
    epochs_cnn: int,
    epochs_siam: int,
) -> tuple[NCCBaseline, CNNClassifier, SiameseClassifier, WienerDenoiser, int, dict[str, int], list[tuple[str, int]]]:
    """
    Train stack on IEEE SP ``train/`` split; return test list from held-out split.

    Returns
    -------
    tuple
        Models, denoiser, patch size, label mapping, ieee test list ``(path,int)``.
    """
    ieee = IEEESPLoader(ieee_root, seed=int(cfg["seed"]))
    sp = ieee.get_splits()
    sp = apply_train_sampling(sp, cfg)
    mapping = ieee.get_label_mapping()
    denoiser = WienerDenoiser(window_size=int(cfg["wiener_window"]))
    fp_dir = ROOT / "data" / "processed" / "fingerprints_ieee"
    fp_dir.mkdir(parents=True, exist_ok=True)
    est = fingerprint_estimator_from_config(denoiser, cfg)
    by_dev: dict[str, list[str]] = defaultdict(list)
    for p, lab in sp["train"]:
        # reverse lookup label->name
        inv = {v: k for k, v in mapping.items()}
        by_dev[inv[lab]].append(p)
    for dev in tqdm(sorted(by_dev.keys()), desc="IEEE fingerprints"):
        if est.has_cache(dev, fp_dir):
            continue
        fp = est.estimate(by_dev[dev], dev)
        est.save(fp, dev, fp_dir)
    ncc = NCCBaseline(denoiser=WienerDenoiser(window_size=int(cfg["wiener_window"])))
    ncc.fit(fp_dir)
    ps = int(cfg["patch_size"])
    train_ds = PRNUPatchDataset(sp["train"], patch_size=ps, denoiser=denoiser)
    tr_ld = DataLoader(
        train_ds,
        batch_size=int(cfg["cnn_batch_size"]),
        shuffle=True,
        num_workers=2,
        collate_fn=prnu_collate,
    )
    cnn = CNNClassifier(
        len(mapping),
        device=device,
        lr=float(cfg["cnn_lr"]),
        weight_decay=float(cfg["cnn_weight_decay"]),
    )
    for _ in range(epochs_cnn):
        cnn.train_epoch(tr_ld)
    siam = SiameseClassifier(
        num_classes=len(mapping),
        embedding_dim=int(cfg["siamese_embedding_dim"]),
        device=device,
        lr=float(cfg["siamese_lr"]),
        margin=float(cfg["siamese_margin"]),
        mode="triplet",
    )
    s_tr = DataLoader(
        train_ds,
        batch_size=int(cfg["siamese_batch_size"]),
        shuffle=True,
        num_workers=2,
        collate_fn=prnu_collate,
    )
    for _ in range(epochs_siam):
        siam.train_epoch(s_tr)
    siam.fit_centroids(s_tr)
    san_map_ieee = {sanitize_device_id(k): int(mapping[k]) for k in mapping}
    test_list = sp["test"]
    return ncc, cnn, siam, denoiser, ps, san_map_ieee, test_list


def _ieee_jpeg_curve(
    ncc: NCCBaseline,
    cnn: CNNClassifier,
    siam: SiameseClassifier,
    test: list[tuple[str, int]],
    san_map: dict[str, int],
    denoiser: WienerDenoiser,
    ps: int,
    cfg: dict[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    """
    JPEG sweep for IEEE SP models (uses ``san_map`` for NCC string keys).

    Returns
    -------
    dict[str, Any]
        Nested accuracies similar to Dresden sweep.
    """
    levels = list(cfg.get("jpeg_quality_levels", JPEGSweep.QUALITY_LEVELS))
    out: dict[str, dict[str, Any]] = {"NCC": {}, "CNN": {}, "Siamese": {}}
    accs = {"NCC": [], "CNN": [], "Siamese": []}
    for Q in levels:
        js = JPEGSweep(int(Q))
        ds = PRNUPatchDataset(
            test, patch_size=ps, transform=js, denoiser=denoiser
        )
        ld = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=prnu_collate,
        )
        a_nc = _top1_nc(ncc, test, san_map, js)
        a_cn = _top1_cnn(cnn, ld)
        a_si = _top1_siam(siam, ld)
        out["NCC"][f"Q{Q}"] = a_nc
        out["CNN"][f"Q{Q}"] = a_cn
        out["Siamese"][f"Q{Q}"] = a_si
        accs["NCC"].append(a_nc)
        accs["CNN"].append(a_cn)
        accs["Siamese"].append(a_si)
    for name in out:
        curve = accs[name]
        out[name]["AURC"] = aurc(curve)
        out[name]["RAD"] = relative_accuracy_drop(curve[0], curve[-1])
    return out


def _calibration_block(
    cnn: CNNClassifier,
    test_ds: PRNUPatchDataset,
    num_classes: int,
    batch_size: int,
    n_bins: int,
) -> dict[str, Any]:
    """
    Calibration metrics for CNN on a patch dataset.

    Returns
    -------
    dict[str, Any]
        ECE, Brier, reliability bins, selective risk curve.
    """
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=prnu_collate,
    )
    cnn.model.eval()
    confs: list[float] = []
    correct: list[float] = []
    probs_all: list[np.ndarray] = []
    y_all: list[int] = []
    dev = cnn.device
    with torch.no_grad():
        for x, y, _p in tqdm(loader, desc="Calibration"):
            x = x.to(dev)
            y = y.to(dev)
            prob = F.softmax(cnn.model(x), dim=1)
            pred = prob.argmax(dim=1)
            p_max = prob.max(dim=1).values
            confs.extend(p_max.detach().cpu().numpy().tolist())
            correct.extend((pred == y).float().cpu().numpy().tolist())
            probs_all.append(prob.detach().cpu().numpy())
            y_all.extend(y.detach().cpu().numpy().tolist())
    conf_arr = np.asarray(confs, dtype=np.float64)
    corr_arr = np.asarray(correct, dtype=np.float64)
    probs = np.concatenate(probs_all, axis=0)
    y_onehot = np.zeros((probs.shape[0], num_classes), dtype=np.float64)
    for i, yi in enumerate(y_all):
        y_onehot[i, int(yi)] = 1.0
    ece = expected_calibration_error(conf_arr, corr_arr, n_bins=n_bins)
    brier = brier_score(y_onehot, probs)
    rel = reliability_diagram_data(conf_arr, corr_arr, n_bins=n_bins)
    sel = selective_risk_curve(conf_arr, corr_arr)
    return {
        "ECE": {"value": ece},
        "Brier": {"value": brier},
        "reliability_bins": rel,
        "selective_risk": sel,
    }


def main() -> None:
    """Run Group C experiments."""
    parser = argparse.ArgumentParser(description="Group C robustness")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--max-devices", type=int, default=None)
    parser.add_argument("--cnn-epochs", type=int, default=None)
    parser.add_argument("--siamese-epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip-ieee", action="store_true")
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

    san_map = {sanitize_device_id(k): v for k, v in mapping.items()}

    epochs_cnn = int(args.cnn_epochs or cfg["cnn_epochs"])
    epochs_siam = int(args.siamese_epochs or cfg["siamese_epochs"])

    ncc, cnn, siam, denoiser, ps = _train_dresden_stack(
        cfg, splits, mapping, fp_dir, args.device, epochs_cnn, epochs_siam
    )
    test = attach_labels(splits["test"], mapping)

    jpeg_dresden = _jpeg_curve_models(
        ncc,
        cnn,
        siam,
        test,
        san_map,
        denoiser,
        ps,
        cfg,
        int(cfg["cnn_batch_size"]),
    )

    ieee_block: dict[str, Any] = {}
    if not args.skip_ieee:
        ieee_root = Path(cfg["ieee_sp_root"])
        if not ieee_root.is_absolute():
            ieee_root = ROOT / ieee_root
        if (ieee_root / "train").is_dir():
            ncc_i, cnn_i, siam_i, den_i, ps_i, smap_i, test_i = _train_ieee_models(
                cfg, ieee_root, args.device, epochs_cnn, epochs_siam
            )
            ieee_block = _ieee_jpeg_curve(
                ncc_i,
                cnn_i,
                siam_i,
                test_i,
                smap_i,
                den_i,
                ps_i,
                cfg,
                int(cfg["cnn_batch_size"]),
            )
        else:
            ieee_block = {"note": "ieee_sp train/ not found; skipped"}

    resize = _resize_curve_models(
        ncc,
        cnn,
        siam,
        test,
        san_map,
        denoiser,
        ps,
        cfg,
        int(cfg["cnn_batch_size"]),
    )

    test_clean = PRNUPatchDataset(test, patch_size=ps, denoiser=denoiser)
    c3 = _calibration_block(
        cnn,
        test_clean,
        len(mapping),
        int(cfg["cnn_batch_size"]),
        int(cfg["ece_bins"]),
    )

    out = {
        "experiment": "group_C",
        "config": cfg,
        "git_hash": get_git_hash(),
        "C1": {"jpeg_sweep": jpeg_dresden, "ieee_sp_crossdataset": ieee_block},
        "C2": {"resize_sweep": resize},
        "C3": c3,
    }
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "results" / "group_C.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
