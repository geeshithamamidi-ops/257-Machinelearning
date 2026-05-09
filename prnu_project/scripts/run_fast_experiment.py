"""
Run a fast PRNU experiment on a small Dresden subset.

Example:
    python scripts/run_fast_experiment.py --config configs/fast_dev.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from src.data_loader import (
    DresdenLoader,
    PRNUPatchDataset,
    apply_device_image_subset,
    attach_labels,
    build_device_label_map,
    prnu_collate,
)
from src.evaluate import (
    image_level_predictions_cnn,
    image_level_predictions_siamese,
    top_k_accuracy,
)
from src.gpu_prnu_extractor import GPUResidualExtractor
from src.models.cnn_classifier import CNNClassifier
from src.models.siamese_network import SiameseClassifier
from src.prnu_extraction import WienerDenoiser
from src.train import get_git_hash, load_config, seed_everything, train_cnn, train_siamese


def _resolve_path(project_root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (project_root / path)


def _build_subset_splits(cfg: dict[str, Any]) -> tuple[dict[str, list[tuple[str, str]]], dict[str, int]]:
    loader = DresdenLoader(
        cfg["dresden_root"],
        min_images_per_device=int(cfg.get("min_images_per_device", 1)),
        seed=int(cfg.get("seed", 42)),
        train_ratio=float(cfg.get("train_ratio", 0.60)),
        val_ratio=float(cfg.get("val_ratio", 0.20)),
        test_ratio=float(cfg.get("test_ratio", 0.20)),
    )
    splits = loader.get_splits()
    splits = apply_device_image_subset(
        splits=splits,
        num_devices=int(cfg.get("num_devices", 5)),
        images_per_device=int(cfg.get("images_per_device", 100)),
        seed=int(cfg.get("seed", 42)),
    )
    mapping = build_device_label_map(splits["train"] + splits["val"] + splits["test"])
    return splits, mapping


def _log_subset_stats(splits: dict[str, list[tuple[str, str]]], cfg: dict[str, Any]) -> None:
    all_samples = splits["train"] + splits["val"] + splits["test"]
    devs = sorted({d for _, d in all_samples})
    print(
        f"[fast-dev] num_devices={len(devs)} configured_num_devices={int(cfg.get('num_devices', 5))}",
        flush=True,
    )
    print(
        "[fast-dev] split_sizes="
        f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}",
        flush=True,
    )
    by_dev: dict[str, int] = {}
    for _p, d in all_samples:
        by_dev[d] = by_dev.get(d, 0) + 1
    shown = ", ".join(f"{k}:{by_dev[k]}" for k in sorted(by_dev)[:10])
    print(
        f"[fast-dev] images_per_device_target={int(cfg.get('images_per_device', 100))} "
        f"actual_counts=[{shown}]",
        flush=True,
    )
    print(f"[fast-dev] total_samples={len(all_samples)}", flush=True)


def _experiment_slug() -> str:
    """Unique folder name so Colab runs never overwrite prior artifacts."""
    return f"experiment_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}"


def _save_experiment_artifacts(
    project_root: Path,
    cfg: dict[str, Any],
    cnn: CNNClassifier,
    siam: SiameseClassifier,
    cnn_history: dict[str, Any],
    siamese_history: dict[str, Any],
    metrics_extra: dict[str, Any],
    config_source_path: Path,
    wall_start: datetime,
    wall_end: datetime,
    global_start: float,
) -> dict[str, str]:
    """
    Persist weights + metrics under timestamped outputs/ and checkpoints/.

    Duplicating weights under checkpoints/ keeps a stable layout for downstream
    scripts while outputs/ holds the full experiment record (metrics + config).
    """
    slug = _experiment_slug()
    out_root = project_root / "outputs" / slug
    ckpt_root = project_root / "checkpoints" / slug
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    cnn_out = out_root / "cnn_model.pth"
    siam_out = out_root / "siamese_model.pth"
    cnn.save(cnn_out)
    siam.save(siam_out)
    # Same filenames under checkpoints/ (timestamped parent avoids clobbering).
    cnn_ckpt = ckpt_root / "cnn_model.pth"
    siam_ckpt = ckpt_root / "siamese_model.pth"
    shutil.copy2(cnn_out, cnn_ckpt)
    shutil.copy2(siam_out, siam_ckpt)

    metrics_path = out_root / "metrics.json"
    payload = {
        "experiment_slug": slug,
        "git_hash": get_git_hash(),
        "cnn": {
            "train_loss_per_epoch": cnn_history.get("train_loss", []),
            "val_accuracy_per_epoch": cnn_history.get("val_acc", []),
            "val_loss_per_epoch": cnn_history.get("val_loss", []),
        },
        "siamese": {
            "train_loss_per_epoch": siamese_history.get("train_loss", []),
        },
        **metrics_extra,
        "runtime": {
            "wall_start_iso": wall_start.isoformat(),
            "wall_end_iso": wall_end.isoformat(),
            "total_wall_s": time.perf_counter() - global_start,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    cfg_dump = out_root / "config.yaml"
    with open(cfg_dump, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    src_meta = out_root / "config_source.txt"
    src_meta.write_text(str(config_source_path.resolve()), encoding="utf-8")

    drive_root = str(cfg.get("experiment_drive_root", "") or "").strip() or os.environ.get(
        "PRNU_EXPERIMENT_DRIVE_ROOT", ""
    ).strip()
    drive_copy: str | None = None
    if drive_root:
        dest = Path(drive_root).expanduser() / slug
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest = Path(drive_root).expanduser() / f"{slug}_copy"
        shutil.copytree(out_root, dest, dirs_exist_ok=False)
        drive_copy = str(dest.resolve())

    return {
        "outputs_dir": str(out_root.resolve()),
        "checkpoints_dir": str(ckpt_root.resolve()),
        "metrics_json": str(metrics_path.resolve()),
        "config_yaml": str(cfg_dump.resolve()),
        "drive_copy": drive_copy or "",
    }


def _maybe_precompute_subset(
    cfg: dict[str, Any],
    splits: dict[str, list[tuple[str, str]]],
    residual_root: Path,
) -> None:
    all_samples = splits["train"] + splits["val"] + splits["test"]
    image_paths = [p for p, _ in all_samples]
    dev_ids = [d for _, d in all_samples]
    extractor = GPUResidualExtractor(
        window_size=int(cfg.get("wiener_window", 3)),
        batch_size=int(cfg.get("batch_size", 32)),
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        use_amp=bool(cfg.get("use_amp", True)),
        save_format="pt",
        write_legacy_npy=True,
        multiprocessing_context="spawn",
        prefetch_factor=1,
        persistent_workers=True,
        worker_sharing_strategy="file_system",
    )
    t0 = time.perf_counter()
    stats = extractor.run(
        image_paths=image_paths,
        device_ids=dev_ids,
        out_dir=residual_root,
        force=False,
        target_size=(
            int(cfg.get("image_size", 128)),
            int(cfg.get("image_size", 128)),
        ),
    )
    dt = time.perf_counter() - t0
    print(
        "[fast-dev] subset precompute "
        f"written={stats['written']} failed={stats['failed']} skipped={stats['skipped_cache']} "
        f"time_s={dt:.2f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fast PRNU dev experiment")
    parser.add_argument("--config", type=str, default="configs/fast_dev.yaml")
    parser.add_argument(
        "--skip_precompute",
        action="store_true",
        help="Compute PRNU on-the-fly for selected subset (do not precompute cache).",
    )
    args = parser.parse_args()

    wall_start = datetime.now()
    global_start = time.perf_counter()
    project_root = Path(__file__).resolve().parents[1]
    config_source_path = _resolve_path(project_root, args.config)
    cfg = load_config(config_source_path)
    cfg["dresden_root"] = str(_resolve_path(project_root, str(cfg["dresden_root"])))
    residual_root = _resolve_path(
        project_root, str(cfg.get("residual_cache_dir", "data/processed/residuals_fast"))
    )
    residual_root.mkdir(parents=True, exist_ok=True)

    seed_everything(int(cfg.get("seed", 42)))
    run_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[fast-dev] device={run_device}", flush=True)
    if run_device == "cuda":
        print(f"[fast-dev] gpu={torch.cuda.get_device_name(0)}", flush=True)

    # fast_dev_mode consolidates all speed-focused knobs in one switch while
    # preserving backwards-compatible defaults when disabled.
    if bool(cfg.get("fast_dev_mode", False)):
        cfg["num_devices"] = min(int(cfg.get("num_devices", 5)), int(cfg.get("fast_num_devices", 5)))
        cfg["images_per_device"] = min(
            int(cfg.get("images_per_device", 200)),
            int(cfg.get("fast_images_per_device", 200)),
        )
        cfg["epochs"] = int(cfg.get("fast_epochs", 1))
        cfg["cnn_epochs"] = int(cfg.get("fast_epochs", 1))
        cfg["siamese_epochs"] = int(cfg.get("fast_epochs", 1))
        cfg["skip_expensive_preprocessing"] = bool(cfg.get("skip_expensive_preprocessing", True))
        print(
            "[fast-dev] fast_dev_mode enabled: reduced subset + 1 epoch + optional preprocessing skips",
            flush=True,
        )

    splits, mapping = _build_subset_splits(cfg)
    if not splits["train"] or not mapping:
        raise RuntimeError("No samples selected. Check dresden_root/num_devices/images_per_device.")
    _log_subset_stats(splits, cfg)

    skip_precompute = bool(args.skip_precompute or cfg.get("skip_expensive_preprocessing", False))
    if not skip_precompute:
        _maybe_precompute_subset(cfg, splits, residual_root)
        residual_arg: str | None = str(residual_root)
    else:
        print("[fast-dev] skip_precompute=True -> using on-the-fly residual extraction", flush=True)
        residual_arg = None

    # In skip_precompute mode, residual extraction happens inside Dataset workers.
    # CUDA in forked workers is unstable in Colab, so keep denoising on CPU there.
    denoiser_torch_device = run_device
    if skip_precompute and int(cfg.get("num_workers", 8)) > 0:
        denoiser_torch_device = "cpu"
        print(
            "[fast-dev] skip_precompute with num_workers>0 -> using CPU denoiser in workers "
            "to avoid CUDA multiprocessing errors.",
            flush=True,
        )
    denoiser = WienerDenoiser(
        window_size=int(cfg.get("wiener_window", 3)),
        backend="torch",
        torch_device=denoiser_torch_device,
    )
    # Resize to 128x128 in fast mode reduces residual extraction + patch indexing cost.
    image_resize = int(cfg.get("image_size", 128))
    resize_hw = (image_resize, image_resize)
    patch_size = int(cfg.get("patch_size", 64))
    max_patches_per_image = int(cfg.get("max_patches_per_image", 10))
    train_ds = PRNUPatchDataset(
        attach_labels(splits["train"], mapping),
        patch_size=patch_size,
        resize_hw=resize_hw,
        denoiser=denoiser,
        max_patches_per_image=max_patches_per_image,
        residual_root=residual_arg,
    )
    val_ds = PRNUPatchDataset(
        attach_labels(splits["val"], mapping),
        patch_size=patch_size,
        resize_hw=resize_hw,
        denoiser=denoiser,
        max_patches_per_image=max_patches_per_image,
        residual_root=residual_arg,
    )
    test_ds = PRNUPatchDataset(
        attach_labels(splits["test"], mapping),
        patch_size=patch_size,
        resize_hw=resize_hw,
        denoiser=denoiser,
        max_patches_per_image=max_patches_per_image,
        residual_root=residual_arg,
    )
    print(
        f"[fast-dev] patch_dataset_sizes train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"patch_size={patch_size} max_patches_per_image={max_patches_per_image}",
        flush=True,
    )

    loader_kwargs: dict[str, Any] = {
        # More workers overlap CPU-side patch loading with GPU compute.
        "num_workers": int(cfg.get("num_workers", 8)),
        "pin_memory": bool(cfg.get("pin_memory", True)) and run_device == "cuda",
        "collate_fn": prnu_collate,
    }
    if int(loader_kwargs["num_workers"]) > 0:
        # Persistent workers amortize startup overhead in Colab notebook loops.
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
        loader_kwargs["multiprocessing_context"] = "spawn"
    batch_size = int(cfg.get("batch_size", 32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    print(
        f"[fast-dev] estimated_train_batches={len(train_loader)} "
        f"batch_size={batch_size} num_workers={loader_kwargs['num_workers']}",
        flush=True,
    )

    num_classes = len(mapping)
    epochs = int(cfg.get("epochs", 5))
    cnn = CNNClassifier(
        num_classes=num_classes,
        device=run_device,
        lr=float(cfg.get("cnn_lr", 1e-4)),
        weight_decay=float(cfg.get("cnn_weight_decay", 1e-4)),
        use_amp=bool(cfg.get("use_amp", True)),
        grad_accum_steps=int(cfg.get("grad_accum_steps", 1)),
    )
    t0 = time.perf_counter()
    cnn_history = train_cnn(cnn, train_loader, val_loader, epochs=epochs)
    cnn_train_s = time.perf_counter() - t0
    print(f"[fast-dev] cnn_total_train_time_s={cnn_train_s:.2f}", flush=True)

    y_true, y_ranked = image_level_predictions_cnn(cnn, test_loader)
    cnn_top1 = top_k_accuracy(y_true, y_ranked, k=1)
    print(f"[fast-dev] cnn_test_top1={cnn_top1:.4f}", flush=True)

    siam = SiameseClassifier(
        num_classes=num_classes,
        embedding_dim=int(cfg.get("siamese_embedding_dim", 128)),
        device=run_device,
        lr=float(cfg.get("siamese_lr", 1e-4)),
        margin=float(cfg.get("siamese_margin", 1.0)),
        mode="triplet",
        use_amp=bool(cfg.get("use_amp", True)),
        grad_accum_steps=int(cfg.get("grad_accum_steps", 1)),
    )
    t1 = time.perf_counter()
    siamese_history = train_siamese(siam, train_loader, epochs=epochs)
    siam.fit_centroids(train_loader)
    siamese_train_s = time.perf_counter() - t1
    print(f"[fast-dev] siamese_total_train_time_s={siamese_train_s:.2f}", flush=True)

    siam_val_top1 = None
    if len(val_ds) > 0:
        yt_sv, rk_sv = image_level_predictions_siamese(siam, val_loader)
        siam_val_top1 = float(top_k_accuracy(yt_sv, rk_sv, k=1))
    yt_st, rk_st = image_level_predictions_siamese(siam, test_loader)
    siam_test_top1 = float(top_k_accuracy(yt_st, rk_st, k=1))
    print(f"[fast-dev] siamese_test_top1={siam_test_top1:.4f}", flush=True)

    wall_end = datetime.now()
    print(f"[fast-dev] training_start={wall_start.isoformat()} training_end={wall_end.isoformat()}", flush=True)
    print(f"[fast-dev] total_wall_time_s={time.perf_counter() - global_start:.2f}", flush=True)

    metrics_extra: dict[str, Any] = {
        "cnn_test_top1": float(cnn_top1),
        "siamese_val_top1": siam_val_top1,
        "siamese_test_top1": siam_test_top1,
        "dataset": {
            "num_classes": num_classes,
            "train_images": len(splits["train"]),
            "val_images": len(splits["val"]),
            "test_images": len(splits["test"]),
            "train_patches": len(train_ds),
            "val_patches": len(val_ds),
            "test_patches": len(test_ds),
            "estimated_train_batches": len(train_loader),
            "batch_size": batch_size,
        },
        "training_times_s": {
            "cnn": cnn_train_s,
            "siamese": siamese_train_s,
        },
        "config_path": str(config_source_path.resolve()),
    }
    paths = _save_experiment_artifacts(
        project_root=project_root,
        cfg=cfg,
        cnn=cnn,
        siam=siam,
        cnn_history=cnn_history,
        siamese_history=siamese_history,
        metrics_extra=metrics_extra,
        config_source_path=config_source_path,
        wall_start=wall_start,
        wall_end=wall_end,
        global_start=global_start,
    )
    print("[fast-dev] saved experiment artifacts:", flush=True)
    for k, v in paths.items():
        if v:
            print(f"  {k}: {v}", flush=True)
    print("[fast-dev] experiment complete", flush=True)


if __name__ == "__main__":
    main()
