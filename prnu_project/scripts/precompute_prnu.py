"""
Precompute PRNU residuals for Dresden images and persist to disk.

Example:
    python scripts/precompute_prnu.py --config configs/default.yaml --split train
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.data_loader import DresdenLoader
from src.gpu_prnu_extractor import GPUResidualExtractor
from src.train import load_config


def _resolve_cfg_path(project_root: Path, cfg_arg: str) -> Path:
    p = Path(cfg_arg)
    return p if p.is_absolute() else (project_root / p)


def _iter_target_samples(
    cfg: dict[str, Any],
    split: str,
    max_devices: int | None,
) -> list[tuple[str, str]]:
    loader = DresdenLoader(
        cfg["dresden_root"],
        min_images_per_device=int(cfg.get("min_images_per_device", 50)),
        seed=int(cfg.get("seed", 42)),
        train_ratio=float(cfg.get("train_ratio", 0.60)),
        val_ratio=float(cfg.get("val_ratio", 0.20)),
        test_ratio=float(cfg.get("test_ratio", 0.20)),
    )
    splits = loader.get_splits()
    if split == "all":
        out = splits["train"] + splits["val"] + splits["test"]
    else:
        out = list(splits[split])
    if max_devices is not None and max_devices > 0:
        keep = {d for _, d in out}
        keep_sorted = sorted(keep)[:max_devices]
        keep_set = set(keep_sorted)
        out = [(p, d) for p, d in out if d in keep_set]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute Dresden PRNU residual cache")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
    parser.add_argument("--out-dir", type=str, default="data/processed/residuals")
    parser.add_argument("--max-devices", type=int, default=None)
    parser.add_argument("--max-images-per-device", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Recompute even if cached exists.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-format", choices=["pt", "npy"], default="pt")
    parser.add_argument(
        "--no-legacy-npy",
        action="store_true",
        help="If set, do not write compatibility .npy files when saving .pt.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(_resolve_cfg_path(project_root, args.config))
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = _iter_target_samples(cfg, args.split, args.max_devices)
    samples = sorted(samples, key=lambda t: (t[1], t[0]))
    if args.max_images_per_device is not None and args.max_images_per_device > 0:
        per_dev: dict[str, int] = {}
        limited: list[tuple[str, str]] = []
        for p, d in samples:
            cnt = per_dev.get(d, 0)
            if cnt >= args.max_images_per_device:
                continue
            limited.append((p, d))
            per_dev[d] = cnt + 1
        samples = limited
    image_paths = [p for p, _ in samples]
    device_ids = [d for _, d in samples]

    extractor = GPUResidualExtractor(
        window_size=int(cfg.get("wiener_window", 3)),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(cfg.get("pin_memory", True)),
        use_amp=bool(cfg.get("use_amp", True)),
        save_format=args.save_format,
        write_legacy_npy=not bool(args.no_legacy_npy),
    )
    stats = extractor.run(
        image_paths=image_paths,
        device_ids=device_ids,
        out_dir=out_dir,
        force=bool(args.force),
    )
    print(
        f"Residual precompute complete. written={stats['written']} "
        f"failed={stats['failed']} skipped={stats['skipped_cache']} "
        f"out_dir={out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
