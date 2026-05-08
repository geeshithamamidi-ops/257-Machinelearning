"""
Precompute PRNU residuals for Dresden images and persist to disk.

Example:
    python scripts/precompute_prnu.py --config configs/default.yaml --split train
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.data_loader import DresdenLoader, residual_cache_path
from src.prnu_extraction import WienerDenoiser
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
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(_resolve_cfg_path(project_root, args.config))
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    denoiser = WienerDenoiser(
        window_size=int(cfg.get("wiener_window", 3)),
        backend=str(cfg.get("wiener_backend", "scipy")),
        torch_device=str(cfg.get("wiener_torch_device", "cpu")),
    )

    samples = _iter_target_samples(cfg, args.split, args.max_devices)
    by_device: dict[str, list[str]] = defaultdict(list)
    for p, d in samples:
        by_device[d].append(p)
    for d in by_device:
        by_device[d] = sorted(by_device[d])
        if args.max_images_per_device is not None and args.max_images_per_device > 0:
            by_device[d] = by_device[d][: args.max_images_per_device]

    total = sum(len(v) for v in by_device.values())
    done, skipped = 0, 0
    pbar = tqdm(total=total, desc=f"PRNU residuals ({args.split})")
    for dev in sorted(by_device):
        for img_path in by_device[dev]:
            out_path = residual_cache_path(out_dir, img_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and not args.force:
                skipped += 1
                pbar.update(1)
                continue
            try:
                from src.preprocessing import load_image_rgb_float

                rgb = load_image_rgb_float(img_path)
                res = denoiser.residual(rgb)
                np.save(out_path, res.astype(np.float16))
                done += 1
            except Exception:
                skipped += 1
            pbar.update(1)
    pbar.close()
    print(
        f"Residual precompute complete. written={done} skipped={skipped} "
        f"out_dir={out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
