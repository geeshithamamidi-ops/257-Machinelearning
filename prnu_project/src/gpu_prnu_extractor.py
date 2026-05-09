"""
GPU-accelerated PRNU residual extraction using PyTorch.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class PRNUImageDataset(Dataset[tuple[torch.Tensor, str, str]]):
    """
    Dataset that loads images as CHW float tensors and preserves metadata.
    """

    def __init__(
        self,
        image_paths: list[str],
        device_ids: list[str],
        target_size: Optional[tuple[int, int]] = None,
    ) -> None:
        if len(image_paths) != len(device_ids):
            raise ValueError("image_paths and device_ids must have the same length")
        self.image_paths = image_paths
        self.device_ids = device_ids
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, str]:
        path = self.image_paths[index]
        dev = self.device_ids[index]
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            if self.target_size is not None:
                rgb = rgb.resize((self.target_size[1], self.target_size[0]), Image.Resampling.BILINEAR)
            arr = np.asarray(rgb, dtype=np.uint8)
            x = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()
        return x, path, dev


def _safe_device_from_path(image_path: str) -> str:
    p = Path(image_path)
    return (p.parent.name or "unknown").replace("/", "_")


def _residual_cache_path_pt(residual_root: str | Path, image_path: str) -> Path:
    import hashlib

    p = Path(image_path)
    device = _safe_device_from_path(image_path)
    h = hashlib.md5(str(p.resolve()).encode("utf-8")).hexdigest()
    return Path(residual_root) / device / f"{h}.pt"


def _residual_cache_path_npy(residual_root: str | Path, image_path: str) -> Path:
    import hashlib

    p = Path(image_path)
    device = _safe_device_from_path(image_path)
    h = hashlib.md5(str(p.resolve()).encode("utf-8")).hexdigest()
    return Path(residual_root) / device / f"{h}.npy"


def _collate_resize(batch: list[tuple[torch.Tensor, str, str]]) -> tuple[torch.Tensor, list[str], list[str]]:
    xs, paths, devs = zip(*batch)
    min_h = min(int(x.shape[1]) for x in xs)
    min_w = min(int(x.shape[2]) for x in xs)
    resized = []
    for x in xs:
        if int(x.shape[1]) != min_h or int(x.shape[2]) != min_w:
            resized.append(
                F.interpolate(x.unsqueeze(0), size=(min_h, min_w), mode="bilinear", align_corners=False).squeeze(0)
            )
        else:
            resized.append(x)
    return torch.stack(resized, dim=0), list(paths), list(devs)


class GPUWienerLikeDenoiser:
    """
    Wiener-like denoiser implemented with conv2d local moments.
    """

    def __init__(
        self,
        window_size: int = 3,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
    ) -> None:
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        self.window_size = int(window_size)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = bool(use_amp)
        k = self.window_size
        kernel = torch.ones((3, 1, k, k), dtype=torch.float32) / float(k * k)
        self.kernel = kernel.to(self.device)
        self.padding = k // 2

    def denoise_batch(self, images: torch.Tensor) -> torch.Tensor:
        x = images.to(self.device, non_blocking=True).float()
        autocast_enabled = self.use_amp and self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, enabled=autocast_enabled):
            local_mean = F.conv2d(x, self.kernel, stride=1, padding=self.padding, groups=3)
            local_sq_mean = F.conv2d(x * x, self.kernel, stride=1, padding=self.padding, groups=3)
            local_var = torch.clamp(local_sq_mean - local_mean * local_mean, min=0.0)
            noise = local_var.mean(dim=(2, 3), keepdim=True)
            gain = torch.clamp(local_var - noise, min=0.0) / (local_var + 1e-6)
            return local_mean + gain * (x - local_mean)

    def residual_batch(self, images: torch.Tensor) -> torch.Tensor:
        return images.to(self.device, non_blocking=True).float() - self.denoise_batch(images)


class GPUResidualExtractor:
    """
    Batched PRNU residual extractor with GPU preprocessing and persistence.
    """

    def __init__(
        self,
        window_size: int = 3,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_amp: bool = True,
        save_dtype: torch.dtype = torch.float16,
        save_format: str = "pt",
        write_legacy_npy: bool = True,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory and self.device.type == "cuda")
        self.save_dtype = save_dtype
        self.save_format = str(save_format).strip().lower()
        if self.save_format not in {"pt", "npy"}:
            raise ValueError("save_format must be one of: pt, npy")
        self.write_legacy_npy = bool(write_legacy_npy)
        self.denoiser = GPUWienerLikeDenoiser(
            window_size=window_size,
            device=self.device,
            use_amp=use_amp,
        )

    def _save_residual(self, residual: torch.Tensor, image_path: str, out_dir: Path) -> Path:
        out_pt = _residual_cache_path_pt(out_dir, image_path)
        out_npy = _residual_cache_path_npy(out_dir, image_path)
        out_pt.parent.mkdir(parents=True, exist_ok=True)
        x = residual.detach().to("cpu").to(self.save_dtype).permute(1, 2, 0).contiguous()

        if self.save_format == "pt":
            torch.save(x, out_pt)
            if self.write_legacy_npy:
                np.save(out_npy, x.numpy())
            return out_pt

        np.save(out_npy, x.numpy())
        return out_npy

    def run(
        self,
        image_paths: list[str],
        device_ids: list[str],
        out_dir: str | Path,
        force: bool = False,
        target_size: Optional[tuple[int, int]] = None,
    ) -> dict[str, int]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        kept_paths: list[str] = []
        kept_devs: list[str] = []
        skipped_cache = 0
        for p, d in zip(image_paths, device_ids):
            pt_path = _residual_cache_path_pt(out, p)
            npy_path = _residual_cache_path_npy(out, p)
            exists = pt_path.is_file() if self.save_format == "pt" else npy_path.is_file()
            if exists and not force:
                skipped_cache += 1
                continue
            kept_paths.append(p)
            kept_devs.append(d)

        # Optional resize (e.g., 128x128) greatly reduces per-batch GPU denoising cost.
        ds = PRNUImageDataset(kept_paths, kept_devs, target_size=target_size)
        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=_collate_resize,
            persistent_workers=self.num_workers > 0,
        )

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()

        t0 = time.perf_counter()
        written = 0
        failed = 0
        for batch_idx, (xb, paths, _devs) in enumerate(loader):
            batch_t0 = time.perf_counter()
            try:
                wb = self.denoiser.residual_batch(xb)
                for i in range(wb.shape[0]):
                    self._save_residual(wb[i], paths[i], out)
                    written += 1
            except Exception:
                failed += len(paths)
            batch_dt = time.perf_counter() - batch_t0
            print(
                f"[gpu-prnu] batch={batch_idx:05d} size={len(paths)} time={batch_dt:.3f}s",
                flush=True,
            )

        if self.device.type == "cuda":
            end_evt.record()
            torch.cuda.synchronize(self.device)
            gpu_ms = start_evt.elapsed_time(end_evt)
            peak_mem_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)
            print(f"[gpu-prnu] gpu_elapsed={gpu_ms/1000.0:.3f}s peak_mem={peak_mem_mb:.1f}MB", flush=True)
            try:
                util = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=2.0,
                ).strip()
                print(f"[gpu-prnu] nvidia_smi={util}", flush=True)
            except Exception:
                pass

        total_dt = time.perf_counter() - t0
        print(f"[gpu-prnu] total_time={total_dt:.3f}s batches={len(loader)}", flush=True)
        return {
            "written": written,
            "failed": failed,
            "skipped_cache": skipped_cache,
            "total_input": len(image_paths),
        }
