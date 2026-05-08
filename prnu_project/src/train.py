"""
High-level training helpers and configuration loading.
"""

from __future__ import annotations

import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.cnn_classifier import CNNClassifier
from src.models.siamese_network import SiameseClassifier


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path : str | Path
        Path to ``.yaml`` / ``.yml``.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_git_hash() -> str:
    """
    Return short git SHA if available, else ``\"unknown\"``.

    Returns
    -------
    str
        Revision string for reproducibility logging.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=Path(__file__).resolve().parents[1]
        )
        return out.decode().strip()[:12]
    except Exception:
        return "unknown"


def seed_everything(seed: int) -> None:
    """
    Seed Python, NumPy, and Torch RNGs.

    Parameters
    ----------
    seed : int
        Integer seed.

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_cnn(
    clf: CNNClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    epochs: int,
) -> dict[str, Any]:
    """
    Train ``CNNClassifier`` for ``epochs`` with tqdm progress.

    Parameters
    ----------
    clf : CNNClassifier
        Wrapper with model/optimizer.
    train_loader : DataLoader
        Training patches.
    val_loader : DataLoader | None
        Optional validation loader.
    epochs : int
        Number of passes.

    Returns
    -------
    dict[str, Any]
        Training history with last losses/accuracy.
    """
    history: dict[str, Any] = {"train_loss": [], "val_acc": []}
    for ep in range(epochs):
        t0 = time.perf_counter()
        loss = clf.train_epoch(train_loader)
        history["train_loss"].append(loss)
        if val_loader is not None:
            ev = clf.evaluate(val_loader)
            history["val_acc"].append(ev["accuracy"])
        dt = time.perf_counter() - t0
        timing = getattr(clf, "last_timing", {})
        gpu_mem = 0.0
        if getattr(clf, "device", torch.device("cpu")).type == "cuda":
            gpu_mem = torch.cuda.max_memory_allocated(clf.device) / (1024**3)
            torch.cuda.reset_peak_memory_stats(clf.device)
        tqdm.write(
            f"[CNN] epoch {ep+1}/{epochs} train_loss={loss:.4f} "
            f"epoch_s={dt:.2f} data_s={timing.get('data_time_s', 0.0):.4f} "
            f"step_s={timing.get('step_time_s', 0.0):.4f} gpu_mem_gb={gpu_mem:.2f}"
        )
    return history


def train_siamese(
    clf: SiameseClassifier,
    train_loader: DataLoader,
    epochs: int,
) -> dict[str, Any]:
    """
    Train ``SiameseClassifier`` encoder.

    Parameters
    ----------
    clf : SiameseClassifier
        Siamese wrapper.
    train_loader : DataLoader
        Training patches.
    epochs : int
        Epoch count.

    Returns
    -------
    dict[str, Any]
        History dict with ``train_loss`` list.
    """
    history: dict[str, Any] = {"train_loss": []}
    for ep in range(epochs):
        t0 = time.perf_counter()
        loss = clf.train_epoch(train_loader)
        history["train_loss"].append(loss)
        dt = time.perf_counter() - t0
        timing = getattr(clf, "last_timing", {})
        gpu_mem = 0.0
        if getattr(clf, "device", torch.device("cpu")).type == "cuda":
            gpu_mem = torch.cuda.max_memory_allocated(clf.device) / (1024**3)
            torch.cuda.reset_peak_memory_stats(clf.device)
        tqdm.write(
            f"[Siamese] epoch {ep+1}/{epochs} train_loss={loss:.4f} "
            f"epoch_s={dt:.2f} data_s={timing.get('data_time_s', 0.0):.4f} "
            f"step_s={timing.get('step_time_s', 0.0):.4f} gpu_mem_gb={gpu_mem:.2f}"
        )
    return history
