"""
Residual CNN classifier (ResNet-18 adapted for 1-channel PRNU patches).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class ResidualCNN(nn.Module):
    """
    ResNet-18 modified for single-channel 128x128 PRNU residual patches.

    - First conv: 1 input channel, 3x3, stride 1, padding 1
    - Removes the initial max-pool (spatial size is small)
    - No ImageNet pretraining
    """

    def __init__(self, num_classes: int) -> None:
        """
        Parameters
        ----------
        num_classes : int
            Number of device classes.
        """
        super().__init__()
        from torchvision.models import resnet18

        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        self.backbone = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward logits.

        Parameters
        ----------
        x : torch.Tensor
            Nx1xHxW tensor.

        Returns
        -------
        torch.Tensor
            N x num_classes logits.
        """
        return self.backbone(x)


class CNNClassifier:
    """Training wrapper for ``ResidualCNN``."""

    def __init__(
        self,
        num_classes: int,
        device: str | torch.device = "cuda",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        use_amp: bool = False,
        grad_accum_steps: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        num_classes : int
            Class count.
        device : str | torch.device
            Torch device.
        lr : float
            Adam learning rate.
        weight_decay : float
            L2 penalty.
        """
        req = str(device).strip().lower()
        if req in {"", "auto"}:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif req.startswith("cuda") and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}", flush=True)
        self.model = ResidualCNN(num_classes).to(self.device)
        print(next(self.model.parameters()).device, flush=True)
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.use_amp = bool(use_amp) and self.device.type == "cuda"
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.last_timing: dict[str, float] = {"data_time_s": 0.0, "step_time_s": 0.0}

    def train_epoch(self, loader: DataLoader) -> float:
        """
        Run one training epoch.

        Parameters
        ----------
        loader : DataLoader
            Yields ``(batch_x, batch_y, ...)``.

        Returns
        -------
        float
            Mean loss.
        """
        self.model.train()
        total, n = 0.0, 0
        data_time, step_time = 0.0, 0.0
        t_prev = time.perf_counter()
        self.opt.zero_grad(set_to_none=True)
        bi = 0
        for bi, batch in enumerate(loader, start=1):
            t_now = time.perf_counter()
            data_time += t_now - t_prev
            x = batch[0].to(self.device, non_blocking=True)
            y = batch[1].to(self.device, non_blocking=True)
            t_step = time.perf_counter()
            amp_ctx = (
                torch.cuda.amp.autocast()
                if self.use_amp
                else nullcontext()
            )
            with amp_ctx:
                logits = self.model(x)
                loss = self.loss_fn(logits, y)
                loss_scaled = loss / self.grad_accum_steps
            self.scaler.scale(loss_scaled).backward()
            if bi % self.grad_accum_steps == 0:
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)
            total += float(loss.item()) * x.size(0)
            n += x.size(0)
            step_time += time.perf_counter() - t_step
            t_prev = time.perf_counter()
        if n > 0 and bi > 0 and (bi % self.grad_accum_steps) != 0:
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
        denom = max(1, len(loader))
        self.last_timing = {
            "data_time_s": data_time / denom,
            "step_time_s": step_time / denom,
        }
        return total / max(1, n)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """
        Evaluate accuracy on a loader.

        Parameters
        ----------
        loader : DataLoader
            Patch loader.

        Returns
        -------
        dict[str, float]
            ``loss`` and ``accuracy`` keys.
        """
        self.model.eval()
        total_loss, correct, seen = 0.0, 0, 0
        for batch in loader:
            x = batch[0].to(self.device, non_blocking=True)
            y = batch[1].to(self.device, non_blocking=True)
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            seen += x.size(0)
            total_loss += float(loss.item()) * x.size(0)
        return {
            "loss": total_loss / max(1, seen),
            "accuracy": correct / max(1, seen),
        }

    def save(self, path: str | Path) -> None:
        """
        Save model weights.

        Parameters
        ----------
        path : str | Path
            Destination ``.pt`` path.

        Returns
        -------
        None
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), p)

    def load(self, path: str | Path) -> None:
        """
        Load model weights.

        Parameters
        ----------
        path : str | Path
            ``.pt`` file from ``save``.

        Returns
        -------
        None
        """
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
