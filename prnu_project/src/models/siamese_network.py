"""
Siamese encoder with contrastive / triplet objectives and centroid inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class SiameseEncoder(nn.Module):
    """
    Shared ResNet-18 trunk (1-channel) producing L2-normalized 256-D embeddings.
    """

    def __init__(self, embedding_dim: int = 256) -> None:
        """
        Parameters
        ----------
        embedding_dim : int
            Output embedding size before normalization.
        """
        super().__init__()
        from torchvision.models import resnet18

        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        in_f = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m
        self.proj = nn.Linear(in_f, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed a batch of 1-channel patches.

        Parameters
        ----------
        x : torch.Tensor
            Nx1xHxW.

        Returns
        -------
        torch.Tensor
            N x embedding_dim L2-normalized.
        """
        h = self.backbone(x)
        z = self.proj(h)
        return F.normalize(z, dim=1)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss: L = y*d^2 + (1-y)*max(margin - d, 0)^2 with Euclidean distance.
    """

    def __init__(self, margin: float = 1.0) -> None:
        """
        Parameters
        ----------
        margin : float
            Inter-class margin for dissimilar pairs.
        """
        super().__init__()
        self.margin = float(margin)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z1, z2 : torch.Tensor
            Embeddings (N, D).
        y : torch.Tensor
            Binary same/different labels (1 same class).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        d = F.pairwise_distance(z1, z2, p=2)
        yf = y.float()
        pos = yf * (d**2)
        neg = (1.0 - yf) * (F.relu(self.margin - d) ** 2)
        return (pos + neg).mean()


class TripletLoss(nn.Module):
    """Standard triplet loss with optional semi-hard mining within batch."""

    def __init__(self, margin: float = 0.5) -> None:
        """
        Parameters
        ----------
        margin : float
            Triplet margin.
        """
        super().__init__()
        self.margin = float(margin)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        anchor, positive, negative : torch.Tensor
            Embeddings (N, D).

        Returns
        -------
        torch.Tensor
            Scalar triplet loss.
        """
        da = F.pairwise_distance(anchor, positive, p=2)
        dn = F.pairwise_distance(anchor, negative, p=2)
        return F.relu(da - dn + self.margin).mean()


def _semi_hard_triplets(
    z: torch.Tensor, labels: torch.Tensor, margin: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """
    Build semi-hard triplets from an embedding batch.

    Parameters
    ----------
    z : torch.Tensor
        N x D embeddings.
    labels : torch.Tensor
        Long tensor of shape (N,).
    margin : float
        Triplet margin for semi-hard condition.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ``(anchor, positive, negative)`` or None if too few samples.
    """
    if z.size(0) < 3:
        return None
    dist = torch.cdist(z, z, p=2)
    anchors, positives, negatives = [], [], []
    labels = labels.cpu().numpy()
    dist_np = dist.detach().cpu().numpy()
    n = z.size(0)
    for i in range(n):
        pos_idx = [j for j in range(n) if j != i and labels[j] == labels[i]]
        neg_idx = [j for j in range(n) if labels[j] != labels[i]]
        if not pos_idx or not neg_idx:
            continue
        for p in pos_idx:
            di_p = dist_np[i, p]
            for ng in neg_idx:
                di_n = dist_np[i, ng]
                if di_n > di_p and di_n < di_p + margin:
                    anchors.append(i)
                    positives.append(p)
                    negatives.append(ng)
                    break
            else:
                ng = neg_idx[int(np.argmin(dist_np[i, neg_idx]))]
                anchors.append(i)
                positives.append(p)
                negatives.append(ng)
            break
    if not anchors:
        return None
    a = z[torch.tensor(anchors, device=z.device)]
    p = z[torch.tensor(positives, device=z.device)]
    ng = z[torch.tensor(negatives, device=z.device)]
    return a, p, ng


class SiameseClassifier:
    """
    Train encoder with contrastive or triplet loss; infer via nearest class centroid.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 256,
        device: str | torch.device = "cuda",
        lr: float = 1e-4,
        margin: float = 1.0,
        mode: str = "triplet",
        use_amp: bool = False,
        grad_accum_steps: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        num_classes : int
            Number of devices (for reference; centroids use observed labels).
        embedding_dim : int
            Embedding dimensionality.
        device : str | torch.device
            Torch device.
        lr : float
            Adam learning rate.
        margin : float
            Margin for contrastive / triplet.
        mode : str
            ``\"contrastive\"`` or ``\"triplet\"``.
        """
        req = str(device).strip().lower()
        if req in {"", "auto"}:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif req.startswith("cuda") and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}", flush=True)
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.margin = float(margin)
        self.mode = mode
        self.encoder = SiameseEncoder(embedding_dim).to(self.device)
        print(next(self.encoder.parameters()).device, flush=True)
        self.opt = torch.optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=1e-4)
        self.contrastive = ContrastiveLoss(margin=margin)
        self.triplet = TripletLoss(margin=min(0.5, margin))
        self.use_amp = bool(use_amp) and self.device.type == "cuda"
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.last_timing: dict[str, float] = {"data_time_s": 0.0, "step_time_s": 0.0}
        self._centroids: dict[int, torch.Tensor] = {}
        self._label_list: list[int] = []

    def train_epoch(self, loader: DataLoader) -> float:
        """
        Train one epoch (triplet semi-hard if possible, else contrastive pairs).

        Parameters
        ----------
        loader : DataLoader
            Patch loader returning ``(x, y, path)``.

        Returns
        -------
        float
            Mean loss.
        """
        self.encoder.train()
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
                z = self.encoder(x)
            loss_t: Optional[torch.Tensor] = None
            if self.mode == "triplet":
                trip = _semi_hard_triplets(z, y, self.margin)
                if trip is not None:
                    a, p, ng = trip
                    loss_t = self.triplet(a, p, ng)
            if loss_t is None:
                # contrastive pairs: shuffle for negatives
                perm = torch.randperm(z.size(0), device=z.device)
                z2 = z[perm]
                y2 = y[perm]
                same = (y == y2).float()
                loss_t = self.contrastive(z, z2, same)
            loss_scaled = loss_t / self.grad_accum_steps
            self.scaler.scale(loss_scaled).backward()
            if bi % self.grad_accum_steps == 0:
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)
            total += float(loss_t.item()) * x.size(0)
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
    def fit_centroids(self, loader: DataLoader) -> None:
        """
        Compute per-class mean embedding (L2-normalized) from training patches.

        Parameters
        ----------
        loader : DataLoader
            Training patch loader.

        Returns
        -------
        None
        """
        self.encoder.eval()
        sums: dict[int, torch.Tensor] = {}
        counts: dict[int, int] = {}
        for batch in loader:
            x = batch[0].to(self.device, non_blocking=True)
            y = batch[1].to(self.device, non_blocking=True)
            z = self.encoder(x)
            for lab in y.unique():
                m = lab.item()
                sel = z[y == lab]
                sums[m] = sums.get(m, 0) + sel.sum(dim=0)
                counts[m] = counts.get(m, 0) + int(sel.size(0))
        self._centroids = {
            k: F.normalize(sums[k] / max(1, counts[k]), dim=0)
            for k in sums
        }
        self._label_list = sorted(self._centroids.keys())

    @torch.no_grad()
    def predict(self, residual_patch: np.ndarray) -> list[tuple[str, float]]:
        """
        Rank integer-labeled devices by Euclidean distance to centroids.

        Parameters
        ----------
        residual_patch : np.ndarray
            HxW or HxWx1 float patch.

        Returns
        -------
        list[tuple[str, float]]
            ``(str(label_int), distance)`` sorted ascending (best first).
        """
        self.encoder.eval()
        if residual_patch.ndim == 2:
            x = torch.from_numpy(residual_patch)[None, None, ...].float()
        else:
            g = residual_patch.mean(axis=2)
            x = torch.from_numpy(g)[None, None, ...].float()
        x = x.to(self.device)
        z = self.encoder(x)
        dists: list[tuple[str, float]] = []
        for lab in self._label_list:
            c = self._centroids[lab]
            d = float(torch.norm(z - c[None, :], dim=1).item())
            dists.append((str(lab), d))
        dists.sort(key=lambda t: t[1])
        return dists

    def save(self, path: str | Path) -> None:
        """
        Save encoder weights and centroids.

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
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "centroids": {k: v.cpu() for k, v in self._centroids.items()},
                "label_list": self._label_list,
            },
            p,
        )

    def load(self, path: str | Path) -> None:
        """
        Load encoder and centroids.

        Parameters
        ----------
        path : str | Path
            File written by ``save``.

        Returns
        -------
        None
        """
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self._centroids = {int(k): v.to(self.device) for k, v in ckpt["centroids"].items()}
        self._label_list = [int(x) for x in ckpt["label_list"]]
