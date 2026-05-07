"""
Evaluation metrics for ranked predictions and robustness curves.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.models.cnn_classifier import CNNClassifier
from src.models.siamese_network import SiameseClassifier
from src.preprocessing import load_image_rgb_float, sanitize_device_id


def top_k_accuracy(
    y_true: Sequence[int], y_pred_ranked: Sequence[Sequence[int]], k: int = 1
) -> float:
    """
    Compute top-k accuracy from ranked class index lists.

    Parameters
    ----------
    y_true : Sequence[int]
        Ground-truth class indices.
    y_pred_ranked : Sequence[Sequence[int]]
        For each sample, ranking of class indices (best first).
    k : int
        k for top-k.

    Returns
    -------
    float
        Fraction of samples where true label appears in the first ``k`` slots.
    """
    if len(y_true) == 0:
        return 0.0
    hit = 0
    for yt, ranks in zip(y_true, y_pred_ranked):
        rr = list(ranks)[:k]
        if yt in rr:
            hit += 1
    return hit / len(y_true)


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """
    Macro-averaged F1 for multi-class integer labels.

    Parameters
    ----------
    y_true : Sequence[int]
        Ground truth.
    y_pred : Sequence[int]
        Predicted labels.

    Returns
    -------
    float
        Macro F1 score in [0,1].
    """
    if len(y_true) == 0:
        return 0.0
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def aurc(accuracy_list: list[float]) -> float:
    """
    Trapezoidal area under the accuracy-vs-compression curve, normalized to [0,1].

    Assumes ``accuracy_list`` is ordered along the compression axis (e.g., Q high→low).

    Parameters
    ----------
    accuracy_list : list[float]
        Accuracies in [0,1].

    Returns
    -------
    float
        Normalized AURC.
    """
    if not accuracy_list:
        return 0.0
    y = np.asarray(accuracy_list, dtype=np.float64)
    if y.size == 1:
        return float(np.clip(y[0], 0.0, 1.0))
    area = np.trapz(y)
    denom = float((y.size - 1) * 1.0)
    return float(np.clip(area / denom, 0.0, 1.0))


def image_level_predictions_cnn(
    clf: CNNClassifier, loader: DataLoader
) -> tuple[list[int], list[list[int]]]:
    """
    Aggregate patch softmax predictions per parent image by averaging probabilities.

    Parameters
    ----------
    clf : CNNClassifier
        Trained classifier.
    loader : DataLoader
        ``PRNUPatchDataset`` loader with ``prnu_collate``.

    Returns
    -------
    tuple[list[int], list[list[int]]]
        ``y_true`` per image and class-index rankings (best first).
    """
    clf.model.eval()
    device = clf.device
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    labels: dict[str, int] = {}
    with torch.no_grad():
        for x, y, paths in loader:
            x = x.to(device, non_blocking=True)
            prob = F.softmax(clf.model(x), dim=1).detach().cpu()
            for i in range(len(paths)):
                pt = paths[i]
                sums[pt] = sums.get(pt, torch.zeros_like(prob[0])) + prob[i]
                counts[pt] = counts.get(pt, 0) + 1
                labels[pt] = int(y[i].item())
    y_true: list[int] = []
    ranked: list[list[int]] = []
    for pt in sorted(sums.keys()):
        meanp = sums[pt] / max(1, counts[pt])
        order = torch.argsort(meanp, descending=True).tolist()
        ranked.append(order)
        y_true.append(labels[pt])
    return y_true, ranked


def image_level_predictions_siamese(
    clf: SiameseClassifier, loader: DataLoader
) -> tuple[list[int], list[list[int]]]:
    """
    Average patch embeddings per image, then rank devices by centroid distance.

    Parameters
    ----------
    clf : SiameseClassifier
        Trained encoder with ``fit_centroids`` already called.
    loader : DataLoader
        Patch loader with ``prnu_collate``.

    Returns
    -------
    tuple[list[int], list[list[int]]]
        ``y_true`` and ranked integer labels (best first).
    """
    clf.encoder.eval()
    device = clf.device
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    labels: dict[str, int] = {}
    with torch.no_grad():
        for x, y, paths in loader:
            x = x.to(device, non_blocking=True)
            z = clf.encoder(x).detach().cpu()
            for i in range(len(paths)):
                pt = paths[i]
                sums[pt] = sums.get(pt, torch.zeros_like(z[0])) + z[i]
                counts[pt] = counts.get(pt, 0) + 1
                labels[pt] = int(y[i].item())
    y_true: list[int] = []
    ranked: list[list[int]] = []
    for pt in sorted(sums.keys()):
        emean = F.normalize(sums[pt] / max(1, counts[pt]), dim=0).to(device)
        dist_pairs: list[tuple[int, float]] = []
        for lab in clf._label_list:
            c = clf._centroids[int(lab)].to(device)
            d = float(torch.norm(emean - c).item())
            dist_pairs.append((int(lab), d))
        dist_pairs.sort(key=lambda t: t[1])
        ranked.append([p for p, _ in dist_pairs])
        y_true.append(labels[pt])
    return y_true, ranked


def image_level_predictions_ncc(
    ncc,
    paths_and_labels: list[tuple[str, int]],
    device_str_to_int: dict[str, int],
    transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> tuple[list[int], list[list[int]]]:
    """
    Run NCC attribution on full images (one ranking per path).

    Parameters
    ----------
    ncc
        Fitted ``NCCBaseline``.
    paths_and_labels : list[tuple[str, int]]
        Test image paths and integer labels.
    device_str_to_int : dict[str, int]
        Maps **sanitized** device keys (matching fingerprint filenames) to ints.
    transform : callable, optional
        Applied to uint8 RGB **before** residual extraction.

    Returns
    -------
    tuple[list[int], list[list[int]]]
        Ground-truth labels and ranked class indices per image.
    """
    from tqdm import tqdm

    y_true: list[int] = []
    ranked: list[list[int]] = []
    for path, y in tqdm(paths_and_labels, desc="NCC images"):
        rgb = load_image_rgb_float(path)
        u8 = np.clip(rgb, 0, 255).astype(np.uint8)
        if transform is not None:
            u8 = transform(u8)
        ranks = ncc.predict(u8)
        ids: list[int] = []
        for dev, _s in ranks:
            sid = sanitize_device_id(dev)
            ids.append(device_str_to_int.get(sid, -1))
        ranked.append(ids)
        y_true.append(int(y))
    return y_true, ranked


def majority_vote_int(labels: list[int]) -> int:
    """
    Return the most frequent integer label (deterministic tie-break by smallest int).

    Parameters
    ----------
    labels : list[int]
        Patch-level predictions.

    Returns
    -------
    int
        Majority class.
    """
    if not labels:
        raise ValueError("empty labels")
    counts: dict[int, int] = {}
    for x in labels:
        counts[x] = counts.get(x, 0) + 1
    best = max(counts.values())
    cands = sorted([k for k, v in counts.items() if v == best])
    return cands[0]


def relative_accuracy_drop(acc_clean: float, acc_worst: float) -> float:
    """
    RAD = (acc_clean - acc_worst) / acc_clean with guard for zero division.

    Parameters
    ----------
    acc_clean : float
        Accuracy at lowest compression / clean setting.
    acc_worst : float
        Accuracy at strongest compression / worst case.

    Returns
    -------
    float
        Relative drop in [0, +inf), capped sensibly if ``acc_clean`` is tiny.
    """
    if acc_clean <= 1e-12:
        return 0.0
    return float(max(0.0, (acc_clean - acc_worst) / acc_clean))
