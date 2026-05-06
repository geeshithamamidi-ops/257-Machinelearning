"""
Confidence calibration metrics and diagnostics.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def expected_calibration_error(
    confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 15
) -> float:
    """
    Expected Calibration Error (ECE) with equal-width bins on [0,1].

    Parameters
    ----------
    confidences : np.ndarray
        Shape (N,), predicted probabilities for the predicted class.
    correctness : np.ndarray
        Shape (N,) boolean/int whether prediction was correct.
    n_bins : int
        Number of bins.

    Returns
    -------
    float
        ECE in [0,1].
    """
    conf = np.asarray(confidences, dtype=np.float64).ravel()
    corr = np.asarray(correctness, dtype=np.float64).ravel()
    if conf.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = conf.size
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if i == 0:
            m = (conf >= bins[i]) & (conf <= bins[i + 1])
        cnt = m.sum()
        if cnt == 0:
            continue
        acc = corr[m].mean()
        conf_mean = conf[m].mean()
        ece += (cnt / n) * abs(acc - conf_mean)
    return float(ece)


def brier_score(y_true_onehot: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Multiclass Brier score: mean squared error between one-hot labels and probs.

    Parameters
    ----------
    y_true_onehot : np.ndarray
        Shape (N, C) one-hot encoded labels.
    y_proba : np.ndarray
        Shape (N, C) predicted probabilities summing to 1 per row.

    Returns
    -------
    float
        Brier score (lower is better).
    """
    yt = np.asarray(y_true_onehot, dtype=np.float64)
    yp = np.asarray(y_proba, dtype=np.float64)
    if yt.size == 0:
        return 0.0
    return float(np.mean(np.sum((yp - yt) ** 2, axis=1)))


def reliability_diagram_data(
    confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 15
) -> dict[str, list[float]]:
    """
    Bin-wise statistics for reliability diagrams.

    Parameters
    ----------
    confidences : np.ndarray
        Max-probabilities for predicted class, shape (N,).
    correctness : np.ndarray
        Binary correctness, shape (N,).
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    dict[str, list[float]]
        ``bin_confidence``, ``bin_accuracy``, ``bin_count`` lists aligned per bin.
    """
    conf = np.asarray(confidences, dtype=np.float64).ravel()
    corr = np.asarray(correctness, dtype=np.float64).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bc: list[float] = []
    ba: list[float] = []
    bn: list[float] = []
    for i in range(n_bins):
        if i == 0:
            m = (conf >= bins[i]) & (conf <= bins[i + 1])
        else:
            m = (conf > bins[i]) & (conf <= bins[i + 1])
        cnt = int(m.sum())
        bn.append(float(cnt))
        if cnt == 0:
            bc.append(0.0)
            ba.append(0.0)
        else:
            bc.append(float(conf[m].mean()))
            ba.append(float(corr[m].mean()))
    return {"bin_confidence": bc, "bin_accuracy": ba, "bin_count": bn}


def selective_risk_curve(
    confidences: np.ndarray,
    correctness: np.ndarray,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """
    Accuracy vs coverage for confidence thresholds (selective classification).

    Parameters
    ----------
    confidences : np.ndarray
        Shape (N,) confidence scores in [0,1].
    correctness : np.ndarray
        Shape (N,) binary correctness.
    thresholds : list[float] | None
        Increasing confidence cutoffs; default linspace grid.

    Returns
    -------
    dict[str, Any]
        ``thresholds``, ``coverage``, ``accuracy`` lists of equal length.
    """
    conf = np.asarray(confidences, dtype=np.float64).ravel()
    corr = np.asarray(correctness, dtype=np.float64).ravel()
    if thresholds is None:
        thresholds = list(np.linspace(0.0, 1.0, 21))
    covs: list[float] = []
    accs: list[float] = []
    n = conf.size
    for t in thresholds:
        m = conf >= t
        c = float(m.mean()) if n else 0.0
        if m.sum() == 0:
            a = 0.0
        else:
            a = float(corr[m].mean())
        covs.append(c)
        accs.append(a)
    return {"thresholds": thresholds, "coverage": covs, "accuracy": accs}
