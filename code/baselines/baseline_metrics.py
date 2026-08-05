"""Evaluation metrics shared by the Random Forest and MaxEnt baselines."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, assigning equal ranks to ties."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    sorted_values = values[order]

    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end

    return ranks


def _spearman_tie_aware(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman correlation using average ranks for ties."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or y.size < 3:
        return np.nan

    x_rank = _average_ranks(x)
    y_rank = _average_ranks(y)
    if np.all(x_rank == x_rank[0]) or np.all(y_rank == y_rank[0]):
        return np.nan
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def continuous_boyce(
    y_true: np.ndarray,
    scores: np.ndarray,
    nbins_max: int = 20,
    min_per_group: int = 10,
) -> float:
    """Compute the continuous Boyce index from binary labels and scores."""
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    background = scores[labels == 0]
    presence = scores[labels == 1]

    if background.size < min_per_group or presence.size < min_per_group:
        return np.nan

    unique_background = np.unique(background)
    if unique_background.size < 3:
        return np.nan

    n_bins = min(nbins_max, max(3, unique_background.size - 1))
    bounds = np.quantile(background, np.linspace(0.0, 1.0, n_bins + 1))
    bounds[0] -= 1e-12
    bounds[-1] += 1e-12

    ratios: list[float] = []
    centers: list[float] = []
    n_background = float(background.size)
    n_presence = float(presence.size)

    for lower, upper in zip(bounds[:-1], bounds[1:]):
        in_background = (background >= lower) & (background < upper)
        background_count = int(in_background.sum())
        if background_count == 0:
            continue

        presence_count = int(((presence >= lower) & (presence < upper)).sum())
        ratios.append((presence_count / n_presence) / (background_count / n_background))
        centers.append(0.5 * (lower + upper))

    if len(ratios) < 3:
        return np.nan
    return _spearman_tie_aware(np.asarray(centers), np.asarray(ratios))


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    nbins_boyce: int = 20,
) -> dict[str, float]:
    """Return continuous Boyce and ROC AUC after removing non-finite scores."""
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]

    if labels.size == 0:
        return {"Boyce": np.nan, "ROC_AUC": np.nan}

    auc = (
        float(roc_auc_score(labels, scores))
        if np.unique(labels).size == 2
        else np.nan
    )
    return {
        "Boyce": continuous_boyce(labels, scores, nbins_max=nbins_boyce),
        "ROC_AUC": auc,
    }
