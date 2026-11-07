"""Metrics with patient-level bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, roc_auc_score)


def kl5_metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    """5-class KL grading. ``prob`` has shape (n, 5)."""
    pred = prob.argmax(1)
    oa_true = (y >= 2).astype(int)
    oa_prob = prob[:, 2:].sum(1)
    out = {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "qwk": cohen_kappa_score(y, pred, weights="quadratic"),
        "mae_grade": float(np.abs(y - pred).mean()),
    }
    out.update({f"oa_{k}": v for k, v in binary_metrics(oa_true, oa_prob).items()})
    return out


def binary_metrics(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "accuracy": (tp + tn) / len(y),
        "auc": roc_auc_score(y, prob) if len(set(y)) == 2 else np.nan,
    }


def bootstrap_ci(metric_fn, y, prob, groups, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Resample patients (not images) with replacement; return point estimate and 95% CI."""
    rng = np.random.default_rng(seed)
    point = metric_fn(y, prob)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    samples = {k: [] for k in point}
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        try:
            m = metric_fn(y[idx], prob[idx])
        except ValueError:
            continue
        for k, v in m.items():
            samples[k].append(v)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: {"value": float(point[k]),
                "ci_low": float(np.nanpercentile(samples[k], lo)),
                "ci_high": float(np.nanpercentile(samples[k], hi))} for k in point}
