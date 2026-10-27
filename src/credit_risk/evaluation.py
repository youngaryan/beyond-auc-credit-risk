"""Probability metrics and paired bootstrap intervals conditional on case mix."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

KEY_METRICS = ("roc_auc", "average_precision", "brier", "log_loss")
PAIRS = (("boosting", "logistic"), ("calibrated", "boosting"))


def validate_probabilities(y: ArrayLike, p: ArrayLike) -> tuple[NDArray, NDArray]:
    target, prob = np.asarray(y), np.asarray(p, dtype=float)
    if target.ndim != 1 or prob.ndim != 1 or target.shape != prob.shape or len(target) == 0:
        raise ValueError("Targets and probabilities must be aligned nonempty 1D arrays")
    if not np.isin(target, [0, 1]).all():
        raise ValueError("Target must use 1=bad, 0=good")
    if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
        raise ValueError("Probabilities must be finite and between zero and one")
    return target.astype(int), prob


def metric_values(y: ArrayLike, p: ArrayLike, threshold: float = 0.5) -> dict[str, float]:
    target, prob = validate_probabilities(y, p)
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between zero and one")
    pred = prob >= threshold
    tp = int(np.sum(pred & (target == 1)))
    tn = int(np.sum(~pred & (target == 0)))
    fp = int(np.sum(pred & (target == 0)))
    fn = int(np.sum(~pred & (target == 1)))
    two_classes = len(np.unique(target)) == 2
    recall = tp / (tp + fn) if tp + fn else float("nan")
    return {
        "roc_auc": float(roc_auc_score(target, prob)) if two_classes else float("nan"),
        "average_precision": float(average_precision_score(target, prob))
        if two_classes
        else float("nan"),
        "brier": float(brier_score_loss(target, prob)),
        "log_loss": float(log_loss(target, prob, labels=[0, 1])),
        "sensitivity": recall,
        "recall": recall,
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "predicted_positive": tp + fp,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def paired_bootstrap(
    y: ArrayLike,
    predictions: Mapping[str, ArrayLike],
    n_boot: int = 2000,
    seed: int = 72,
    stratified: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Same resampled observations for every model; no model refitting.

    Stratification conditions on the observed class counts. An ordinary bootstrap
    is available for tests: one-class draws are excluded from all metrics together
    and their number is disclosed. An initially single-class target is rejected.
    """
    if n_boot < 2 or not predictions:
        raise ValueError("At least two replicates and one prediction vector are required")
    arrays = {key: validate_probabilities(y, p)[1] for key, p in predictions.items()}
    target = np.asarray(y, dtype=int)
    groups = [np.flatnonzero(target == label) for label in [0, 1]]
    if any(len(group) == 0 for group in groups):
        raise ValueError("Bootstrap discrimination requires both outcome classes")
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    rejected = 0
    for replicate in range(n_boot):
        indices = (
            np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
            if stratified
            else rng.integers(0, len(target), len(target))
        )
        if len(np.unique(target[indices])) < 2:
            rejected += 1
            continue
        for model, probabilities in arrays.items():
            values = metric_values(target[indices], probabilities[indices])
            records.append(
                {
                    "replicate": replicate,
                    "model": model,
                    **{name: values[name] for name in KEY_METRICS},
                }
            )
    if not records:
        raise ValueError("No valid bootstrap resamples")
    draws = pd.DataFrame(records)
    summary: list[dict] = []
    points = {name: metric_values(target, p) for name, p in arrays.items()}
    for name in arrays:
        subset = draws.loc[draws.model == name]
        for metric in KEY_METRICS:
            lower, upper = np.quantile(subset[metric], [0.025, 0.975])
            summary.append(
                {
                    "kind": "model",
                    "comparison": name,
                    "metric": metric,
                    "estimate": points[name][metric],
                    "lower": lower,
                    "upper": upper,
                }
            )
    for first, second in PAIRS:
        if first not in arrays or second not in arrays:
            continue
        for metric in KEY_METRICS:
            wide = draws.pivot(index="replicate", columns="model", values=metric)
            lower, upper = np.quantile(wide[first] - wide[second], [0.025, 0.975])
            summary.append(
                {
                    "kind": "difference",
                    "comparison": f"{first}_minus_{second}",
                    "metric": metric,
                    "estimate": points[first][metric] - points[second][metric],
                    "lower": lower,
                    "upper": upper,
                }
            )
    return (
        draws,
        pd.DataFrame(summary),
        {
            "requested": n_boot,
            "accepted": n_boot - rejected,
            "rejected_one_class": rejected,
            "stratified": stratified,
            "seed": seed,
            "interval": "percentile 95%",
            "conditioning": "fitted models and observed class counts; not training or prevalence uncertainty",
        },
    )
