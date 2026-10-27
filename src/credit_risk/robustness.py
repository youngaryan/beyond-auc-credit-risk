"""Hypothetical label-shift scenarios and fold stability summaries."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


def prior_shift(
    probabilities: ArrayLike, source_prevalence: float, target_prevalence: float
) -> NDArray:
    """Bayes odds adjustment, valid only under stable p(X|Y) and source posteriors."""
    if not 0 < source_prevalence < 1 or not 0 < target_prevalence < 1:
        raise ValueError("Prevalences must be strictly between zero and one")
    p = np.asarray(probabilities, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    multiplier = (target_prevalence / (1 - target_prevalence)) / (
        source_prevalence / (1 - source_prevalence)
    )
    return multiplier * p / (1 - p + multiplier * p)


def stability_summary(cv: pd.DataFrame) -> dict:
    wide = cv.pivot(index="fold", columns="model", values="roc_auc")
    difference = wide.boosting - wide.logistic
    scores = cv.pivot(index="fold", columns="model", values="log_loss")
    return {
        "n_folds": len(wide),
        "boosting_auc_wins": int((difference > 0).sum()),
        "boosting_auc_ties": int((abs(difference) < 1e-12).sum()),
        "auc_difference_mean": float(difference.mean()),
        "auc_difference_min": float(difference.min()),
        "auc_difference_max": float(difference.max()),
        "calibration_log_loss_wins": int((scores.calibrated < scores.boosting).sum()),
        "interpretation": "overlapping training folds; descriptive stability, not independent replications",
    }
