"""Descriptive reliability bins and evaluation-only logistic diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import minimize, root_scalar
from scipy.special import expit, logit

from credit_risk.evaluation import validate_probabilities


def wilson_interval(successes: int, n: int) -> tuple[float, float]:
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("Invalid binomial counts")
    z = 1.959963984540054
    rate = successes / n
    denom = 1 + z * z / n
    middle = (rate + z * z / (2 * n)) / denom
    half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denom
    return (
        0.0 if successes == 0 else max(0.0, middle - half),
        1.0 if successes == n else min(1.0, middle + half),
    )


def reliability_bins(y: ArrayLike, p: ArrayLike, n_bins: int = 5) -> pd.DataFrame:
    target, prob = validate_probabilities(y, p)
    if n_bins < 2:
        raise ValueError("At least two calibration bins are required")
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.minimum(np.searchsorted(edges, prob, side="right") - 1, n_bins - 1)
    records = []
    for index in range(n_bins):
        mask = bins == index
        n = int(mask.sum())
        successes = int(target[mask].sum())
        lower, upper = wilson_interval(successes, n) if n else (np.nan, np.nan)
        records.append(
            {
                "bin": index,
                "left": edges[index],
                "right": edges[index + 1],
                "n": n,
                "bad": successes,
                "mean_probability": float(prob[mask].mean()) if n else np.nan,
                "observed_frequency": successes / n if n else np.nan,
                "wilson_lower": lower,
                "wilson_upper": upper,
            }
        )
    return pd.DataFrame(records)


def calibration_diagnostics(y: ArrayLike, p: ArrayLike) -> dict:
    """Fit y ~ intercept + slope*logit(p) diagnostically; never alter predictions.

    Wald uncertainty is approximate. Singular, one-class and separated solutions
    are explicitly non-estimable. Constant predictions do not identify a slope.
    """
    target, prob = validate_probabilities(y, p)
    bins = reliability_bins(target, prob)
    nonempty = bins.loc[bins.n > 0]
    ece = float(
        np.sum(nonempty.n * abs(nonempty.observed_frequency - nonempty.mean_probability))
        / len(target)
    )
    result: dict = {
        "ece_5_bins": ece,
        "intercept": None,
        "slope": None,
        "intercept_se": None,
        "slope_se": None,
        "calibration_in_large": None,
    }
    if len(np.unique(target)) < 2:
        return {**result, "status": "not estimable: one outcome class"}
    scores = logit(np.clip(prob, 1e-8, 1 - 1e-8))
    offset_fit = root_scalar(lambda a: np.sum(expit(a + scores) - target), bracket=[-40, 40])
    result["calibration_in_large"] = float(offset_fit.root)
    if np.std(scores) < 1e-10:
        return {**result, "status": "not estimable: constant predictions"}
    design = np.column_stack([np.ones(len(target)), scores])

    def objective(beta: np.ndarray) -> float:
        eta = design @ beta
        return float(np.sum(np.logaddexp(0, eta) - target * eta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        return design.T @ (expit(design @ beta) - target)

    fit = minimize(
        objective, np.array([0.0, 1.0]), jac=gradient, method="BFGS", options={"gtol": 1e-6}
    )
    fitted = expit(design @ fit.x)
    info = design.T @ (design * (fitted * (1 - fitted))[:, None])
    if (
        (not fit.success and np.linalg.norm(gradient(fit.x)) > 1e-4)
        or np.max(abs(fit.x)) > 25
        or np.linalg.cond(info) > 1e10
    ):
        return {**result, "status": "not estimable: separation or unstable optimisation"}
    standard_errors = np.sqrt(np.diag(np.linalg.inv(info)))
    return {
        **result,
        "intercept": float(fit.x[0]),
        "slope": float(fit.x[1]),
        "intercept_se": float(standard_errors[0]),
        "slope_se": float(standard_errors[1]),
        "status": "estimated; approximate Wald SE; evaluation diagnostic only",
    }
