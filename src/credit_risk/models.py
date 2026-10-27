"""Nested tuning and cross-fitted calibration with an identical raw comparator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from credit_risk.preprocessing import make_preprocessor

MODEL_NAMES = {
    "constant": "Constant prevalence",
    "logistic": "Logistic regression",
    "boosting": "Gradient boosting",
    "calibrated": "Boosting + sigmoid",
}
GRIDS = {
    "logistic": {"model__C": [0.1, 1.0, 10.0]},
    "boosting": {"model__learning_rate": [0.05, 0.1], "model__max_depth": [1, 2]},
}


def make_pipeline(kind: str, seed: int = 42) -> Pipeline:
    if kind == "logistic":
        model = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=3000, tol=1e-8, random_state=seed
        )
    elif kind == "boosting":
        model = GradientBoostingClassifier(
            n_estimators=120, min_samples_leaf=15, subsample=1.0, random_state=seed
        )
    else:
        raise ValueError(f"Unknown model {kind}")
    return Pipeline([("preprocess", make_preprocessor()), ("model", model)])


def make_search(kind: str, seed: int = 42, folds: int = 5) -> GridSearchCV:
    return GridSearchCV(
        make_pipeline(kind, seed),
        GRIDS[kind],
        scoring="neg_log_loss",
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + 10),
        refit=True,
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
    )


def make_calibrated(seed: int = 42, folds: int = 5) -> CalibratedClassifierCV:
    # Crucially, tuning is INSIDE each calibration training fold.
    return CalibratedClassifierCV(
        estimator=make_search("boosting", seed, folds),
        method="sigmoid",
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + 20),
        ensemble=False,
        n_jobs=1,
    )


@dataclass
class FittedModels:
    models: dict[str, Any]
    searches: dict[str, GridSearchCV]


def fit_models(X: pd.DataFrame, y: pd.Series, seed: int = 42, folds: int = 5) -> FittedModels:
    constant = DummyClassifier(strategy="prior").fit(X, y)
    logistic_search = make_search("logistic", seed, folds).fit(X, y)
    calibrated = make_calibrated(seed, folds).fit(X, y)
    boosting_search = calibrated.calibrated_classifiers_[0].estimator
    # This is the exact full-data base estimator used by the sigmoid model.
    raw_boosting = boosting_search.best_estimator_
    return FittedModels(
        {
            "constant": constant,
            "logistic": logistic_search.best_estimator_,
            "boosting": raw_boosting,
            "calibrated": calibrated,
        },
        {"logistic": logistic_search, "boosting": boosting_search},
    )
