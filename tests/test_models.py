import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from threadpoolctl import threadpool_limits

from credit_risk import models


@pytest.fixture(scope="module")
def fitted_small(dataset):
    X, y, _ = dataset
    # Source ordering is not random: include both outcome blocks deliberately.
    ids = np.r_[np.flatnonzero(y == 0)[:100], np.flatnonzero(y == 1)[:60]]
    with threadpool_limits(limits=1):
        fitted = models.fit_models(X.iloc[ids], y.iloc[ids], folds=2)
    return fitted, X.drop(index=ids).iloc[:30]


def test_real_models_produce_valid_probabilities(fitted_small):
    fitted, X = fitted_small
    for model in fitted.models.values():
        probabilities = model.predict_proba(X)
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)


def test_raw_calibrated_pair_uses_identical_base_estimator(fitted_small):
    fitted, X = fitted_small
    calibrator = fitted.models["calibrated"]
    search = calibrator.calibrated_classifiers_[0].estimator
    assert fitted.models["boosting"] is search.best_estimator_
    assert calibrator.ensemble is False
    assert isinstance(calibrator.estimator, GridSearchCV)
    assert calibrator.method == "sigmoid"
    # A fitted positive sigmoid map preserves ordering and hence ranking metrics.
    raw = fitted.models["boosting"].predict_proba(X)[:, 1]
    calibrated = calibrator.predict_proba(X)[:, 1]
    assert np.all(np.diff(calibrated[np.argsort(raw)]) >= -1e-12)


class AuditedEstimator(ClassifierMixin, BaseEstimator):
    events = []

    def __init__(self, strength=1.0):
        self.strength = strength

    def fit(self, X, y):
        self.fit_ids_ = set(X.index)
        self.classes_ = np.array([0, 1])
        type(self).events.append(("fit", self.fit_ids_))
        return self

    def decision_function(self, X):
        # Fires during tuning scoring AND cross-fitted calibration prediction.
        assert self.fit_ids_.isdisjoint(X.index), "Training row used for its own evaluation score"
        type(self).events.append(("score", set(X.index)))
        return X["score"].to_numpy() * self.strength

    def predict_proba(self, X):
        p = expit(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


def test_calibration_scores_exclude_rows_from_tuning_and_fitting(monkeypatch):
    AuditedEstimator.events = []

    def search(*args):
        return GridSearchCV(
            AuditedEstimator(),
            {"strength": [0.5, 1.0]},
            cv=StratifiedKFold(2, shuffle=True, random_state=3),
            scoring="neg_log_loss",
            error_score="raise",
        )

    monkeypatch.setattr(models, "make_search", search)
    X = pd.DataFrame({"score": np.linspace(-1, 1, 40)}, index=range(40))
    y = pd.Series(np.tile([0, 1], 20), index=X.index)
    calibrated = models.make_calibrated(folds=2).fit(X, y)
    assert len([event for event in AuditedEstimator.events if event[0] == "score"]) >= 10
    new_X = pd.DataFrame({"score": [-1.0, 1.0]}, index=[101, 102])
    assert calibrated.predict_proba(new_X).shape == (2, 2)
