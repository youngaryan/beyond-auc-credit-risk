import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from credit_risk.data import split_indices
from credit_risk.evaluation import metric_values
from credit_risk.models import make_pipeline


@pytest.mark.parametrize("kind", ["logistic", "boosting"])
def test_fixed_seed_pipeline_is_reproducible(dataset, kind):
    X, y, _ = dataset
    train, test = split_indices(y)
    assert (train, test) == split_indices(y)
    estimator = make_pipeline(kind)
    with threadpool_limits(limits=1):
        a = clone(estimator).fit(X.loc[train], y.loc[train])
        b = clone(estimator).fit(X.loc[train], y.loc[train])
    np.testing.assert_array_equal(a.predict_proba(X.loc[test]), b.predict_proba(X.loc[test]))


@pytest.mark.integration
def test_artifact_bundle_is_complete_and_consistent(root):
    from credit_risk.report import verify_artifacts

    assert verify_artifacts(root)["status"] == "complete"
    metrics = json.loads((root / "results/metrics.json").read_text())
    predictions = pd.read_csv(root / "results/holdout_predictions.csv")
    splits = json.loads((root / "results/split_manifest.json").read_text())
    assert predictions.row_id.tolist() == splits["test_row_ids"]
    for name in metrics["holdout"]:
        recomputed = metric_values(predictions.bad_credit, predictions[name])
        for metric in ["roc_auc", "average_precision", "brier", "log_loss"]:
            assert recomputed[metric] == pytest.approx(metrics["holdout"][name][metric], abs=1e-12)
    outer_predictions = pd.read_csv(root / "results/cv_predictions.csv")
    assert outer_predictions.row_id.is_unique
    assert set(outer_predictions.row_id) == set(splits["train_row_ids"])
    for fold in splits["outer_folds"]:
        assert set(fold["fit_row_ids"]).isdisjoint(fold["validation_row_ids"])
        assert set(fold["fit_row_ids"]).isdisjoint(splits["test_row_ids"])
        assert set(fold["validation_row_ids"]).isdisjoint(splits["test_row_ids"])
