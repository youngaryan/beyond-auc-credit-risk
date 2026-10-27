import numpy as np
import pytest

from credit_risk.data import CATEGORICAL, LEVELS, NUMERIC, split_indices
from credit_risk.preprocessing import make_preprocessor


def test_categorical_encoding_is_not_continuous(dataset):
    X, _, _ = dataset
    fitted = make_preprocessor().fit(X.iloc[:80])
    output = fitted.transform(X.iloc[:3])
    expected_width = len(NUMERIC) + sum(len(LEVELS[name]) - 1 for name in CATEGORICAL)
    assert output.shape == (3, expected_width)
    assert set(np.unique(output[:, 3:])).issubset({0, 1})
    for category, levels in zip(
        fitted.named_transformers_["categorical"]["onehot"].categories_,
        [LEVELS[name] for name in CATEGORICAL],
        strict=True,
    ):
        np.testing.assert_array_equal(category, levels)


def test_learning_is_training_only_and_transform_does_not_refit(dataset):
    X, y, _ = dataset
    train, test = split_indices(y)
    fitted = make_preprocessor().fit(X.loc[train])
    scaler = fitted.named_transformers_["numeric"]["scale"]
    np.testing.assert_allclose(scaler.mean_, X.loc[train, NUMERIC].mean())
    held = X.loc[test].copy()
    held[NUMERIC] = held[NUMERIC] + 1_000_000
    expected = scaler.mean_.copy()
    output = fitted.transform(held)
    np.testing.assert_array_equal(scaler.mean_, expected)
    assert output[:, :3].min() > 10


def test_imputation_uses_training_median(dataset):
    X, _, _ = dataset
    training = X.iloc[:100].astype(float).copy()
    training.loc[training.index[0], "duration"] = np.nan
    fitted = make_preprocessor().fit(training)
    assert (
        fitted.named_transformers_["numeric"]["impute"].statistics_[0] == training.duration.median()
    )
    assert np.isfinite(fitted.transform(training)).all()


def test_documented_unseen_category_and_invalid_category(dataset):
    X, _, _ = dataset
    fitted = make_preprocessor().fit(X.iloc[:100])
    record = X.iloc[:1].copy()
    record["purpose"] = 7
    assert np.isfinite(fitted.transform(record)).all()
    record["purpose"] = 99
    with pytest.raises(ValueError, match="unknown categories"):
        fitted.transform(record)
