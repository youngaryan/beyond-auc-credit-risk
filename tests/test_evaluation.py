import numpy as np
import pytest

from credit_risk.calibration import calibration_diagnostics, reliability_bins, wilson_interval
from credit_risk.evaluation import metric_values, paired_bootstrap
from credit_risk.robustness import prior_shift


def test_known_metrics():
    result = metric_values([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert result["roc_auc"] == pytest.approx(0.75)
    assert result["average_precision"] == pytest.approx(5 / 6)
    assert result["brier"] == pytest.approx((0.01 + 0.16 + 0.65**2 + 0.04) / 4)
    assert result["log_loss"] == pytest.approx(-np.log([0.9, 0.6, 0.35, 0.8]).mean())
    assert result["sensitivity"] == 0.5 and result["specificity"] == 1
    assert result["precision"] == 1


def test_constant_baseline_and_undefined_precision():
    result = metric_values([0] * 7 + [1] * 3, [0.3] * 10)
    assert result["roc_auc"] == 0.5
    assert result["average_precision"] == pytest.approx(0.3)
    assert result["brier"] == pytest.approx(0.21)
    assert np.isnan(result["precision"])
    assert result["predicted_positive"] == 0


@pytest.mark.parametrize("p", [[0.2], [0.2, np.nan], [-0.1, 1], [0, 1.01], [np.inf, 0.2]])
def test_invalid_probability_inputs_rejected(p):
    with pytest.raises(ValueError):
        metric_values([0, 1], p)


def test_bootstrap_pairing_identity_and_reproducibility():
    y, p = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
    draws, summary, metadata = paired_bootstrap(y, {"boosting": p, "logistic": p}, 40, 9)
    second, _, _ = paired_bootstrap(y, {"boosting": p, "logistic": p}, 40, 9)
    np.testing.assert_allclose(draws.select_dtypes("number"), second.select_dtypes("number"))
    assert (
        (summary.loc[summary.kind == "difference", ["estimate", "lower", "upper"]] == 0).all().all()
    )
    assert metadata["accepted"] == 40 and metadata["rejected_one_class"] == 0


def test_bootstrap_single_class_and_degenerate_draws():
    with pytest.raises(ValueError, match="both outcome"):
        paired_bootstrap([1, 1], {"model": [0.4, 0.5]}, 10)
    _, _, metadata = paired_bootstrap([0, 1], {"model": [0.2, 0.8]}, 100, 9, stratified=False)
    assert 0 < metadata["rejected_one_class"] < 100
    assert metadata["accepted"] + metadata["rejected_one_class"] == 100


def test_reliability_endpoints_empty_bins_and_counts():
    bins = reliability_bins([0, 1, 1], [0, 0.5, 1])
    assert bins.n.sum() == 3 and bins.bad.sum() == 2
    assert bins.iloc[0].n == bins.iloc[-1].n == 1
    assert bins.loc[bins.n == 0, "observed_frequency"].isna().all()
    assert wilson_interval(0, 10)[0] == 0
    assert wilson_interval(10, 10)[1] == 1


def test_calibration_diagnostic_recovers_known_relationship():
    p = np.repeat([0.1, 0.3, 0.5, 0.7, 0.9], 100)
    y = np.concatenate([np.r_[np.ones(n), np.zeros(100 - n)] for n in [10, 30, 50, 70, 90]])
    result = calibration_diagnostics(y, p)
    assert result["intercept"] == pytest.approx(0, abs=1e-5)
    assert result["slope"] == pytest.approx(1, abs=1e-5)
    assert result["calibration_in_large"] == pytest.approx(0, abs=1e-5)
    assert result["slope_se"] > 0


def test_constant_calibration_slope_is_unidentified():
    diagnostic = calibration_diagnostics([0, 0, 1], [1 / 3] * 3)
    assert diagnostic["slope"] is None and "constant" in diagnostic["status"]
    assert diagnostic["calibration_in_large"] == pytest.approx(0, abs=1e-6)


def test_prior_adjustment_bayes_identity_monotonicity_and_boundaries():
    p = np.linspace(0, 1, 21)
    np.testing.assert_allclose(prior_shift(p, 0.3, 0.3), p)
    adjusted = prior_shift(p, 0.3, 0.1)
    assert adjusted[0] == 0 and adjusted[-1] == 1
    assert np.all(np.diff(adjusted) > 0)
    assert prior_shift(0.3, 0.3, 0.1) == pytest.approx(0.1)
    assert prior_shift(0.5, 0.3, 0.1) == pytest.approx(7 / 34)
    with pytest.raises(ValueError):
        prior_shift(p, 0, 0.1)
