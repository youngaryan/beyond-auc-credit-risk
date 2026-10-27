"""Execute the prespecified research; persist all evidence before reporting."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from credit_risk.calibration import calibration_diagnostics, reliability_bins
from credit_risk.data import (
    CATEGORICAL,
    LEVELS,
    NUMERIC,
    codebook,
    counts,
    load_data,
    sha256,
    split_indices,
)
from credit_risk.evaluation import KEY_METRICS, metric_values, paired_bootstrap
from credit_risk.models import fit_models
from credit_risk.robustness import prior_shift, stability_summary


@dataclass(frozen=True)
class Config:
    seed: int = 42
    folds: int = 5
    bootstrap_replicates: int = 2000
    threshold: float = 0.5
    permutation_repeats: int = 10


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def exploratory_outputs(X: pd.DataFrame, y: pd.Series, root: Path) -> dict:
    numerical = X[NUMERIC].describe().T
    outliers = {}
    for feature in NUMERIC:
        q1, q3 = X[feature].quantile([0.25, 0.75])
        flags = (X[feature] < q1 - 1.5 * (q3 - q1)) | (X[feature] > q3 + 1.5 * (q3 - q1))
        outliers[feature] = int(flags.sum())
        numerical.loc[feature, "iqr_flag_count"] = int(flags.sum())
    numerical.to_csv(root / "results/training_numeric_summary.csv", index_label="feature")
    records = []
    for feature in CATEGORICAL:
        for level in LEVELS[feature]:
            mask = X[feature] == level
            records.append(
                {
                    "feature": feature,
                    "level": level,
                    "n": int(mask.sum()),
                    "bad": int(y.loc[mask].sum()),
                    "frequency": float(y.loc[mask].mean()) if mask.any() else np.nan,
                }
            )
    pd.DataFrame(records).to_csv(root / "results/training_categories.csv", index=False)
    return {
        "population": "training only",
        "numeric": numerical.to_dict(orient="index"),
        "iqr_flags": outliers,
        "policy": "descriptive flags retained; no data-dependent outlier deletion",
    }


def coefficient_table(pipeline: Any, labels: dict) -> pd.DataFrame:
    descriptions = [f"{feature} (per training SD)" for feature in NUMERIC]
    references = ["numeric; other features held fixed"] * len(NUMERIC)
    for feature in CATEGORICAL:
        for level in LEVELS[feature][1:]:
            descriptions.append(f"{feature}: {labels[feature][level]}")
            references.append(f"{LEVELS[feature][0]}: {labels[feature][LEVELS[feature][0]]}")
    return pd.DataFrame(
        {
            "feature": pipeline.named_steps["preprocess"].get_feature_names_out(),
            "description": descriptions,
            "reference": references,
            "coefficient": pipeline.named_steps["model"].coef_[0],
        }
    )


def run_experiment(root: Path, config: Config | None = None) -> dict:
    config = config or Config()
    start = time.perf_counter()
    started_utc = datetime.now(UTC).isoformat()
    result_dir = root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    X, y, audit = load_data(root / "data/raw")
    train_ids, test_ids = split_indices(y, config.seed)
    X_train, y_train = X.loc[train_ids], y.loc[train_ids]
    partition = {
        "seed": config.seed,
        "training": counts(y_train),
        "holdout": counts(y.loc[test_ids]),
        "train_row_ids": train_ids,
        "test_row_ids": test_ids,
        "outer_folds": [],
    }
    write_json(result_dir / "data_audit.json", audit)
    eda = exploratory_outputs(X_train, y_train, root)
    write_json(result_dir / "training_eda.json", eda)
    outer = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.seed + 1)
    cv_records, cv_predictions, importance_records, search_records = [], [], [], []
    # Single-threaded BLAS makes small-data fitting fast and more repeatable.
    with threadpool_limits(limits=1):
        for fold, (fit_pos, val_pos) in enumerate(outer.split(X_train, y_train), 1):
            fit_X, fit_y = X_train.iloc[fit_pos], y_train.iloc[fit_pos]
            val_X, val_y = X_train.iloc[val_pos], y_train.iloc[val_pos]
            print(
                f"Outer fold {fold}/{config.folds}: fit {len(fit_y)}, validate {len(val_y)}",
                flush=True,
            )
            partition["outer_folds"].append(
                {
                    "fold": fold,
                    "fit": counts(fit_y),
                    "validation": counts(val_y),
                    "fit_row_ids": fit_X.index.tolist(),
                    "validation_row_ids": val_X.index.tolist(),
                }
            )
            fitted = fit_models(fit_X, fit_y, config.seed, config.folds)
            prediction_frame = pd.DataFrame(
                {"row_id": val_X.index, "fold": fold, "bad_credit": val_y.to_numpy()}
            )
            for name, model in fitted.models.items():
                probabilities = model.predict_proba(val_X)[:, 1]
                prediction_frame[name] = probabilities
                cv_records.append(
                    {
                        "fold": fold,
                        "model": name,
                        "n_fit": len(fit_y),
                        "n_validation": len(val_y),
                        **metric_values(val_y, probabilities, config.threshold),
                    }
                )
            cv_predictions.append(prediction_frame)
            for name, search in fitted.searches.items():
                for index, params in enumerate(search.cv_results_["params"]):
                    search_records.append(
                        {
                            "stage": f"outer_{fold}",
                            "model": name,
                            "params": json.dumps(params, sort_keys=True),
                            "mean_log_loss": -search.cv_results_["mean_test_score"][index],
                            "selected": index == search.best_index_,
                        }
                    )
            importance = permutation_importance(
                fitted.models["boosting"],
                val_X,
                val_y,
                scoring="neg_log_loss",
                n_repeats=config.permutation_repeats,
                random_state=config.seed + 50 + fold,
                n_jobs=1,
            )
            for feature, mean, std in zip(
                X.columns, importance.importances_mean, importance.importances_std, strict=True
            ):
                importance_records.append(
                    {"fold": fold, "feature": feature, "log_loss_increase": mean, "shuffle_sd": std}
                )
        cv = pd.DataFrame(cv_records)
        cv.to_csv(result_dir / "cv_results.csv", index=False)
        pd.concat(cv_predictions).sort_values("row_id").to_csv(
            result_dir / "cv_predictions.csv", index=False
        )
        pd.DataFrame(importance_records).to_csv(
            result_dir / "permutation_importance.csv", index=False
        )
        print("Fitting final workflows using the 800 training observations only", flush=True)
        fitted = fit_models(X_train, y_train, config.seed, config.folds)
        selected = {name: search.best_params_ for name, search in fitted.searches.items()}
        for name, search in fitted.searches.items():
            for index, params in enumerate(search.cv_results_["params"]):
                search_records.append(
                    {
                        "stage": "full_training",
                        "model": name,
                        "params": json.dumps(params, sort_keys=True),
                        "mean_log_loss": -search.cv_results_["mean_test_score"][index],
                        "selected": index == search.best_index_,
                    }
                )
        pd.DataFrame(search_records).to_csv(result_dir / "tuning_results.csv", index=False)
        coefficient_table(fitted.models["logistic"], codebook(root / "data/raw")).to_csv(
            result_dir / "logistic_coefficients.csv", index=False
        )
        # Materialise the final decisions before obtaining holdout predictions.
        write_json(
            result_dir / "selection_locked.json",
            {
                "config": asdict(config),
                "selected": selected,
                "protocol_sha256": sha256(root / "docs/protocol.md"),
                "holdout_used_for_selection": False,
            },
        )
        print("Selection locked. Evaluating the final holdout once.", flush=True)
        X_test, y_test = X.loc[test_ids], y.loc[test_ids]
        predictions = {
            name: model.predict_proba(X_test)[:, 1] for name, model in fitted.models.items()
        }
        held = pd.DataFrame({"row_id": test_ids, "bad_credit": y_test.to_numpy(), **predictions})
        held.to_csv(result_dir / "holdout_predictions.csv", index=False)
        holdout_metrics = {
            name: metric_values(y_test, p, config.threshold) for name, p in predictions.items()
        }
        diagnostic = {name: calibration_diagnostics(y_test, p) for name, p in predictions.items()}
        pd.concat(
            [reliability_bins(y_test, p).assign(model=name) for name, p in predictions.items()]
        ).to_csv(result_dir / "calibration_bins.csv", index=False)
        print(
            f"Computing {config.bootstrap_replicates:,} paired stratified bootstrap replicates",
            flush=True,
        )
        draws, intervals, bootstrap = paired_bootstrap(
            y_test,
            predictions,
            config.bootstrap_replicates,
            config.seed + 30,
        )
        draws.to_csv(result_dir / "bootstrap_draws.csv", index=False)
        intervals.to_csv(result_dir / "bootstrap_intervals.csv", index=False)
    scenarios = []
    for prevalence in [0.05, 0.1, 0.2, 0.3]:
        for source_p in np.linspace(0, 1, 101):
            scenarios.append(
                {
                    "source_probability": source_p,
                    "source_prevalence": float(y_train.mean()),
                    "target_prevalence": prevalence,
                    "adjusted_probability": float(
                        prior_shift(source_p, float(y_train.mean()), prevalence)
                    ),
                }
            )
    pd.DataFrame(scenarios).to_csv(result_dir / "prevalence_scenarios.csv", index=False)
    thresholds = []
    for model, probabilities in predictions.items():
        for threshold in np.linspace(0, 1, 101):
            thresholds.append(
                {
                    "model": model,
                    "threshold": threshold,
                    **metric_values(y_test, probabilities, threshold),
                }
            )
    pd.DataFrame(thresholds).to_csv(result_dir / "threshold_metrics.csv", index=False)
    cv_summary = {}
    for name in fitted.models:
        subset = cv.loc[cv.model == name]
        cv_summary[name] = {
            metric: {"mean": float(subset[metric].mean()), "sd": float(subset[metric].std(ddof=1))}
            for metric in KEY_METRICS
        }
    metrics = {
        "config": asdict(config),
        "data": audit,
        "partitions": {"training": partition["training"], "holdout": partition["holdout"]},
        "holdout": holdout_metrics,
        "cv": cv_summary,
        "calibration": diagnostic,
        "selected_parameters": selected,
        "bootstrap": bootstrap,
        "stability": stability_summary(cv),
    }
    write_json(result_dir / "split_manifest.json", partition)
    write_json(result_dir / "metrics.json", metrics)
    manifest = {
        "started_utc": started_utc,
        "elapsed_seconds": time.perf_counter() - start,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": asdict(config),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "matplotlib",
                "reportlab",
                "pypdf",
                "threadpoolctl",
            ]
        },
        "source_hashes": {
            path.name: sha256(path) for path in (root / "src/credit_risk").glob("*.py")
        },
        "protocol_sha256": sha256(root / "docs/protocol.md"),
        "data_sha256": audit["sha256"],
        "selection_lock_sha256": sha256(result_dir / "selection_locked.json"),
    }
    write_json(result_dir / "run_manifest.json", manifest)
    print(f"Research completed in {manifest['elapsed_seconds']:.1f}s", flush=True)
    return metrics
