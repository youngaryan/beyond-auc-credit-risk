"""Fourteen original research figures, always rebuilt from persisted evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import precision_recall_curve, roc_curve

from credit_risk.calibration import wilson_interval
from credit_risk.data import load_data
from credit_risk.models import MODEL_NAMES

COLORS = {
    "constant": "#7B8794",
    "logistic": "#176B9A",
    "boosting": "#D27627",
    "calibrated": "#6F4C9B",
}
LINESTYLES = {"constant": ":", "logistic": "-", "boosting": "--", "calibrated": "-."}
MARKERS = {"constant": "x", "logistic": "o", "boosting": "s", "calibrated": "D"}
SHORT = {
    "constant": "Constant",
    "logistic": "Logistic",
    "boosting": "Boosting",
    "calibrated": "Calibrated",
}
FIGURE_NAMES = [
    "01_methodology",
    "02_outcome_distribution",
    "03_feature_distributions",
    "04_conditional_frequency",
    "05_roc_curves",
    "06_precision_recall",
    "07_cross_validation",
    "08_reliability",
    "09_probability_distribution",
    "10_proper_scores",
    "11_paired_uncertainty",
    "12_feature_interpretation",
    "13_prevalence_sensitivity",
    "14_threshold_tradeoffs",
]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#A7B0B9",
            "axes.labelcolor": "#263546",
            "text.color": "#263546",
            "grid.color": "#E6E9ED",
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "savefig.dpi": 220,
            "svg.fonttype": "none",
            "svg.hashsalt": "beyond-auc-42",
        }
    )


def save(fig: plt.Figure, directory: Path, number: int) -> None:
    name = FIGURE_NAMES[number - 1]
    fig.savefig(
        directory / f"{name}.png",
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "Beyond AUC"},
    )
    fig.savefig(
        directory / f"{name}.svg", bbox_inches="tight", facecolor="white", metadata={"Date": None}
    )
    plt.close(fig)


def generate_figures(root: Path) -> None:
    style()
    directory = root / "results/figures"
    directory.mkdir(parents=True, exist_ok=True)
    results = root / "results"
    metrics = json.loads((results / "metrics.json").read_text())
    split = json.loads((results / "split_manifest.json").read_text())
    predictions = pd.read_csv(results / "holdout_predictions.csv")
    cv = pd.read_csv(results / "cv_results.csv")
    intervals = pd.read_csv(results / "bootstrap_intervals.csv")
    bins = pd.read_csv(results / "calibration_bins.csv")
    X, y, _ = load_data(root / "data/raw")
    train_X, train_y = X.loc[split["train_row_ids"]], y.loc[split["train_row_ids"]]

    # 1. The design and the holdout boundary.
    fig, ax = plt.subplots(figsize=(10, 3.6), layout="constrained")
    ax.set(xlim=(0, 10), ylim=(0, 3.6))
    ax.axis("off")
    labels = [
        "Public data\n1,000 records",
        "Integrity checks\nCoding + provenance",
        "Stratified split\n800 train / 200 test",
        "Training CV\nFive outer folds",
        "Model selection\nInner 5-fold search",
        "Sigmoid calibration\nCross-fitted searches",
        "Locked holdout\nAll models frozen",
        "Uncertainty\nPaired bootstrap",
        "Interpretation\nEvidence + limitations",
    ]
    positions = [
        (0.1, 2.55),
        (3.55, 2.55),
        (7.0, 2.55),
        (7.0, 1.35),
        (3.55, 1.35),
        (0.1, 1.35),
        (0.1, 0.15),
        (3.55, 0.15),
        (7.0, 0.15),
    ]
    for index, ((x, yy), label) in enumerate(zip(positions, labels, strict=True)):
        ax.add_patch(
            FancyBboxPatch(
                (x, yy),
                2.85,
                0.8,
                boxstyle="round,pad=0.06",
                linewidth=1,
                edgecolor="#176B9A",
                facecolor="#EFF5F9" if index < 6 else "#F4F0F8",
            )
        )
        ax.text(x + 1.425, yy + 0.4, label, ha="center", va="center", fontsize=11)
        if index:
            px, py = positions[index - 1]
            if yy == py:
                sign = 1 if x > px else -1
                start, end = (
                    ((px + 2.95, py + 0.4), (x - 0.08, yy + 0.4))
                    if sign == 1
                    else ((px - 0.08, py + 0.4), (x + 2.95, yy + 0.4))
                )
            else:
                start, end = (px + 1.425, py - 0.07), (x + 1.425, yy + 0.9)
            ax.add_patch(
                FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, color="#526477")
            )
    save(fig, directory, 1)

    # 2. Entire-source balance is an integrity fact, not population prevalence.
    fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
    values = [metrics["data"]["good"], metrics["data"]["bad"]]
    bars = ax.barh(
        ["Good credit", "Bad credit"],
        values,
        color=[COLORS["logistic"], COLORS["boosting"]],
        height=0.55,
    )
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value + 15,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}  ({value / sum(values):.0%})",
            va="center",
        )
    ax.set(
        xlim=(0, 900),
        xlabel="Number of historical credit records",
        title="The 30% bad-credit share is imposed by sampling",
    )
    ax.grid(axis="x")
    save(fig, directory, 2)

    # 3. Training-only distributions; normalise within outcome groups.
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), layout="constrained")
    for ax, feature, label in zip(
        axes,
        ["duration", "amount", "age"],
        ["Duration (months)", "Amount (transformed units)", "Age (years)"],
        strict=True,
    ):
        edges = np.linspace(train_X[feature].min(), train_X[feature].max(), 19)
        for outcome, name, color, linestyle in [
            (0, "Good", COLORS["logistic"], "-"),
            (1, "Bad", COLORS["boosting"], "--"),
        ]:
            ax.hist(
                train_X.loc[train_y == outcome, feature],
                bins=edges,
                density=True,
                histtype="step",
                linewidth=2,
                color=color,
                label=name,
                linestyle=linestyle,
            )
        ax.set(xlabel=label, ylabel="Within-group density", title=feature.title())
        ax.grid(axis="y")
    axes[0].legend()
    save(fig, directory, 3)

    # 4. Conditional frequencies with observed denominators and uncertainty.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), layout="constrained")
    category_labels = {
        "status": [
            "No checking\naccount",
            "Balance\n< 0 DM",
            "Balance\n0-<200 DM",
            ">=200 DM /\nsalary history",
        ],
        "credit_history": [
            "Past\npayment delay",
            "Critical /\nother credits",
            "No credits /\nall repaid",
            "Existing\nrepaid duly",
            "All at bank\nrepaid duly",
        ],
    }
    for ax, feature in zip(axes, category_labels, strict=True):
        groups = train_y.groupby(train_X[feature]).agg(["sum", "count", "mean"])
        x = np.arange(len(groups))
        uncertainty = np.array(
            [wilson_interval(int(row["sum"]), int(row["count"])) for _, row in groups.iterrows()]
        )
        ax.bar(x, groups["mean"], color=COLORS["logistic"], width=0.6, alpha=0.85)
        ax.errorbar(
            x,
            groups["mean"],
            yerr=[groups["mean"] - uncertainty[:, 0], uncertainty[:, 1] - groups["mean"]],
            fmt="none",
            color="#263546",
            capsize=3,
        )
        for i, (_, row) in enumerate(groups.iterrows()):
            ax.text(i, uncertainty[i, 1] + 0.04, f"n={int(row['count'])}", ha="center", fontsize=9)
        ax.set(
            xticks=x,
            xticklabels=category_labels[feature],
            ylim=(0, 1),
            ylabel="Observed bad-credit fraction",
            title=feature.replace("_", " ").title(),
        )
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y")
    save(fig, directory, 4)

    # 5-6. Discrimination at the single locked holdout.
    fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    for name in ["logistic", "boosting", "calibrated"]:
        fpr, tpr, _ = roc_curve(predictions.bad_credit, predictions[name])
        ax.plot(
            fpr,
            tpr,
            color=COLORS[name],
            linestyle=LINESTYLES[name],
            linewidth=2,
            label=f"{SHORT[name]}: AUC {metrics['holdout'][name]['roc_auc']:.3f}",
        )
    ax.plot([0, 1], [0, 1], color=COLORS["constant"], linestyle=":", label="Constant: AUC 0.500")
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="False positive rate (1 - specificity)",
        ylabel="True positive rate (sensitivity)",
    )
    ax.legend(loc="lower right")
    ax.grid()
    save(fig, directory, 5)
    fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    for name in ["logistic", "boosting", "calibrated"]:
        precision, recall, _ = precision_recall_curve(predictions.bad_credit, predictions[name])
        ax.step(
            recall,
            precision,
            where="post",
            color=COLORS[name],
            linestyle=LINESTYLES[name],
            linewidth=2,
            label=f"{SHORT[name]}: AP {metrics['holdout'][name]['average_precision']:.3f}",
        )
    ax.axhline(
        predictions.bad_credit.mean(),
        color=COLORS["constant"],
        linestyle=":",
        label="Sample prevalence: 0.300",
    )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="Recall of bad-credit outcomes",
        ylabel="Precision among flagged records",
    )
    ax.legend(loc="lower left")
    ax.grid()
    save(fig, directory, 6)

    # 7. Display every outer-fold result, not a bar of means alone.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for ax, metric, label in zip(
        axes,
        ["roc_auc", "log_loss"],
        ["ROC-AUC (higher is better)", "Log loss (lower is better)"],
        strict=True,
    ):
        for name in MODEL_NAMES:
            subset = cv.loc[cv.model == name]
            ax.plot(
                subset.fold,
                subset[metric],
                color=COLORS[name],
                linestyle=LINESTYLES[name],
                marker=MARKERS[name],
                label=SHORT[name],
            )
        ax.set(xticks=range(1, 6), xlabel="Outer training-only fold", ylabel=label, ylim=(0, 1))
        ax.grid(axis="y")
    axes[0].legend(loc="lower left", ncols=2)
    save(fig, directory, 7)

    # 8. Separate panels keep uncertainty legible; same five fixed bins.
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), layout="constrained")
    for ax, name in zip(axes, ["logistic", "boosting", "calibrated"], strict=True):
        subset = bins.loc[(bins.model == name) & (bins.n > 0)]
        ax.plot([0, 1], [0, 1], color="#7B8794", linestyle=":")
        ax.errorbar(
            subset.mean_probability,
            subset.observed_frequency,
            yerr=[
                subset.observed_frequency - subset.wilson_lower,
                subset.wilson_upper - subset.observed_frequency,
            ],
            color=COLORS[name],
            marker=MARKERS[name],
            linestyle=LINESTYLES[name],
            capsize=3,
        )
        for _, row in subset.iterrows():
            ax.annotate(
                f"n={int(row.n)}",
                (row.mean_probability, row.observed_frequency),
                xytext=(3, 7 if row.observed_frequency < 0.85 else -14),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1.04),
            xlabel="Mean predicted probability",
            ylabel="Observed bad-credit fraction",
            title=SHORT[name],
        )
        ax.grid()
    save(fig, directory, 8)

    # 9. Probability mass explains which parts of each reliability curve are weak.
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), layout="constrained", sharey=True)
    for ax, name in zip(axes, ["logistic", "boosting", "calibrated"], strict=True):
        ax.hist(
            predictions[name],
            bins=np.linspace(0, 1, 11),
            color=COLORS[name],
            alpha=0.85,
            edgecolor="white",
        )
        ax.axvline(0.5, color="#263546", linestyle=":", linewidth=1.2)
        ax.set(xlim=(0, 1), xlabel="Predicted bad-credit probability", title=SHORT[name])
        ax.grid(axis="y")
    axes[0].set_ylabel("Holdout records per bin")
    save(fig, directory, 9)

    # 10. Proper scores with their conditional bootstrap CIs; zero baseline.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for ax, metric, title in zip(
        axes, ["brier", "log_loss"], ["Brier score", "Log loss"], strict=True
    ):
        subset = intervals.loc[
            (intervals.kind == "model") & (intervals.metric == metric)
        ].set_index("comparison")
        for pos, name in enumerate(MODEL_NAMES):
            row = subset.loc[name]
            ax.hlines(pos, row.lower, row.upper, color=COLORS[name])
            ax.plot([row.lower, row.upper], [pos, pos], "|", color=COLORS[name])
            ax.plot(row.estimate, pos, MARKERS[name], color=COLORS[name], markersize=7)
        ax.set(
            yticks=range(4),
            yticklabels=[SHORT[n] for n in MODEL_NAMES],
            xlim=(0, max(subset.upper) * 1.12),
            xlabel=f"{title} (lower is better)",
        )
        ax.invert_yaxis()
        ax.grid(axis="x")
    save(fig, directory, 10)

    # 11. Paired contrasts answer whether differences are distinguishable from zero.
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
    contrasts = ["boosting_minus_logistic", "calibrated_minus_boosting"]
    for ax, metric, title in zip(
        axes,
        ["roc_auc", "brier", "log_loss"],
        ["AUC difference", "Brier difference", "Log-loss difference"],
        strict=True,
    ):
        subset = intervals.loc[
            (intervals.kind == "difference") & (intervals.metric == metric)
        ].set_index("comparison")
        for pos, comparison in enumerate(contrasts):
            row = subset.loc[comparison]
            color = COLORS["boosting" if pos == 0 else "calibrated"]
            ax.hlines(pos, row.lower, row.upper, color=color)
            ax.plot([row.lower, row.upper], [pos, pos], "|", color=color)
            ax.plot(row.estimate, pos, "o", color=color)
        ax.axvline(0, color="#7B8794", linestyle=":")
        bound = max(abs(subset.lower).max(), abs(subset.upper).max()) * 1.18
        ax.set(
            xlim=(-max(bound, 0.001), max(bound, 0.001)),
            ylim=(-0.6, 1.6),
            yticks=[0, 1],
            yticklabels=["Boost - logistic", "Calibrated - raw"]
            if metric == "roc_auc"
            else ["", ""],
            xlabel=title,
        )
        ax.grid(axis="x")
    save(fig, directory, 11)

    # 12. Conditional associations, with reference codes disclosed in the caption/CSV.
    coefficients = pd.read_csv(results / "logistic_coefficients.csv")
    top = coefficients.loc[coefficients.coefficient.abs().nlargest(7).index].sort_values(
        "coefficient"
    )
    importance = (
        pd.read_csv(results / "permutation_importance.csv")
        .groupby("feature")
        .log_loss_increase.agg(["mean", "std"])
    )
    top_importance = importance.nlargest(7, "mean").sort_values("mean")
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.2), layout="constrained", gridspec_kw={"width_ratios": [1.05, 1]}
    )
    axes[0].barh(top.feature.str.replace("_", " "), top.coefficient, color=COLORS["logistic"])
    axes[0].axvline(0, color="#7B8794", linewidth=0.8)
    axes[0].set(
        xlabel="Regularised log-odds coefficient", title="Logistic: largest absolute coefficients"
    )
    axes[0].tick_params(axis="y", labelsize=9)
    axes[1].errorbar(
        top_importance["mean"],
        np.arange(len(top_importance)),
        xerr=top_importance["std"],
        fmt="s",
        color=COLORS["boosting"],
        capsize=3,
    )
    axes[1].axvline(0, color="#7B8794", linewidth=0.8)
    axes[1].set(
        yticks=np.arange(len(top_importance)),
        yticklabels=top_importance.index.str.replace("_", " "),
        xlabel="Validation log-loss increase on shuffling",
        title="Boosting: raw-feature importance",
    )
    axes[1].tick_params(axis="y", labelsize=9)
    for ax in axes:
        ax.grid(axis="x")
    save(fig, directory, 12)

    # 13. Hypothetical label shift is a mathematical scenario, not measured performance.
    scenarios = pd.read_csv(results / "prevalence_scenarios.csv")
    fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    for (prevalence, rows), color, linestyle in zip(
        scenarios.groupby("target_prevalence"),
        ["#176B9A", "#6F4C9B", "#D27627", "#7B8794"],
        ["-", "--", "-.", ":"],
        strict=True,
    ):
        ax.plot(
            rows.source_probability,
            rows.adjusted_probability,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=f"Hypothetical target prevalence: {prevalence:.0%}",
        )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Probability under 30% source prevalence",
        ylabel="Adjusted probability under assumed target prevalence",
    )
    ax.text(
        0.04,
        0.96,
        "Assumptions: stable P(X|Y); calibrated source posterior",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    ax.legend(loc="lower right")
    ax.grid()
    save(fig, directory, 13)

    # 14. Threshold = 0.5 was chosen in the protocol; curves do not select a policy.
    thresholds = pd.read_csv(results / "threshold_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for ax, metric, title in zip(
        axes, ["sensitivity", "specificity"], ["Sensitivity / recall", "Specificity"], strict=True
    ):
        for name in ["logistic", "boosting", "calibrated"]:
            rows = thresholds.loc[thresholds.model == name]
            ax.plot(
                rows.threshold,
                rows[metric],
                color=COLORS[name],
                linestyle=LINESTYLES[name],
                linewidth=2,
                label=SHORT[name],
            )
        ax.axvline(0.5, color="#7B8794", linestyle=":", label="Illustrative threshold 0.5")
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Probability threshold", ylabel=title)
        ax.grid()
    axes[0].legend(loc="lower left", fontsize=9)
    save(fig, directory, 14)
    print(f"Generated {len(FIGURE_NAMES)} figures in PNG and SVG", flush=True)
