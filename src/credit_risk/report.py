"""One structured narrative renders to Markdown, a paginated PDF and portfolio copy.

All numerical findings are read from saved experiment outputs. This module never
fits a model or changes a prediction. Tables and prose share the same values.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from credit_risk.data import sha256
from credit_risk.experiment import write_json
from credit_risk.models import MODEL_NAMES
from credit_risk.plotting import FIGURE_NAMES, SHORT
from credit_risk.robustness import prior_shift

TITLE = "Beyond AUC: Credit Risk Probability Modelling, Calibration, and Model Validation"
SUBTITLE = "An Empirical Comparison of Interpretable Statistical Models and Machine Learning Under Sampling Bias and Limited Data"

REFERENCES = [
    (
        "UCI Machine Learning Repository (2019). South German Credit [Dataset]. CC BY 4.0.",
        "https://doi.org/10.24432/C5X89F",
    ),
    (
        "Grömping, U. (2019). South German Credit Data: Correcting a Widely Used Data Set. Report 4/2019, Beuth University of Applied Sciences Berlin.",
        "https://www1.beuth-hochschule.de/FB_II/reports/Report-2019-004.pdf",
    ),
    (
        "Cox, D. R. (1958). The Regression Analysis of Binary Sequences. Journal of the Royal Statistical Society, Series B, 20(2), 215-232.",
        "https://doi.org/10.1111/j.2517-6161.1958.tb00292.x",
    ),
    (
        "Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics, 29(5), 1189-1232.",
        "https://doi.org/10.1214/aos/1013203451",
    ),
    (
        "Niculescu-Mizil, A., and Caruana, R. (2005). Predicting Good Probabilities With Supervised Learning. Proceedings of the 22nd International Conference on Machine Learning.",
        "https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf",
    ),
    (
        "Gneiting, T., and Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. Journal of the American Statistical Association, 102(477), 359-378.",
        "https://doi.org/10.1198/016214506000001437",
    ),
    (
        "Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife. Annals of Statistics, 7(1), 1-26.",
        "https://doi.org/10.1214/aos/1176344552",
    ),
    (
        "Saerens, M., Latinne, P., and Decaestecker, C. (2002). Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure. Neural Computation, 14(1), 21-41.",
        "https://doi.org/10.1162/089976602753284446",
    ),
    (
        "scikit-learn contributors. Probability calibration and CalibratedClassifierCV documentation. Accessed 27 October 2026; executed version recorded in results/run_manifest.json.",
        "https://scikit-learn.org/stable/modules/calibration.html",
    ),
    (
        "scikit-learn contributors. Metrics and scoring: ROC-AUC, average precision, Brier score and log loss. Accessed 27 October 2026.",
        "https://scikit-learn.org/stable/modules/model_evaluation.html",
    ),
]


@dataclass
class Block:
    kind: str
    text: str = ""
    rows: list[list[str]] = field(default_factory=list)
    figure: int = 0
    max_height: float = 230


@dataclass
class Page:
    title: str
    blocks: list[Block]
    cover: bool = False


def paragraph(text: str) -> Block:
    return Block("paragraph", text)


def subheading(text: str) -> Block:
    return Block("subheading", text)


def figure(number: int, caption: str, max_height: float = 230) -> Block:
    return Block("figure", caption, figure=number, max_height=max_height)


def table(caption: str, rows: list[list[str]]) -> Block:
    return Block("table", caption, rows)


def f(value: float | None, digits: int = 3) -> str:
    return "not defined" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def narrative(root: Path) -> tuple[list[Page], dict]:
    results = root / "results"
    m = json.loads((results / "metrics.json").read_text())
    eda = json.loads((results / "training_eda.json").read_text())
    intervals = pd.read_csv(results / "bootstrap_intervals.csv")
    coefficients = pd.read_csv(results / "logistic_coefficients.csv")
    importance = (
        pd.read_csv(results / "permutation_importance.csv")
        .groupby("feature")
        .log_loss_increase.mean()
        .sort_values(ascending=False)
    )
    calibration_bins = pd.read_csv(results / "calibration_bins.csv")
    h, cv, c, stability = m["holdout"], m["cv"], m["calibration"], m["stability"]
    config, data = m["config"], m["data"]

    def interval(comparison: str, metric: str, digits: int = 3) -> str:
        row = intervals.loc[
            (intervals.comparison == comparison) & (intervals.metric == metric)
        ].iloc[0]
        return f"[{row.lower:.{digits}f}, {row.upper:.{digits}f}]"

    auc_delta = h["boosting"]["roc_auc"] - h["logistic"]["roc_auc"]
    brier_delta = h["calibrated"]["brier"] - h["boosting"]["brier"]
    loss_delta = h["calibrated"]["log_loss"] - h["boosting"]["log_loss"]
    auc_ci = intervals.loc[
        (intervals.comparison == "boosting_minus_logistic") & (intervals.metric == "roc_auc")
    ].iloc[0]
    conditional_evidence = (
        "The conditional interval excludes zero in favour of boosting"
        if auc_ci.lower > 0
        else "The conditional interval does not establish a positive boosting advantage"
    )
    calibration_conclusion = (
        "a consistent improvement is not established"
        if brier_delta * loss_delta <= 0
        else "the directions agree, but the paired intervals still determine uncertainty"
    )
    summary = (
        f"Boosting achieves holdout ROC-AUC {f(h['boosting']['roc_auc'])} versus {f(h['logistic']['roc_auc'])} "
        f"for logistic regression, a difference of {signed(auc_delta)} (paired 95% CI "
        f"{interval('boosting_minus_logistic', 'roc_auc')}). Yet boosting leads in only "
        f"{stability['boosting_auc_wins']} of {stability['n_folds']} training-only outer folds. "
        "The evidence supports a holdout-specific advantage, not a stable general preference for complexity."
    )
    abstract = (
        "Credit-risk decisions require both useful risk rankings and credible outcome probabilities. "
        "This study asks whether gradient boosting earns its additional complexity over regularised logistic "
        f"regression on the corrected South German Credit dataset. We compare a constant-prevalence baseline, "
        f"logistic regression, gradient boosting and sigmoid-calibrated boosting on {data['n_rows']:,} historical "
        f"records. A stratified {m['partitions']['training']['n']}/{m['partitions']['holdout']['n']} split reserves "
        "a final holdout; nested training-only cross-validation separates preprocessing, tuning and calibration. "
        f"Evaluation includes discrimination, proper scoring rules, reliability diagrams and {config['bootstrap_replicates']:,} "
        f"paired stratified bootstrap resamples. Holdout AUC is {f(h['logistic']['roc_auc'])} for logistic regression "
        f"and {f(h['boosting']['roc_auc'])} for boosting; the difference is {signed(auc_delta)} with conditional "
        f"95% interval {interval('boosting_minus_logistic', 'roc_auc')}. However, their mean outer-fold AUCs "
        f"are {f(cv['logistic']['roc_auc']['mean'])} and {f(cv['boosting']['roc_auc']['mean'])}, respectively. "
        f"Calibration changes holdout Brier score by {signed(brier_delta, 5)} and log loss by {signed(loss_delta, 5)}; "
        "both paired intervals include zero. These findings caution against selecting model complexity from a "
        "single favourable split. The deliberately enriched sample, uncertain outcome horizon and 1970s origin "
        "prevent interpreting its calibrated probabilities as contemporary portfolio default probabilities. "
        "This is a reproducible empirical model-validation exercise, with evidence for predictive signal but "
        "mixed evidence for a general boosting advantage and no demonstrated calibration benefit."
    )
    pages = [
        Page(
            TITLE,
            [
                Block("subtitle", SUBTITLE),
                Block("eyebrow", "QUANTITATIVE RESEARCH  /  INDEPENDENT PORTFOLIO STUDY"),
                paragraph("Aryan Golbaghi  |  27 October 2026"),
                subheading("1. Abstract"),
                paragraph(abstract),
                Block(
                    "callout",
                    "A favourable holdout is evidence to examine, not a substitute for stable validation.",
                ),
                paragraph(
                    "Research scope: historical bad-credit outcomes, probability quality and validation uncertainty. "
                    "Prepared as an independent professional portfolio project; no bank sponsorship, affiliation or approval is implied."
                ),
            ],
            cover=True,
        )
    ]

    pages.append(
        Page(
            "2. Introduction and motivation",
            [
                paragraph(
                    "Credit risk is the risk that a borrower fails to meet contractual obligations. A probability of default "
                    "model gives a probability for a specified event over a stated horizon, conditional on information available "
                    "at prediction time. Such probabilities can inform expected-loss analysis when combined with exposure and "
                    "loss severity assumptions. Those ingredients are not established by this historical benchmark."
                ),
                paragraph(
                    "Ranking and probability estimation answer different questions. ROC-AUC asks how often a randomly selected "
                    "bad outcome receives a higher score than a good outcome, with half credit for ties. It is unchanged by a "
                    "strictly increasing score transformation. Calibration asks whether, among observations assigned a probability "
                    "near p, the outcome occurs about p of the time. A strong ranking can therefore coexist with systematically "
                    "wrong absolute risk estimates. This distinction matters whenever probabilities enter valuation, loss estimates "
                    "or decisions [6,10]."
                ),
                paragraph(
                    "The study compares a transparent statistical benchmark with shallow-tree gradient boosting, rather than "
                    "assuming flexibility improves a decision system. The career context is graduate quantitative analytics, "
                    "including Barclays Risk Quantitative Analytics roles; this work was not conducted for or endorsed by Barclays."
                ),
                subheading("3. Research questions and prespecified hypotheses"),
                table(
                    "Hypothesis framework: declared before model fitting in docs/protocol.md.",
                    [
                        ["Hypothesis", "Concern and planned evidence"],
                        [
                            "H1: discrimination",
                            "Compare trained AUC with 0.5 in outer CV and the locked holdout. A lower interval above 0.5 supports predictive signal.",
                        ],
                        [
                            "H2: complexity",
                            "Compare boosting minus logistic with paired intervals and fold-level signs. Stable positive evidence strengthens a complexity case.",
                        ],
                        [
                            "H3: calibration",
                            "Contrast ranking with reliability diagnostics and scoring rules. Higher AUC with weaker calibration illustrates the distinction.",
                        ],
                        [
                            "H4: intervention",
                            "Compare the same boosting model before and after sigmoid calibration. Consistent lower proper scores would support a benefit.",
                        ],
                        [
                            "H5: sampling",
                            "Audit selection and explore hypothetical prior shifts. Population PD calibration cannot be established without external information.",
                        ],
                    ],
                ),
                paragraph(
                    "These are hypotheses, not target outcomes. The protocol does not specify an economically material improvement "
                    "because no decision costs or representative portfolio are available. Contradictory and inconclusive findings "
                    "are retained. The local protocol is auditable but was not externally preregistered."
                ),
            ],
        )
    )

    numeric_rows = [["Training variable", "Median", "Mean", "Range", "IQR flags"]]
    for feature in ["duration", "amount", "age"]:
        values = eda["numeric"][feature]
        numeric_rows.append(
            [
                feature,
                f(values["50%"], 1),
                f(values["mean"], 1),
                f"{values['min']:.0f}-{values['max']:.0f}",
                str(int(values["iqr_flag_count"])),
            ]
        )
    pages.append(
        Page(
            "4. Dataset, provenance and target meaning",
            [
                paragraph(
                    f"The corrected UCI South German Credit release contains {data['n_rows']:,} records and "
                    f"{data['n_features']} predictors. We retain the original archive, codebook and R reader, with SHA-256 "
                    "checksums. The landing-page count of 21 includes the target; the parsed file confirms 20 predictors. "
                    "The source and correction are attributed under CC BY 4.0 [1,2]."
                ),
                paragraph(
                    "The codebook defines kredit=0 as bad and kredit=1 as good. The loader explicitly constructs "
                    "Y=1[kredit=0], then renames the predictors. A test checks every label against the original file. "
                    f"The audit finds {data['missing_cells']} missing cells, {data['duplicate_records']} duplicate records "
                    f"and {data['duplicate_feature_vectors']} duplicate predictor vectors. No rows are removed."
                ),
                figure(
                    2,
                    "Figure 2. Full-source outcome counts and proportions. These are integrity statistics; the enriched 30% "
                    "bad-credit fraction is not an estimate of a bank's default rate.",
                    180,
                ),
                table(
                    "Table 1. Training-only descriptive statistics. IQR flags use the 1.5-IQR rule and are retained.",
                    numeric_rows,
                ),
                paragraph(
                    "Duration is in months and age in years. The amount is an unknown monotonic transformation of historical "
                    "credit amounts: its numerical units cannot support currency or exposure calculations. All 17 other "
                    "predictors are categorical or discretised quantities. One-hot encoding avoids imposing an unsupported "
                    "equal-distance relationship between integer codes."
                ),
                Block(
                    "callout",
                    "Four distinct quantities: observed bad-credit outcome; modern horizon-specific default event; "
                    "study-sample calibrated probability; representative-portfolio probability. This study estimates only the first "
                    "outcome under the study's sampling design.",
                ),
            ],
        )
    )

    pages.append(
        Page(
            "4. Dataset: exploratory evidence",
            [
                figure(
                    3,
                    "Figure 3. Within-outcome densities of duration, transformed amount and age in the training partition. "
                    "Common bin edges permit comparison; density normalisation prevents the majority class from dominating.",
                    205,
                ),
                paragraph(
                    f"Training medians are {eda['numeric']['duration']['50%']:.0f} months for duration, "
                    f"{eda['numeric']['amount']['50%']:.0f} transformed units for amount and "
                    f"{eda['numeric']['age']['50%']:.0f} years for age. The distributions differ in scale and have uneven tails. "
                    "Scaling benefits penalised logistic fitting; the tree model does not require it but uses the same pipeline. "
                    "Outlier flags do not establish errors, and were not used to trim either partition."
                ),
                figure(
                    4,
                    "Figure 4. Training-only bad-credit frequencies by checking-account status and credit history, with counts "
                    "and descriptive 95% Wilson intervals. The intervals describe binomial sampling variation, not causal effects.",
                    230,
                ),
                paragraph(
                    "The conditional patterns motivate examining nonlinear and categorical effects, while their overlapping "
                    "intervals discourage overinterpreting sparse groups. These associations may reflect lending policy, "
                    "selection and other predictors. They do not identify effects of changing a borrower's characteristics. "
                    "Exploratory associations use the 800 training observations only; holdout outcomes did not guide feature choices."
                ),
            ],
        )
    )

    pages.append(
        Page(
            "5. Methodology: keep the holdout isolated",
            [
                figure(
                    1,
                    "Figure 1. Prespecified research workflow. Model selection and calibration are nested within training. "
                    "The sequence is schematic: each calibration fold runs its own complete hyperparameter search before "
                    "the final full-training model is fitted.",
                    200,
                ),
                table(
                    "Table 2. Major partitions; both outcome classes occur throughout the design.",
                    [
                        ["Partition", "Total", "Good", "Bad", "Purpose"],
                        [
                            "Source",
                            str(data["n_rows"]),
                            str(data["good"]),
                            str(data["bad"]),
                            "Integrity checks",
                        ],
                        ["Training", "800", "560", "240", "EDA, selection, calibration"],
                        ["Final holdout", "200", "140", "60", "Frozen-model evaluation"],
                        ["Each outer fit", "640", "448", "192", "Fit complete workflow"],
                        ["Each outer validation", "160", "112", "48", "Training-only stability"],
                    ],
                ),
                paragraph(
                    f"The stratified split uses seed {config['seed']}; row IDs are retained in split_manifest.json. "
                    "Outer five-fold CV evaluates the complete learning procedure on the training set. Each outer fit runs "
                    "a separate five-fold grid search scored by mean log loss. This prevents choosing hyperparameters on the "
                    "same observations later used to estimate the workflow's outer-fold performance."
                ),
                paragraph(
                    "For calibrated boosting, the cross-fitted estimator is GridSearchCV itself. Each calibration observation "
                    "is excluded from its score's imputation, scaling, model fitting and hyperparameter selection. "
                    "The sigmoid is then fitted to these out-of-fold scores, and the base search is refitted on all available "
                    "training data. With ensemble=False, raw and calibrated predictions share that exact fitted base model [9]."
                ),
                paragraph(
                    "Within an outer fit, calibration folds use 512 fit / 128 calibration rows, with about 154 / 38 bad outcomes. "
                    "Their inner tuning fits have 409-410 observations. The final calibration cross-fit uses 640 / 160 rows. "
                    "This costs additional fitting but avoids optimistic calibration scores from global hyperparameter selection. "
                    "It still assumes the score distribution transfers from smaller fitted models to the full-training model."
                ),
            ],
        )
    )

    pages.append(
        Page(
            "5. Methodology: models and probability quality",
            [
                subheading("Statistical baseline and flexible alternative"),
                paragraph(
                    "Logistic regression models a linear predictor on the log-odds scale [3]. Numerical effects are expressed "
                    "per training standard deviation; each categorical effect compares a category with the first documented "
                    "code. L2 shrinkage reduces instability but biases coefficients. Additive linear log-odds and the chosen "
                    "reference coding constrain interpretation; neither standardised coefficients nor odds ratios are causal."
                ),
                Block("equation", r"p(x)=\frac{1}{1+\exp[-(\beta_0+\beta^T x)]}"),
                paragraph(
                    "Gradient boosting adds shallow trees to reduce binomial deviance, capturing selected nonlinearities and "
                    "interactions [4]. A deliberately small search limits adaptive overfitting. The constant model predicts "
                    "training prevalence. Class weighting and synthetic oversampling are omitted because they would change "
                    "the probability target. Median/mode imputation, scaling and encoding occur within every pipeline fit."
                ),
                Block(
                    "equation",
                    r"F_M(x)=F_0+\eta\sum_{m=1}^{M}h_m(x),\qquad p(x)=\{1+e^{-F_M(x)}\}^{-1}",
                ),
                table(
                    "Table 3. Prespecified searches and full-training selections; selection criterion is log loss.",
                    [
                        ["Workflow", "Search", "Selected on training"],
                        [
                            "Logistic",
                            "C: 0.1, 1, 10",
                            f"C = {m['selected_parameters']['logistic']['model__C']:g}",
                        ],
                        [
                            "Boosting",
                            "Depth: 1, 2; rate: 0.05, 0.1",
                            f"Depth {m['selected_parameters']['boosting']['model__max_depth']}; rate {m['selected_parameters']['boosting']['model__learning_rate']:g}",
                        ],
                        [
                            "Common boosting settings",
                            "120 trees; minimum leaf 15",
                            "No additional search",
                        ],
                        ["Calibration", "Sigmoid; five cross-fit folds", "Fixed a priori"],
                    ],
                ),
                paragraph(
                    "Sigmoid calibration is a two-parameter mapping of scores to probabilities. It is less flexible than isotonic "
                    "regression, a deliberate variance trade-off for this small sample [5]. The fitted mapping is monotone here, "
                    "so it changes probability values and thresholds while preserving AUC and average precision."
                ),
                Block(
                    "equation",
                    r"\mathrm{BS}=\frac{1}{n}\sum_i(p_i-y_i)^2,\quad\mathrm{LL}=-\frac{1}{n}\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]",
                ),
                paragraph(
                    "Both scores reward honest probabilities in expectation [6]. Brier score measures squared probability error; "
                    "log loss penalises confident mistakes strongly. Both combine calibration and discrimination information. "
                    "AUC and average precision describe ranking; AP is the recall-weighted average of precisions, not trapezoidal "
                    "PR area. All metrics refer to the study sample. The threshold of 0.5 is illustrative, fixed before evaluation."
                ),
            ],
        )
    )

    result_rows = [["Model", "AUC", "AP", "Brier", "Log loss"]]
    for name in MODEL_NAMES:
        result_rows.append(
            [
                SHORT[name],
                *[
                    f(h[name][metric])
                    for metric in ["roc_auc", "average_precision", "brier", "log_loss"]
                ],
            ]
        )
    pages.append(
        Page(
            "6. Experimental results: locked holdout",
            [
                table(
                    "Table 4. Final holdout performance, n=200 (60 bad). Higher AUC/AP and lower Brier/log loss are preferred.",
                    result_rows,
                ),
                paragraph(
                    f"The constant baseline has AUC {f(h['constant']['roc_auc'])}, AP {f(h['constant']['average_precision'])} "
                    f"and Brier score {f(h['constant']['brier'])}. Both trained families contain predictive information. "
                    f"Logistic AUC is {f(h['logistic']['roc_auc'])} with 95% interval {interval('logistic', 'roc_auc')}; "
                    f"boosting AUC is {f(h['boosting']['roc_auc'])} with interval {interval('boosting', 'roc_auc')}. "
                    "These intervals describe repeated sampling of this fixed holdout case mix, not refitting uncertainty."
                ),
                figure(
                    5,
                    "Figure 5. ROC curves for the locked holdout. The calibrated and raw boosting curves overlap because "
                    "the fitted sigmoid preserves the ranking. AUC describes pairwise ordering, not absolute probability accuracy.",
                    275,
                ),
                paragraph(
                    f"The boosting-logistic AUC difference is {signed(auc_delta)}, with paired interval "
                    f"{interval('boosting_minus_logistic', 'roc_auc')}. {conditional_evidence}. This is useful evidence "
                    "for these frozen models on this partition, but training-fold instability must also be considered. "
                    "Overlapping marginal AUC intervals are not used as a test of the paired difference."
                ),
                paragraph(
                    "No model was selected, retuned or recalibrated after observing these results. The selected hyperparameters "
                    "and protocol hash were written to selection_locked.json before any holdout prediction. The final table "
                    "reports every prespecified configuration, including the constant and calibrated models."
                ),
            ],
        )
    )

    cv_rows = [["Model", "Outer AUC mean (SD)", "Outer log loss mean (SD)"]]
    for name in MODEL_NAMES:
        cv_rows.append(
            [
                SHORT[name],
                f"{f(cv[name]['roc_auc']['mean'])} ({f(cv[name]['roc_auc']['sd'])})",
                f"{f(cv[name]['log_loss']['mean'])} ({f(cv[name]['log_loss']['sd'])})",
            ]
        )
    pages.append(
        Page(
            "6. Results: precision and validation stability",
            [
                figure(
                    6,
                    "Figure 6. Holdout precision-recall curves. AP depends on prevalence; the 0.300 reference applies only "
                    "to this enriched sample. Calibrated and raw boosting have identical ranking metrics.",
                    205,
                ),
                paragraph(
                    f"Holdout AP is {f(h['logistic']['average_precision'])} for logistic and "
                    f"{f(h['boosting']['average_precision'])} for boosting, a difference of "
                    f"{signed(h['boosting']['average_precision'] - h['logistic']['average_precision'])}. "
                    "This advantage does not imply the same precision in a less enriched portfolio."
                ),
                figure(
                    7,
                    "Figure 7. Every training-only outer-fold AUC and log loss. The shared axis scales and constant baseline "
                    "show the magnitude of variation. Folds reuse overlapping training data and are not independent experiments.",
                    190,
                ),
                table(
                    "Table 5. Outer-fold mean and sample SD across five folds. All four metric summaries are saved in metrics.json.",
                    cv_rows,
                ),
                paragraph(
                    f"Boosting wins AUC in {stability['boosting_auc_wins']}/{stability['n_folds']} folds; the fold differences "
                    f"range from {signed(stability['auc_difference_min'])} to {signed(stability['auc_difference_max'])}. "
                    "The contrast with the holdout is direct evidence that a single split can change the preferred model."
                ),
            ],
        )
    )

    diagnostic_rows = [["Model", "Joint intercept (SE)", "Slope (SE)", "Offset intercept", "ECE"]]
    for name in ["logistic", "boosting", "calibrated"]:
        diagnostic_rows.append(
            [
                SHORT[name],
                f"{f(c[name]['intercept'])} ({f(c[name]['intercept_se'])})",
                f"{f(c[name]['slope'])} ({f(c[name]['slope_se'])})",
                f(c[name]["calibration_in_large"]),
                f(c[name]["ece_5_bins"]),
            ]
        )
    pages.append(
        Page(
            "6. Results: reliability and calibration",
            [
                figure(
                    8,
                    "Figure 8. Reliability in five fixed probability bins, with counts and descriptive Wilson 95% intervals. "
                    "The diagonal is ideal calibration; intervals ignore fitting and within-bin heterogeneity.",
                    235,
                ),
                table(
                    "Table 6. Evaluation-only calibration diagnostics. ECE uses the same five bins; lower is descriptive, not definitive.",
                    diagnostic_rows,
                ),
                paragraph(
                    "We estimate logit P(Y=1) = a + b logit(p) diagnostically on the holdout. Ideal joint intercept/slope "
                    "are 0/1. A slope below one can indicate overly dispersed predictions; above one can indicate insufficient "
                    "spread. The offset-only intercept fixes b=1 and measures calibration-in-the-large, a different quantity. "
                    "These diagnostic fits never modify the reported predictions. Standard errors use observed Fisher information "
                    "and a large-sample approximation, which is imperfect with only 60 bad outcomes."
                ),
                paragraph(
                    f"The raw boosting slope is {f(c['boosting']['slope'])} (SE {f(c['boosting']['slope_se'])}); "
                    f"after calibration it is {f(c['calibrated']['slope'])} (SE {f(c['calibrated']['slope_se'])}). "
                    f"Logistic ECE is {f(c['logistic']['ece_5_bins'])}, compared with {f(c['boosting']['ece_5_bins'])} "
                    "for boosting. Thus the model with higher holdout AUC need not look better on a binned calibration "
                    "diagnostic. The near-ideal slopes do not prove exact reliability across individual risk ranges."
                ),
                paragraph(
                    "The constant model has no identifiable calibration slope. Its pooled frequency can match prevalence "
                    "while its ranking is uninformative, illustrating why calibration alone is insufficient. ECE also depends "
                    "on bin boundaries: a small ECE is not a guarantee of accurate individual probabilities."
                ),
            ],
        )
    )

    tail_counts = {
        name: int(
            calibration_bins.loc[
                (calibration_bins.model == name) & (calibration_bins.left >= 0.8), "n"
            ].sum()
        )
        for name in ["logistic", "boosting", "calibrated"]
    }
    pages.append(
        Page(
            "6. Results: probability mass and scoring rules",
            [
                figure(
                    9,
                    "Figure 9. Holdout probability distributions in ten bins. The dotted line is the fixed illustrative "
                    "threshold; histogram bins are finer than the five-bin reliability diagnostic.",
                    180,
                ),
                paragraph(
                    "Sparse high-probability regions constrain reliability assessment. The number of predictions at or above "
                    f"0.8 is {tail_counts['logistic']} for logistic, {tail_counts['boosting']} for boosting and "
                    f"{tail_counts['calibrated']} after calibration. A small count can yield a visually large calibration "
                    "departure from only a few outcomes. Five reliability bins are a compromise between resolution and variance."
                ),
                figure(
                    10,
                    "Figure 10. Holdout proper scoring rules with conditional paired-bootstrap marginal intervals. Lower is "
                    "better; axes start at zero. The constant's interval collapses because stratification fixes prevalence.",
                    210,
                ),
                paragraph(
                    f"Boosting Brier score is {f(h['boosting']['brier'], 5)} versus {f(h['logistic']['brier'], 5)} "
                    f"for logistic. Calibration changes Brier by {signed(brier_delta, 5)} and log loss by {signed(loss_delta, 5)}. "
                    f"The directions differ and {calibration_conclusion}. The apparently improved reliability curve cannot "
                    "replace this proper-score comparison."
                ),
                paragraph(
                    f"Calibration reduces outer-fold log loss in {stability['calibration_log_loss_wins']}/{stability['n_folds']} "
                    f"folds. Its mean outer log loss is {f(cv['calibrated']['log_loss']['mean'])}, compared with "
                    f"{f(cv['boosting']['log_loss']['mean'])} before calibration. This weakens an automatic-calibration policy "
                    "for limited data. Isotonic regression was not added after seeing the result."
                ),
            ],
        )
    )

    comparison_rows = [["Paired contrast", "Metric", "Difference", "95% interval"]]
    for comparison, label in [
        ("boosting_minus_logistic", "Boosting - logistic"),
        ("calibrated_minus_boosting", "Calibrated - raw"),
    ]:
        for metric, label_metric in [
            ("roc_auc", "AUC"),
            ("brier", "Brier"),
            ("log_loss", "Log loss"),
        ]:
            row = intervals.loc[
                (intervals.comparison == comparison) & (intervals.metric == metric)
            ].iloc[0]
            comparison_rows.append(
                [label, label_metric, signed(row.estimate, 4), interval(comparison, metric, 4)]
            )
    pages.append(
        Page(
            "7. Statistical interpretation and uncertainty",
            [
                figure(
                    11,
                    "Figure 11. Paired differences with 95% percentile intervals. Positive AUC and negative scoring-rule "
                    "differences favour the first model. Each contrast uses the same resampled records for both models.",
                    195,
                ),
                table(
                    "Table 7. Conditional paired comparisons; no multiplicity-adjusted global superiority claim is made.",
                    comparison_rows,
                ),
                paragraph(
                    f"The bootstrap draws {m['partitions']['holdout']['bad']} bad and {m['partitions']['holdout']['good']} "
                    f"good rows with replacement, separately, for {config['bootstrap_replicates']:,} replicates [7]. "
                    f"There were {m['bootstrap']['rejected_one_class']} one-class rejections. Percentiles 2.5 and 97.5 form "
                    "the intervals; the code also detects one-class draws in an optional unstratified procedure. Paired draws "
                    "retain correlation between model errors, making them appropriate for differences."
                ),
                paragraph(
                    f"{conditional_evidence}. However, outer-fold mean AUC difference is "
                    f"{signed(stability['auc_difference_mean'])}, with the opposite direction to the holdout. The holdout "
                    "and outer folds answer related but different questions: performance of the final fitted models versus "
                    "variation of the learning workflow trained on smaller samples. Neither observation should erase the other."
                ),
                paragraph(
                    "These are nominal conditional intervals across several related metrics, not a familywise test or a "
                    "population guarantee. They exclude model retraining, source-bank prevalence, calendar drift and unknown "
                    "borrower dependence. No p-value or independent-validation claim is made. Economic materiality remains "
                    "undetermined without loss severity, exposure, decision costs and representative validation data."
                ),
            ],
        )
    )

    scenario_rows = [["Assumed prevalence", "Source p=0.30", "Source p=0.50", "Source p=0.80"]]
    for prior in [0.05, 0.1, 0.2, 0.3]:
        scenario_rows.append(
            [f"{prior:.0%}", *[f(float(prior_shift(p, 0.3, prior))) for p in [0.3, 0.5, 0.8]]]
        )
    pages.append(
        Page(
            "8. Robustness and prevalence sensitivity",
            [
                paragraph(
                    f"Analysis A is the training-only fold comparison: boosting leads in {stability['boosting_auc_wins']} "
                    f"of {stability['n_folds']} folds, with a mean AUC contrast of {signed(stability['auc_difference_mean'])}. "
                    "This probes partition and tuning sensitivity without reusing the holdout. Overlapping fit sets mean "
                    "the folds are not independent observations for an ordinary t-test. Repeated CV or an alternative seed "
                    "might extend this descriptive analysis, but would not create external validation."
                ),
                paragraph(
                    "Analysis B changes only the assumed class prior. From Bayes' theorem, source posterior odds equal "
                    "the class-conditional likelihood ratio times source prior odds. If P_target(X|Y)=P_source(X|Y) "
                    "and the source posterior is calibrated, replacing prior odds yields [8]:"
                ),
                Block(
                    "equation",
                    r"\frac{p_t(x)}{1-p_t(x)}=\frac{p_s(x)}{1-p_s(x)}\;\frac{\pi_t/(1-\pi_t)}{\pi_s/(1-\pi_s)}",
                ),
                figure(
                    13,
                    "Figure 13. Theoretical odds correction from a 30% source prior to explicitly hypothetical target "
                    "prevalences. The map is monotone and preserves ranking, but changes probability levels substantially.",
                    240,
                ),
                table(
                    "Table 8. Mathematical scenarios, not estimated bank default probabilities.",
                    scenario_rows,
                ),
                paragraph(
                    f"For example, source probability 0.50 becomes {float(prior_shift(0.5, 0.3, 0.1)):.3f} under an assumed "
                    "10% prior. This does not recover the bank's true prevalence, repair a misdefined outcome, or correct "
                    "changes in lending policy and conditional feature distributions. The mean adjusted probability on the "
                    "source sample need not equal the target prior because the source feature mixture has not been replaced."
                ),
            ],
        )
    )

    largest = coefficients.loc[coefficients.coefficient.abs().idxmax()]
    pages.append(
        Page(
            "9. Model risk: interpreting predictive factors",
            [
                figure(
                    12,
                    "Figure 12. Left: seven largest absolute regularised logistic coefficients. Suffixes denote category "
                    "codes, relative to the first documented code; continuous effects are per training SD. Right: "
                    "boosting permutation importance, mean plus/minus one SD across outer folds, not confidence intervals.",
                    260,
                ),
                paragraph(
                    f"The largest absolute logistic coefficient is {largest['description'].strip()}, "
                    f"with coefficient {largest['coefficient']:.3f} relative to {largest['reference'].strip()}. "
                    f"Its fitted conditional odds multiplier is {np.exp(largest['coefficient']):.3f}. This is a penalised "
                    "association holding other encoded predictors fixed, not an intervention estimate or an unpenalised "
                    "inference result. Complete coefficients and reference labels are saved in logistic_coefficients.csv."
                ),
                paragraph(
                    f"The largest mean permutation effects are {', '.join(importance.index[:3]).replace('_', ' ')}. "
                    f"Shuffling {importance.index[0].replace('_', ' ')} raises outer-validation log loss by "
                    f"{importance.iloc[0]:.3f} on average. Each raw feature is shuffled before preprocessing, preserving its "
                    "one-hot group. Ten shuffles per fold reduce Monte Carlo noise. Correlated features can substitute "
                    "for one another, and permutation can create implausible combinations; low importance is not proof "
                    "that a variable is irrelevant."
                ),
                subheading("Demographics, proxies and representativeness"),
                paragraph(
                    "Age, foreign-worker status and combined personal status/sex are present. Housing, job and historical "
                    "telephone access may also act as proxies. The personal-status code mixes male singles with female "
                    "non-singles, so sex cannot be reliably reconstructed. A sex-based fairness score would be poorly "
                    "founded. These variables are retained for a transparent historical comparison, not endorsed for use "
                    "in lending decisions. Sparse subgroups and no decision-outcome study prevent fairness or compliance claims."
                ),
                paragraph(
                    "The data concern issued credits; extrapolation to rejected applicants is unsupported. Borrower identifiers "
                    "and prediction timestamps are unavailable, preventing grouped or forward-time validation. Source-era "
                    "features such as landline access have different meanings in a contemporary portfolio."
                ),
            ],
        )
    )

    threshold_rows = [["Model", "Sensitivity", "Specificity", "Precision", "Flagged / 200"]]
    for name in MODEL_NAMES:
        threshold_rows.append(
            [
                SHORT[name],
                f(h[name]["sensitivity"]),
                f(h[name]["specificity"]),
                f(h[name]["precision"]),
                str(h[name]["predicted_positive"]),
            ]
        )
    pages.append(
        Page(
            "9. Model risk: decisions and responsible use",
            [
                figure(
                    14,
                    "Figure 14. Holdout threshold trade-offs. The prespecified 0.5 threshold is illustrative. The curves "
                    "are descriptive and are not used to optimise a lending policy.",
                    225,
                ),
                table(
                    "Table 9. Threshold 0.5. Constant precision is undefined because it flags no records.",
                    threshold_rows,
                ),
                paragraph(
                    f"At 0.5, boosting identifies {h['boosting']['tp']} of {m['partitions']['holdout']['bad']} bad outcomes "
                    f"and flags {h['boosting']['fp']} good outcomes. Logistic identifies {h['logistic']['tp']} and flags "
                    f"{h['logistic']['fp']}, respectively. Calibration changes probability levels, but these counts provide "
                    "no economic justification for the threshold. The sample omits exposure, loss severity, operational "
                    "constraints, uncertainty in repayment timing and the consequences of approving or declining a loan."
                ),
                subheading("Evidence required before financial-risk use"),
                paragraph(
                    "A financial institution would first need a defined outcome and horizon; representative application and "
                    "performance data with reliable timestamps; handling of maturity, censoring and rejected applications; "
                    "and independently held-out calendar or external portfolios. Validation would include calibration by "
                    "time and relevant segment, economic decision analysis, stability and stress tests, feature legitimacy "
                    "and appropriate subgroup evaluation. Institutional governance and applicable requirements would need "
                    "their own review; this report does not assess regulatory compliance."
                ),
                paragraph(
                    "The actual findings sharpen these needs: the model ranking changes across partitions, the calibration "
                    "intervention lacks a consistent score benefit, and tail reliability rests on few observations. "
                    "Deployment would additionally require data controls, documented ownership, monitoring for drift, "
                    "recalibration rules, independent challenge and change management. None is established by this portfolio study."
                ),
            ],
        )
    )

    pages.append(
        Page(
            "10. What is different about this research?",
            [
                paragraph(
                    "The contribution is a controlled empirical comparison with explicit probability and model-risk questions, "
                    "not a claim of a new learning algorithm. It tests whether complexity helps rather than selecting the "
                    "largest score by default. It uses proper scoring rules and calibration diagnostics alongside ranking; "
                    "keeps tuning inside calibration cross-fitting; isolates the sigmoid intervention by sharing its base "
                    "estimator; reports paired uncertainty; and makes sampling assumptions visible through odds scenarios."
                ),
                paragraph(
                    "Reproducibility is part of the evidence. Source checksums, row-level split manifests, out-of-fold and "
                    "holdout predictions, full small-grid results, bootstrap draws and figure inputs are retained. "
                    "One structured narrative generates both report formats, while numerical statements are read from "
                    "saved outputs. Meaningful tests cover source coding, leakage boundaries, metrics, calibration, "
                    "bootstrap edge cases and complete artifact production."
                ),
                subheading("11. Conclusion: what the experiments support"),
                paragraph(summary),
                paragraph(
                    f"H1 is supported within this sample: trained models discriminate beyond the constant. H2 is mixed: "
                    f"the holdout favours boosting, but outer-fold comparisons do not support a stable advantage. H3 is "
                    "illustrated by the distinction between AUC and reliability diagnostics, not established as a universal "
                    f"empirical law. H4 receives no convincing benefit here: calibrated-minus-raw Brier is "
                    f"{signed(brier_delta, 5)} and log loss is {signed(loss_delta, 5)}, with both paired intervals spanning zero. "
                    "H5 remains a justified identification limitation: population calibration is not testable from this sample alone."
                ),
                subheading("What the experiments do not establish"),
                paragraph(
                    "They do not establish contemporary PD accuracy, economic value, causal effects, regulatory compliance, "
                    "fair credit decisions or external validity. Nor does the mixed evidence prove model equivalence. "
                    "Conditional bootstrap precision cannot replace representative data or uncertainty from repeated fitting."
                ),
                subheading("Practical implications and next research steps"),
                paragraph(
                    "Logistic regression remains a credible benchmark when transparent interpretation and limited data "
                    "matter. Boosting merits further investigation because of its holdout performance, but its complexity "
                    "is not justified by a robust population-level superiority finding. Next steps are a larger representative "
                    "time-stamped cohort, a defined default horizon, external or forward-time validation, assessment of "
                    "training variability, subgroup analysis with valid attributes, and institution-specific decision costs. "
                    "Additional model families would be secondary to improving those data and validation conditions."
                ),
            ],
        )
    )

    reference_blocks = [
        Block("reference", f"[{index}] {text}\n{url}")
        for index, (text, url) in enumerate(REFERENCES, 1)
    ]
    pages.append(
        Page(
            "12. References and reproducibility",
            reference_blocks
            + [
                subheading("Reproduce the study from the repository root"),
                Block(
                    "code",
                    'python -m pip install -e ".[dev]"\npython -m pytest -q\npython -m credit_risk all\npython -m credit_risk verify',
                ),
                paragraph(
                    "On systems with make, the equivalent workflow is make test followed by make all. A fresh Python 3.11+ "
                    "virtual environment is recommended. requirements-lock.txt records the executed dependency versions. "
                    "The original public archive is included with attribution; network access is required only to install "
                    "dependencies or restore absent source files. The report command rebuilds presentation without refitting."
                ),
                paragraph(
                    "Audit trail: results/metrics.json, cv_results.csv, holdout_predictions.csv, bootstrap_intervals.csv, "
                    "split_manifest.json and run_manifest.json. The final artifact manifest hashes deliverables. "
                    "All fourteen figures are available as PNG and SVG. No model artifact is needed because the study "
                    "can refit from its compact source data and fixed protocol."
                ),
            ],
        )
    )
    context = {
        "metrics": m,
        "summary": summary,
        "abstract": abstract,
        "auc_delta": auc_delta,
        "auc_interval": interval("boosting_minus_logistic", "roc_auc"),
        "brier_delta": brier_delta,
        "loss_delta": loss_delta,
    }
    return pages, context


def markdown_document(pages: list[Page]) -> str:
    lines = []
    for page in pages:
        lines.append(f"{'#' if page.cover else '##'} {page.title}\n")
        for block in page.blocks:
            if block.kind == "figure":
                lines += [
                    f"![{block.text}](../results/figures/{FIGURE_NAMES[block.figure - 1]}.png)\n",
                    f"*{block.text}*\n",
                ]
            elif block.kind == "table":
                lines.append(f"**{block.text}**\n")
                lines.append("| " + " | ".join(block.rows[0]) + " |")
                lines.append("| " + " | ".join(["---"] * len(block.rows[0])) + " |")
                lines.extend("| " + " | ".join(row) + " |" for row in block.rows[1:])
                lines.append("")
            elif block.kind == "equation":
                lines.append(f"$$\n{block.text}\n$$\n")
            elif block.kind == "subheading":
                lines.append(f"### {block.text}\n")
            elif block.kind == "code":
                lines.append(f"```bash\n{block.text}\n```\n")
            elif block.kind == "reference":
                text, url = block.text.split("\n", 1)
                lines.append(f"{text} [Source]({url})\n")
            elif block.kind == "callout":
                lines.append(f"> {block.text}\n")
            else:
                lines.append(block.text + "\n")
    return "\n".join(lines)


def render_pdf(pages: list[Page], root: Path) -> None:
    for name, weight in [("ResearchSans", "normal"), ("ResearchBold", "bold")]:
        path = font_manager.findfont(
            font_manager.FontProperties(family="DejaVu Sans", weight=weight)
        )
        pdfmetrics.registerFont(TTFont(name, path))
    pdfmetrics.registerFontFamily(
        "ResearchSans",
        normal="ResearchSans",
        bold="ResearchBold",
        italic="ResearchSans",
        boldItalic="ResearchBold",
    )
    navy, teal, grey = (
        colors.HexColor("#183449"),
        colors.HexColor("#176B9A"),
        colors.HexColor("#5E6D79"),
    )
    body = ParagraphStyle(
        "body",
        fontName="ResearchSans",
        fontSize=9.35,
        leading=13.6,
        textColor=navy,
        spaceAfter=8,
        alignment=TA_LEFT,
    )
    styles = {
        "paragraph": body,
        "subtitle": ParagraphStyle(
            "subtitle", parent=body, fontSize=13, leading=19, textColor=grey, spaceAfter=18
        ),
        "eyebrow": ParagraphStyle(
            "eyebrow",
            parent=body,
            fontName="ResearchBold",
            fontSize=8,
            leading=12,
            textColor=teal,
            spaceAfter=12,
        ),
        "subheading": ParagraphStyle(
            "subhead",
            parent=body,
            fontName="ResearchBold",
            fontSize=11,
            leading=15,
            spaceBefore=7,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "callout": ParagraphStyle(
            "callout",
            parent=body,
            fontName="ResearchBold",
            fontSize=11,
            leading=16,
            backColor=colors.HexColor("#EEF4F7"),
            borderPadding=11,
            spaceBefore=12,
            spaceAfter=17,
        ),
        "reference": ParagraphStyle(
            "reference", parent=body, fontSize=8.0, leading=10.9, spaceAfter=7
        ),
        "code": ParagraphStyle(
            "code",
            parent=body,
            fontSize=9,
            leading=13,
            backColor=colors.HexColor("#F2F4F6"),
            borderPadding=8,
            spaceBefore=7,
            spaceAfter=14,
        ),
    }
    caption_style = ParagraphStyle(
        "caption", parent=body, fontSize=8.0, leading=10.8, textColor=grey, spaceAfter=10
    )
    cell_style = ParagraphStyle("cell", parent=body, fontSize=8.1, leading=11.1, spaceAfter=0)
    header_style = ParagraphStyle(
        "header", parent=cell_style, fontName="ResearchBold", textColor=colors.white
    )
    title_style = ParagraphStyle(
        "title", parent=body, fontName="ResearchBold", fontSize=18, leading=23, spaceAfter=16
    )
    cover_title = ParagraphStyle(
        "cover_title", parent=title_style, fontSize=28, leading=34, spaceBefore=24, spaceAfter=21
    )
    document = SimpleDocTemplate(
        str(root / "reports/research_report.pdf"),
        pagesize=A4,
        leftMargin=42,
        rightMargin=42,
        topMargin=48,
        bottomMargin=42,
        title=TITLE,
        author="Aryan Golbaghi",
        subject=SUBTITLE,
        pageCompression=1,
        invariant=1,
    )
    width = A4[0] - 84
    story = []
    buffers = []
    for index, page in enumerate(pages):
        if index:
            story.append(PageBreak())
        story.append(Paragraph(escape(page.title), cover_title if page.cover else title_style))
        for block in page.blocks:
            if block.kind == "figure":
                path = root / "results/figures" / f"{FIGURE_NAMES[block.figure - 1]}.png"
                with PILImage.open(path) as img:
                    w, h = img.size
                scaling = min(width / w, block.max_height / h)
                story.append(
                    KeepTogether(
                        [
                            Image(str(path), width=w * scaling, height=h * scaling),
                            Spacer(1, 5),
                            Paragraph(escape(block.text), caption_style),
                        ]
                    )
                )
            elif block.kind == "table":
                columns = len(block.rows[0])
                fractions = (
                    [0.27, 0.73]
                    if columns == 2
                    else [0.23, 0.39, 0.38]
                    if columns == 3
                    else [0.34, 0.15, 0.18, 0.33]
                    if columns == 4
                    else [0.24] + [0.76 / (columns - 1)] * (columns - 1)
                )
                if block.text.startswith("Table 8."):
                    fractions = [0.28, 0.24, 0.24, 0.24]
                table_data = [
                    [
                        Paragraph(escape(str(cell)), header_style if i == 0 else cell_style)
                        for cell in row
                    ]
                    for i, row in enumerate(block.rows)
                ]
                rendered = Table(
                    table_data,
                    colWidths=[width * fraction for fraction in fractions],
                    hAlign="LEFT",
                    repeatRows=1,
                )
                rendered.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), navy),
                            (
                                "ROWBACKGROUNDS",
                                (0, 1),
                                (-1, -1),
                                [colors.HexColor("#F1F5F7"), colors.white],
                            ),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 7),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                            ("TOPPADDING", (0, 0), (-1, -1), 6),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                            ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#BBC7D0")),
                        ]
                    )
                )
                story.append(
                    KeepTogether(
                        [Paragraph(escape(block.text), caption_style), rendered, Spacer(1, 10)]
                    )
                )
            elif block.kind == "equation":
                fig = plt.figure(figsize=(10, 0.55))
                fig.text(0.5, 0.5, "$" + block.text + "$", fontsize=16, ha="center", va="center")
                buffer = io.BytesIO()
                fig.savefig(buffer, format="png", dpi=240, bbox_inches="tight", pad_inches=0.08)
                plt.close(fig)
                buffer.seek(0)
                buffers.append(buffer)
                with PILImage.open(buffer) as img:
                    w, h = img.size
                buffer.seek(0)
                scaling = min(width / w, 37 / h)
                story.extend([Image(buffer, width=w * scaling, height=h * scaling), Spacer(1, 8)])
            elif block.kind == "reference":
                description, url = block.text.split("\n", 1)
                story.append(
                    Paragraph(
                        escape(description)
                        + f' <link href="{escape(url)}" color="#176B9A">Source</link>',
                        styles["reference"],
                    )
                )
            else:
                story.append(
                    Paragraph(escape(block.text).replace("\n", "<br/>"), styles[block.kind])
                )

    def chrome(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#CAD5DD"))
        canvas.setLineWidth(0.6)
        canvas.line(42, 31, A4[0] - 42, 31)
        canvas.setFont("ResearchSans", 7.5)
        canvas.setFillColor(grey)
        canvas.drawString(42, 20, "BEYOND AUC  |  Independent quantitative research")
        canvas.drawRightString(A4[0] - 42, 20, str(doc.page))
        if doc.page > 1:
            canvas.setFont("ResearchBold", 7.2)
            canvas.drawString(
                42, A4[1] - 28, "CREDIT RISK  /  PROBABILITY QUALITY  /  MODEL VALIDATION"
            )
        canvas.restoreState()

    document.build(story, onFirstPage=chrome, onLaterPages=chrome)


def portfolio_documents(root: Path, context: dict) -> None:
    m, summary = context["metrics"], context["summary"]
    h, cv, stability = m["holdout"], m["cv"], m["stability"]
    readme = f"""# {TITLE}

*{SUBTITLE}*

An independent quantitative research project comparing interpretable logistic regression with shallow-tree gradient boosting on corrected UCI South German Credit data. The focus is reliable probabilities, uncertainty and model risk, with an executed study and a 16-page research report.

**Research question:** Does extra model complexity produce a defensible improvement in ranking and probability quality, and what can a historical, case-enriched sample establish about lending risk?

- Built a typed, modular Python research pipeline with a locked 80/20 split and nested five-fold validation.
- Cross-fitted complete hyperparameter searches before sigmoid calibration, comparing the identical base model before and after its mapping.
- Calculated proper scoring rules, calibration diagnostics and {m["config"]["bootstrap_replicates"]:,} paired stratified bootstrap replicates.
- Executed fold stability and hypothetical prevalence-shift analyses, with 14 original figures in PNG and SVG.
- Preserved source checksums, split manifests, predictions, tuning results and a tested report-generation workflow.

## Measured findings

{summary}

| Locked holdout (200 records) | ROC-AUC | Average precision | Brier | Log loss |
| --- | ---: | ---: | ---: | ---: |
"""
    for name in MODEL_NAMES:
        readme += f"| {MODEL_NAMES[name]} | {f(h[name]['roc_auc'])} | {f(h[name]['average_precision'])} | {f(h[name]['brier'])} | {f(h[name]['log_loss'])} |\n"
    readme += f"""
Mean outer-fold AUC is **{f(cv["logistic"]["roc_auc"]["mean"])} for logistic** versus **{f(cv["boosting"]["roc_auc"]["mean"])} for boosting**. Calibration changes holdout Brier score by {signed(context["brier_delta"], 5)} and log loss by {signed(context["loss_delta"], 5)}; both paired intervals include zero. Its outer-fold log loss improves in only {stability["calibration_log_loss_wins"]}/{stability["n_folds"]} folds.

The bootstrap conditions on fitted models and the holdout's 60 bad / 140 good mix. Its intervals do not cover model retraining, drift or population prevalence uncertainty. The study does not establish a stable universal model winner.

**[Read the complete PDF report](reports/research_report.pdf)** · [Markdown report](reports/research_report.md) · [CV and interview summary](docs/cv_summary.md)

![Paired uncertainty](results/figures/11_paired_uncertainty.png)

![Reliability diagnostics](results/figures/08_reliability.png)

![Hypothetical prevalence sensitivity](results/figures/13_prevalence_sensitivity.png)

## Design and audit trail

The original target is reversed explicitly: `bad_credit = (kredit == 0)`. Only duration, transformed amount and age are continuous. The remaining 17 features use documented categorical levels. Training-only preprocessing, tuning and calibration are nested; log loss selects from three logistic and four boosting configurations. Threshold 0.5 and sigmoid calibration are fixed before the holdout is inspected.

See [prespecified protocol](docs/protocol.md), [methodology](docs/methodology.md), [research decisions](docs/research_decisions.md), and [data provenance](data/README.md). Every included figure is discussed in the report. Numerical findings are generated from saved outputs rather than hardcoded into report text.

## Reproduce

Use Python 3.11+ in a fresh environment. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .\\.venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
python -m credit_risk all
python -m credit_risk verify
```

Or, after installation: `make test` and `make all`. Windows does not need Make. The full research run takes a few minutes on a typical laptop. For the executed dependency versions, install `requirements-lock.txt` before the editable package; small numerical variation can occur across versions/platforms.

`python -m credit_risk experiment` runs modelling only. `python -m credit_risk report` regenerates figures, Markdown, PDF, README and CV copy from saved evidence, without refitting. `python -m ruff check src tests` checks code style. Output tests inspect the checked-in results; run `all` before tests if intentionally starting without generated artifacts.

The public source archive is included, so experiments can run offline after dependency installation. If raw files are missing, the loader downloads the official UCI archive and verifies known hashes. Network failure or changed data stops the run; no synthetic replacement is used.

## Repository map

| Path | Contents |
| --- | --- |
| `src/credit_risk/` | Loading, preprocessing, models, nested experiment, evaluation, calibration, robustness, figures and reports |
| `tests/` | Behavioural and integration tests |
| `results/metrics.json` | Holdout metrics, calibration diagnostics and CV summaries |
| `results/*predictions.csv` | Row-level holdout and outer-fold predictions |
| `results/bootstrap_*.csv` | Paired draws and percentile intervals |
| `results/*manifest.json` | Split, execution and final artifact provenance |
| `results/figures/` | Fourteen numbered PNG/SVG figure pairs |
| `reports/` | Finished research report in PDF and Markdown |

## Scope and limitations

The records date from 1973-1975; bad credits were deliberately oversampled. The 30% share is **not a bank population default rate**, and the label is not a fully specified modern regulatory default event or horizon. Historical amount units are transformed. Combined personal-status codes cannot reliably identify sex. No claims of fairness, causality, regulatory compliance, independent validation or deployment readiness are made. More representative and time-stamped data would matter more than a larger model search.

Dataset: [South German Credit, UCI](https://doi.org/10.24432/C5X89F), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Correction: [Grömping (2019), Report 4/2019](https://www1.beuth-hochschule.de/FB_II/reports/Report-2019-004.pdf). Code is MIT licensed; source data retain their separate licence. This independent portfolio project has no bank affiliation or endorsement.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    cv_text = f"""# CV and interview summary

## A. One-line project title

Beyond AUC: Credit Risk Probability Modelling, Calibration and Statistical Validation

## B. Two CV bullets

- Built a reproducible Python credit-risk study with nested cross-validation, leakage-controlled sigmoid calibration, proper scoring rules and {m["config"]["bootstrap_replicates"]:,} paired bootstrap replicates on 1,000 public credit records.
- Compared logistic regression and gradient boosting, finding holdout AUCs of {f(h["logistic"]["roc_auc"])} and {f(h["boosting"]["roc_auc"])}, while documenting contradictory fold rankings, calibration uncertainty and prevalence-shift assumptions.

## C. Approximately 45-second explanation

I chose this project because credit-risk models need useful probabilities, not just a good ranking score. I built a reproducible Python study comparing logistic regression, boosting and calibrated boosting, with nested validation and paired bootstrap uncertainty. Boosting reached holdout AUC {f(h["boosting"]["roc_auc"])} versus {f(h["logistic"]["roc_auc"])} for logistic regression, but logistic won most training-validation folds. Calibration produced almost no benefit. That taught me to distinguish a favourable split from a robust modelling conclusion. I also derived a prevalence adjustment to show why this deliberately oversampled historical dataset cannot establish real portfolio default probabilities. With more data, I would prioritise a defined default horizon, representative sampling and forward-time external validation.

## D. Three quantitative interview questions

**1. Why can AUC stay constant while calibration changes?**

AUC is the probability that a randomly drawn bad outcome has a higher score than a good outcome, with half credit for ties. A strictly increasing transformation preserves every ordering, so it preserves AUC. Calibration concerns absolute probability levels. Our sigmoid mapping preserves the boosting ranking and changes probabilities; proper scoring rules can therefore change even when AUC and average precision do not. Brier score and log loss assess probability quality, but also reflect discrimination. A constant prevalence predictor can be calibrated in the aggregate and useless at ranking.

**2. Where could leakage enter, and how did you prevent it?**

Global imputation/scaling, tuning on evaluation folds and fitting a calibrator to in-sample predictions can all leak information. We reserve a stratified 200-record holdout, fit preprocessing inside pipelines, and evaluate entire workflows in five outer training folds. Each calibration score is produced by a full five-fold hyperparameter search fitted without that row. The final raw boosting comparator is exactly the base estimator inside the calibrated model. A spy-estimator test asserts that no score uses its own fit rows. The test set is used only after settings are frozen; diagnostic recalibration is descriptive and never updates predictions.

**3. Can 30% bad credit be used as the bank's default rate?**

No. The sampling deliberately enriches bad outcomes, and contract non-compliance is not automatically a modern default event with a defined horizon. Under label shift, stable P(X|Y), and a calibrated source posterior, target odds equal source odds times [pi_t/(1-pi_t)] / [pi_s/(1-pi_s)]. For a hypothetical 10% target prior, a source probability of 0.5 becomes {float(prior_shift(0.5, 0.3, 0.1)):.3f}. This illustrates sensitivity; it does not estimate the bank's true prior, correct changing conditional distributions, or establish deployable PD calibration. The bootstrap likewise conditions on the observed class mix and excludes prevalence uncertainty.

## Evidence boundaries to defend

The paired holdout AUC difference is {signed(context["auc_delta"])}, with conditional 95% interval {context["auc_interval"]}. That supports an advantage on this fixed partition; the opposing outer-CV pattern prevents a universal superiority claim. Do not describe calibration as a demonstrated improvement or imply bank sponsorship, real-lending deployment, fairness certification or academic novelty.
"""
    (root / "docs/cv_summary.md").write_text(cv_text, encoding="utf-8")


def build_report(root: Path) -> None:
    (root / "reports").mkdir(parents=True, exist_ok=True)
    pages, context = narrative(root)
    (root / "reports/research_report.md").write_text(markdown_document(pages), encoding="utf-8")
    render_pdf(pages, root)
    portfolio_documents(root, context)
    sources = list((root / "results").glob("*.csv")) + [root / "results/metrics.json"]
    outputs = [
        root / "reports/research_report.md",
        root / "reports/research_report.pdf",
        root / "README.md",
        root / "docs/cv_summary.md",
    ]
    outputs += sorted((root / "results/figures").glob("*.png")) + sorted(
        (root / "results/figures").glob("*.svg")
    )
    write_json(
        root / "results/artifact_manifest.json",
        {
            "input_sha256": {
                str(path.relative_to(root)).replace("\\", "/"): sha256(path) for path in sources
            },
            "output_sha256": {
                str(path.relative_to(root)).replace("\\", "/"): sha256(path) for path in outputs
            },
            "intended_pages": len(pages),
            "figure_count": len(FIGURE_NAMES),
        },
    )
    print("Generated PDF, Markdown report, README and CV summary from saved evidence", flush=True)


def verify_artifacts(root: Path) -> dict:
    required = [
        "results/metrics.json",
        "results/cv_results.csv",
        "results/holdout_predictions.csv",
        "results/bootstrap_intervals.csv",
        "results/split_manifest.json",
        "results/run_manifest.json",
        "results/selection_locked.json",
        "reports/research_report.pdf",
        "reports/research_report.md",
        "README.md",
        "docs/cv_summary.md",
        "results/artifact_manifest.json",
    ]
    for path in required:
        if not (root / path).is_file() or (root / path).stat().st_size == 0:
            raise ValueError(f"Required artifact missing or empty: {path}")
    manifest = json.loads((root / "results/artifact_manifest.json").read_text())
    for section in ["input_sha256", "output_sha256"]:
        for path, expected in manifest[section].items():
            if sha256(root / path) != expected:
                raise ValueError(f"Artifact hash mismatch: {path}; regenerate the report")
    for name in FIGURE_NAMES:
        for extension in ["png", "svg"]:
            if not (root / "results/figures" / f"{name}.{extension}").is_file():
                raise ValueError(f"Figure missing: {name}.{extension}")
    reader = PdfReader(root / "reports/research_report.pdf")
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if len(reader.pages) != manifest["intended_pages"]:
        raise ValueError(
            f"PDF has {len(reader.pages)} pages; intended {manifest['intended_pages']}. Inspect overflow."
        )
    for number in range(1, len(FIGURE_NAMES) + 1):
        if f"Figure {number}." not in text:
            raise ValueError(f"Missing figure discussion/caption: {number}")
    if any(token in text for token in ["PLACEHOLDER", "TODO", "\u25a0"]) or re.search(
        r"\bnan\b", text
    ):
        raise ValueError("Unresolved text or missing glyph found in PDF")
    return {
        "status": "complete",
        "pdf_pages": len(reader.pages),
        "numbered_figures": len(FIGURE_NAMES),
        "hashed_outputs": len(manifest["output_sha256"]),
    }
