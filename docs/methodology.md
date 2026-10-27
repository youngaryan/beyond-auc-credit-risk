# Methodology and mathematical audit

The complete, results-driven explanation is in the research report. This document
provides an implementation map and additional details for technical challenge.
The original choices are recorded in [protocol.md](protocol.md), written before
model fitting. The study has one fixed holdout split; no seed search was performed.

## Estimand and sampling

Let Y=1 denote the source's bad-credit outcome. The fitted quantity is a posterior
probability in the deliberately enriched study sample, not a modern horizon-defined
regulatory probability of default. Apparent calibration conditions on this sample's
selection process. An event definition, horizon and representative population are
needed before discussing a portfolio PD interpretation.

The original codebook uses `kredit=0` for bad and `kredit=1` for good. Loading checks
the original file checksum, header, dimensions, levels and outcome counts before
constructing `bad_credit=(kredit==0)`. Public source files retain their original
bytes. The descriptive audit can inspect all rows for integrity; outcome-dependent
exploration and feature interpretation use only the training partition.

## Nested fitting, step by step

1. Stratified 80/20 split, seed 42: 800 development rows and a locked 200-row test.
2. Five outer development folds, seed 43: fit the complete workflow on 640 rows;
   evaluate on the other 160. This is the reported training-only stability analysis.
3. Logistic regression: fit a pipeline and select C from 0.1, 1 and 10 using five
   inner stratified folds, seed 52, minimising mean log loss. Refit on the outer fit.
4. Calibrated boosting: cross-fit a `GridSearchCV` estimator in five calibration
   folds, seed 62. **Each score's own row is excluded from the inner search as well
   as fitting and preprocessing.** Each search has four depth/rate configurations.
5. Fit the sigmoid using all cross-fitted development scores, and fit one final
   search on all development rows. `ensemble=False` produces one full-data base
   estimator plus a sigmoid, not an average of different fold models.
6. Extract the raw comparator from that same fitted base estimator. Differences
   between raw and calibrated predictions isolate the learned mapping.
7. After the outer loop, repeat steps 3-6 on the 800 development rows. Write
   selected settings and the protocol digest before requesting holdout predictions.
8. Evaluate all four configurations once. The diagnostic regressions in the
   report estimate reliability parameters but never replace these predictions.

The full-training search evaluated within the final 800 rows is used for model
selection only. Its winning score is not reported as unbiased workflow performance.
The outer scores evaluate the complete adaptive procedure. Outer folds overlap in
their fit data; their standard deviation is descriptive, not an independent-sample
standard error. Five folds do not identify performance under calendar drift.

In an outer calibration fold, fitting/tuning uses 512 rows and scores the remaining
128; the innermost fits have 409-410 rows. In final calibration, fitting/tuning uses
640 rows and scores 160. Both classes remain in every split. Smaller cross-fitted
models may generate score distributions different from the final model, a residual
calibration transfer limitation even without leakage.

## Feature representation

The 3 continuous quantities are duration, transformed amount and age. Numeric
median imputation and standardisation are fitted in each pipeline. The 17 remaining
variables use mode imputation and one-hot encoding, dropping the first documented
code. Categories are supplied by the source codebook, an external schema, not
learned from the holdout. This includes purpose code 7, which is documented but
unobserved; an all-zero fitted column receives no evidential interpretation.
Undocumented categories raise an error instead of silently becoming a reference.

This conservative treatment of ordinal bins avoids assuming equal spacing or a
linear log-odds response. Its cost is extra parameters and loss of ordered
structure. Scaling is unnecessary for trees but harmless and makes the workflow
uniform. No outlier removal, feature selection, class weighting or synthetic
resampling is applied. The original file contains no missing values; imputation
is still inside the pipeline and is tested as a robustness property.

Logistic coefficients are penalised conditional log-odds contrasts. Numerical
coefficients describe one training SD; category coefficients are contrasts with a
named reference, not mutually comparable changes in a continuous quantity. L2
shrinkage with reference coding can depend on the chosen reference; the convention
is fixed and disclosed. No coefficient confidence intervals or causal claims are
attached to these tuned coefficients.

## Metrics and diagnostics

- AUC = P(score_bad > score_good) + 0.5 P(tie). It is invariant under a strictly
  increasing transformation. AP is a recall-weighted precision average; it is not
  trapezoidal PR-AUC and is prevalence-dependent.
- Brier = mean((p-y)^2); log loss = -mean(y log p + (1-y) log(1-p)). Both are
  strictly proper in expectation, but neither is a pure measure of calibration.
- Sensitivity = TP/(TP+FN), specificity = TN/(TN+FP), precision = TP/(TP+FP),
  at fixed p>=0.5. Undefined denominators become null/NaN and are disclosed.
- Reliability uses 5 fixed equal-width probability bins, left-inclusive and
  right-exclusive except the last bin includes 1. Empty bins remain explicit.
  Wilson intervals approximate within-bin binomial uncertainty; heterogeneity and
  fitting uncertainty are not incorporated. ECE = sum(n_bin/n * abs(mean_p-rate)).
- Joint calibration diagnostics maximise the unpenalised binomial likelihood for
  logit P(Y=1)=a+b logit(p). The ideal a=0,b=1; Fisher-information inverse gives
  approximate standard errors. Constant scores do not identify slope. One-class,
  singular and unstable/separated fits are marked non-estimable.
- Calibration-in-the-large is a separate fit with slope fixed at 1. Probabilities
  are clipped to [1e-8,1-1e-8] only to evaluate diagnostic logits, not to change
  stored predictions. sklearn handles endpoint log-loss arithmetic numerically.

## Bootstrap design

There are 2,000 paired stratified resamples, seed 72. Draw 60 bad and 140 good
holdout rows separately with replacement and reuse exactly those indices for all
models. Report percentile 2.5% and 97.5% intervals for AUC, AP, Brier and log loss,
plus prespecified boosting-minus-logistic and calibrated-minus-raw differences.
The public routine additionally supports an ordinary bootstrap and counts rejected
one-class draws; an initially single-class sample is explicitly rejected.

These intervals condition on the fitted models and observed class mixture. They
exclude refitting, model selection, source prevalence uncertainty, temporal change
and unidentified borrower dependence. Baseline intervals collapse under stratified
resampling because every draw has the same prevalence. The comparisons are nominal,
not multiplicity-adjusted; no global superiority or equivalence conclusion follows.
Overlapping marginal intervals are not used to infer the significance of a difference.

## Prior-probability scenarios

From Bayes' theorem, source posterior odds are

    p_s(x)/(1-p_s(x)) = [f_s(x|Y=1)/f_s(x|Y=0)] * pi_s/(1-pi_s).

If the conditional densities are invariant across source and target, replace the
source prior odds with target prior odds:

    multiplier = [pi_t/(1-pi_t)] / [pi_s/(1-pi_s)]
    p_t(x) = multiplier*p_s(x) / [1-p_s(x) + multiplier*p_s(x)].

The assumptions include a calibrated source posterior, a stable target event,
unchanged class-conditional feature distributions and a known target prior. The
scenarios posit 5%, 10%, 20% and 30%; none estimates the source bank's prevalence.
The mapping preserves ordering, changes absolute probabilities and cannot correct
covariate/outcome shifts that violate these assumptions. No population performance
metrics are inferred by scoring adjusted predictions against the enriched holdout.

## Interpretation, provenance and testing

Raw-feature permutation importance uses outer validation rows, ten shuffles per
feature per fold, scored by increase in log loss. Error bars show SD across outer
fold means, not confidence intervals. This reduces the optimism of in-sample
importance but does not remove correlation or implausible-combination problems.

`metrics.json` contains summaries; row-level CSVs permit independent recalculation.
Every report number reads these outputs. The runtime manifest records package,
source, data and protocol hashes. The final manifest hashes figure/report outputs.
Tests cover the entire requested behaviour, including a spy estimator that asserts
disjoint fit and scored row IDs throughout calibration and its inner searches.
PDF text, figure presence, page count and hashes are checked programmatically;
rendered pages are also inspected visually before delivery.
