# Prespecified research protocol

Written before running any model experiment, 27 October 2026. This is a local
prespecification, not an externally registered protocol. Its SHA-256 is saved
in the execution manifest. No model or threshold is selected using the holdout.

**Question:** Does gradient boosting improve credit-risk discrimination and
probability quality over logistic regression, with enough evidence to justify
its complexity?

## Hypotheses and evidence

- **H1:** Trained models discriminate better than a constant prevalence predictor.
  Compare AUC to 0.5, training-only outer CV and conditional holdout bootstrap CIs.
- **H2:** Boosting may improve discrimination, but the gain may be small or uncertain.
  Use paired AUC differences and signs across outer folds. An interval spanning
  zero leaves direction unresolved; do not infer equivalence or a winner.
- **H3:** Ranking and calibration are distinct. Compare AUC with reliability,
  Brier score, log loss, and calibration intercept/slope. Proper scores also
  reflect discrimination and are not pure calibration measures.
- **H4:** Sigmoid calibration may help or harm. Compare a single fitted boosting
  model before/after its sigmoid mapping, including paired scoring-rule CIs.
- **H5:** Calibration in a case-enriched historical sample does not establish
  population PD calibration. Evaluate explicit prior-shift scenarios and the
  assumptions required; population claims are not identifiable from this sample.

## Locked choices

- Original corrected UCI archive 522 plus its codebook; recode `kredit == 0` as
  `bad_credit = 1`. Validate 1,000 rows, 20 features, 300 bad and 700 good.
- Seed 42, stratified 80/20 split. Full-data access before splitting is limited
  to integrity checks and documented outcome counts. Exploratory associations,
  distributions and interpretation use training data or its outer validation folds.
- Only duration, transformed amount and age are continuous. All other variables,
  including discretised ordinal quantities, use reference-coded one-hot encoding
  with documented category levels. Numerical imputation/scaling occurs in each fit.
- Outer training-only 5-fold stratified CV (seed 43) evaluates entire workflows.
  Each workflow tunes inside 5 folds (seed 52), using mean negative log loss.
- Logistic L2 regularisation: C in {0.1, 1, 10}. Boosting: 120 trees,
  learning rate in {0.05, 0.1}, depth in {1, 2}, minimum leaf size 15.
  No class weights or resampling: the target is the study-sample probability.
- Sigmoid calibration: five cross-fitting folds (seed 62), `ensemble=False`.
  The estimator being cross-fitted is the **entire hyperparameter search**, so a
  calibration observation is excluded from preprocessing, fitting and tuning of
  its own score. The final full-training search estimator supplies the identical
  raw probabilities for the uncalibrated comparator. No isotonic selection.
- Report ROC-AUC, average precision (not trapezoidal PR area), Brier score,
  log loss, sensitivity/recall, specificity and precision. Threshold = 0.5,
  illustrative only. Undefined precision is reported as null, not as success.
- Reliability: five fixed equal-width bins, counts and descriptive Wilson 95%
  intervals. Joint diagnostic logistic recalibration intercept/slope and
  offset-only calibration-in-the-large are diagnostics, not holdout refits.
- Holdout: 2,000 paired stratified bootstrap resamples (seed 72), retaining 60
  bad / 140 good observations. Percentile 95% intervals, conditional on fitted
  models and observed case mix; no claim to training or population uncertainty.
  Key contrasts: boosting minus logistic, calibrated minus raw boosting.
- Stability: five outer-fold results; no t-tests treating overlapping training
  sets as independent. No repeated search over holdout seeds.
- Hypothetical prevalences 5%, 10%, 20%, 30%; adjust odds under label shift only.
  No estimate of the source bank's actual prevalence is attempted.
- Interpret logistic coefficients versus explicit reference levels and numerical
  training SDs; raw-feature permutation importance on outer validation folds.
  Demographics are retained for historical benchmark fidelity and discussed,
  without fairness or deployment claims.

No significance threshold or economically material gain is claimed without
institution-specific decision costs and an external validation population.
