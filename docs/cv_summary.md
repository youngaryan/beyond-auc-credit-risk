# CV and interview summary

## A. One-line project title

Beyond AUC: Credit Risk Probability Modelling, Calibration and Statistical Validation

## B. Two CV bullets

- Built a reproducible Python credit-risk study with nested cross-validation, leakage-controlled sigmoid calibration, proper scoring rules and 2,000 paired bootstrap replicates on 1,000 public credit records.
- Compared logistic regression and gradient boosting, finding holdout AUCs of 0.749 and 0.790, while documenting contradictory fold rankings, calibration uncertainty and prevalence-shift assumptions.

## C. Approximately 45-second explanation

I chose this project because credit-risk models need useful probabilities, not just a good ranking score. I built a reproducible Python study comparing logistic regression, boosting and calibrated boosting, with nested validation and paired bootstrap uncertainty. Boosting reached holdout AUC 0.790 versus 0.749 for logistic regression, but logistic won most training-validation folds. Calibration produced almost no benefit. That taught me to distinguish a favourable split from a robust modelling conclusion. I also derived a prevalence adjustment to show why this deliberately oversampled historical dataset cannot establish real portfolio default probabilities. With more data, I would prioritise a defined default horizon, representative sampling and forward-time external validation.

## D. Three quantitative interview questions

**1. Why can AUC stay constant while calibration changes?**

AUC is the probability that a randomly drawn bad outcome has a higher score than a good outcome, with half credit for ties. A strictly increasing transformation preserves every ordering, so it preserves AUC. Calibration concerns absolute probability levels. Our sigmoid mapping preserves the boosting ranking and changes probabilities; proper scoring rules can therefore change even when AUC and average precision do not. Brier score and log loss assess probability quality, but also reflect discrimination. A constant prevalence predictor can be calibrated in the aggregate and useless at ranking.

**2. Where could leakage enter, and how did you prevent it?**

Global imputation/scaling, tuning on evaluation folds and fitting a calibrator to in-sample predictions can all leak information. We reserve a stratified 200-record holdout, fit preprocessing inside pipelines, and evaluate entire workflows in five outer training folds. Each calibration score is produced by a full five-fold hyperparameter search fitted without that row. The final raw boosting comparator is exactly the base estimator inside the calibrated model. A spy-estimator test asserts that no score uses its own fit rows. The test set is used only after settings are frozen; diagnostic recalibration is descriptive and never updates predictions.

**3. Can 30% bad credit be used as the bank's default rate?**

No. The sampling deliberately enriches bad outcomes, and contract non-compliance is not automatically a modern default event with a defined horizon. Under label shift, stable P(X|Y), and a calibrated source posterior, target odds equal source odds times [pi_t/(1-pi_t)] / [pi_s/(1-pi_s)]. For a hypothetical 10% target prior, a source probability of 0.5 becomes 0.206. This illustrates sensitivity; it does not estimate the bank's true prior, correct changing conditional distributions, or establish deployable PD calibration. The bootstrap likewise conditions on the observed class mix and excludes prevalence uncertainty.

## Evidence boundaries to defend

The paired holdout AUC difference is +0.040, with conditional 95% interval [0.004, 0.077]. That supports an advantage on this fixed partition; the opposing outer-CV pattern prevents a universal superiority claim. Do not describe calibration as a demonstrated improvement or imply bank sponsorship, real-lending deployment, fairness certification or academic novelty.
