# Research decisions and accepted trade-offs

| Decision | Rationale | Alternative and accepted limitation |
| --- | --- | --- |
| Corrected archive plus codebook | Preserve the original target and category semantics and verify bytes | `ucimlrepo` is convenient; original files make metadata discrepancies auditable |
| Fixed stratified holdout | Simple untouched final evaluation and known class counts | Temporal or grouped splitting would be preferable with dates/borrower IDs, which are absent |
| Nested outer validation | Measure the adaptive workflow without evaluating a globally selected grid winner on its selection folds | More fitting, and outer fit sizes are smaller than the final model |
| Search inside calibration cross-fitting | Exclude calibration labels from hyperparameter selection for their own scores | Ordinary calibration of an already globally tuned estimator risks a selection leak |
| Shared full-data boosting estimator | Isolate the calibration mapping | `ensemble=True` mixes ensembling and calibration effects |
| Log loss for tuning | Align selection with probability estimation | An AUC search could rank well but give less reliable probabilities; log loss is sensitive to confident errors |
| Small search | Restrict adaptive flexibility for limited data | A larger search or additional model families might improve scores but increase selection risk |
| Categorical treatment of ordinal bins | Avoid unsupported equal-distance and linearity assumptions | Loses order information and adds coefficients; schema levels and references are fixed |
| L2 logistic regularisation | Stabilise a many-indicator small-data model | Shrinks coefficients; reference-coded penalisation is not invariant to reference changes |
| No class balancing | Preserve the study-sample probability target | Balancing may change ranking/threshold behaviour but needs a separate correction and estimand |
| Sigmoid calibration | Two-parameter mapping with less flexibility than isotonic | Misspecified mapping may harm; cross-fit scores transfer from smaller models |
| Five reliability bins | Keep counts visible with a small holdout | Coarse resolution and binning-dependent ECE; pointwise intervals are descriptive |
| Paired stratified bootstrap | Compare correlated predictions while respecting the designed class mix | Conditions on models and prevalence; excludes training uncertainty and drift |
| Training-only permutation importance | Avoid using holdout outcomes for feature selection or iteration | Correlated predictors share importance; shuffled profiles may be unrealistic |
| Fixed illustrative threshold | Make sensitivity/specificity interpretable without tuning a policy | No decision-cost or deployment justification is possible |
| Hypothetical prior shifts | Expose the relation between sampling and posterior odds | Does not recover actual prevalence or fix changes in P(X|Y) |
| Retain demographics with discussion | Faithfully benchmark the historical source and expose concerns | Does not endorse such features; unreliable sex coding and sparse groups prevent fairness certification |
| ReportLab PDF with explicit pages | Portable generation without TeX, browser or external conversion services | Layout requires visual QA; Markdown is generated from the same narrative structure |
| Saved predictions, no fitted pickle | Retain sufficient evidence to audit results and refit deterministically | Predictions cannot serve new cases; deployment is outside this research scope |

## Findings did not trigger method changes

The holdout and training-only folds disagree on the preferred model; this is
reported directly. The near-null calibration result did not trigger a switch to
isotonic calibration, a larger search, altered threshold or new holdout seed.
Presentation repairs and numerical endpoint handling do not change model settings.

## Scope deliberately not expanded

No dashboard, server, LLM component, employer data, invented cost matrix or new
credit-decision policy was added. Optional controlled feature perturbations and
additional calibration methods were omitted after completing the two prespecified
robustness analyses. External/temporal validation cannot be manufactured from this
dataset; it remains a limitation and a next research step, not an unfinished run.
