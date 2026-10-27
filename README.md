# Beyond AUC: Credit Risk Probability Modelling, Calibration, and Model Validation

*An Empirical Comparison of Interpretable Statistical Models and Machine Learning Under Sampling Bias and Limited Data*

An independent quantitative research project comparing interpretable logistic regression with shallow-tree gradient boosting on corrected UCI South German Credit data. The focus is reliable probabilities, uncertainty and model risk, with an executed study and a 16-page research report.

**Research question:** Does extra model complexity produce a defensible improvement in ranking and probability quality, and what can a historical, case-enriched sample establish about lending risk?

- Built a typed, modular Python research pipeline with a locked 80/20 split and nested five-fold validation.
- Cross-fitted complete hyperparameter searches before sigmoid calibration, comparing the identical base model before and after its mapping.
- Calculated proper scoring rules, calibration diagnostics and 2,000 paired stratified bootstrap replicates.
- Executed fold stability and hypothetical prevalence-shift analyses, with 14 original figures in PNG and SVG.
- Preserved source checksums, split manifests, predictions, tuning results and a tested report-generation workflow.

## Measured findings

Boosting achieves holdout ROC-AUC 0.790 versus 0.749 for logistic regression, a difference of +0.040 (paired 95% CI [0.004, 0.077]). Yet boosting leads in only 1 of 5 training-only outer folds. The evidence supports a holdout-specific advantage, not a stable general preference for complexity.

| Locked holdout (200 records) | ROC-AUC | Average precision | Brier | Log loss |
| --- | ---: | ---: | ---: | ---: |
| Constant prevalence | 0.500 | 0.300 | 0.210 | 0.611 |
| Logistic regression | 0.749 | 0.574 | 0.175 | 0.525 |
| Gradient boosting | 0.790 | 0.635 | 0.162 | 0.494 |
| Boosting + sigmoid | 0.790 | 0.635 | 0.162 | 0.494 |

Mean outer-fold AUC is **0.790 for logistic** versus **0.775 for boosting**. Calibration changes holdout Brier score by -0.00006 and log loss by +0.00049; both paired intervals include zero. Its outer-fold log loss improves in only 1/5 folds.

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
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
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
